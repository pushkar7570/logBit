#!/usr/bin/env python3
"""
Daily Breakout/Breakdown strategy backtester on minute data (generic asset)
---------------------------------------------------------------------------

This script implements, in Python, the MT5 EA logic you provided for XAUUSD,
translated to a generic backtester that works on any asset supported by
Yahoo Finance via `yfinance`. It:

- Downloads intraday OHLCV data (1m by default; auto-fallback to 5m or 15m
  and even alternate tickers like GC=F if needed for gold when Yahoo blocks
  some FX symbols).
- Simulates the D1 breakout strategy at minute granularity:
    * Each server day (data timezone), place two pending orders at the
      previous day's High (Buy Stop) and Low (Sell Stop), with optional buffer.
    * Unfilled pendings are canceled at the next day open; new ones are
      placed based on the new previous-day H/L.
    * If one side triggers, the other remains pending and may also fill.
    * Initial SL/TP in *pips* using a configurable pip size (default 0.10).
    * Trailing: activate at +10 pips, lock to +4 pips, then trail by 10 pips
      with a 10-pip step.
- Produces an MT5-style Excel report with:
    * Parameters sheet (includes **Actual Data Symbol** & **Actual Interval** used)
    * Trades sheet (all deals with per-trade P&L)
    * EquityCurve sheet (with an embedded chart)
    * Daily summary sheet
    * Metrics sheet (net/gross, PF, drawdowns, Sharpe, win rate, etc.)
- Saves raw candles used for the backtest and the Excel report under `reports/`
  with a unique filename including asset, timeframe, and date range.

Notes about data source & reliability:
- Yahoo Finance offers ~7 days of 1-minute history. If your requested period
  exceeds that, the script automatically falls back to 5m (or 15m) so your
  backtest still runs.
- Some FX symbols (e.g., `XAUUSD=X`) intermittently fail on intraday due to
  timezone metadata (YFTzMissingError). This script now **automatically tries
  alternate gold tickers** like `XAU=X`, `GC=F`, `MGC=F` unless you disable it.

Usage examples:
    python xau_d1_breakout_minute_backtest.py \
        --symbol XAUUSD=X --period 5d --interval 1m

    # If the above fails for your region, the script will try XAU=X, GC=F, MGC=F

    python xau_d1_breakout_minute_backtest.py \
        --symbol GC=F --start 2025-09-01 --end 2025-09-07 --interval 1m

Requirements (add these to requirements.txt):
    pandas
    numpy
    yfinance
    xlsxwriter

Author: ChatGPT for Pushkar Singh
"""

from __future__ import annotations
import argparse
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ================================ Strategy Inputs =============================
@dataclass
class StrategyInputs:
    symbol: str
    lots: float = 0.10
    pip_size: float = 0.10  # "pip" in price units (e.g., $0.10 for XAUUSD)
    initial_sl_pips: int = 100
    initial_tp_pips: int = 100
    trail_activate_pips: int = 10
    trail_lock_pips: int = 4
    trail_offset_pips: int = 10
    trail_step_pips: int = 10
    use_buffer_on_stops: bool = False
    stop_buffer_pips: int = 0
    verbose: bool = True
    interval: str = "1m"  # default desired interval
    tz: Optional[str] = None  # if None, use data tz from Yahoo
    data_symbol: Optional[str] = None  # actual symbol used after download

# Known aliases for tricky symbols (Yahoo quirks)
SYMBOL_ALIASES: Dict[str, List[str]] = {
    'XAUUSD=X': ['XAUUSD=X', 'XAU=X', 'GC=F', 'MGC=F'],
    'XAU=X': ['XAU=X', 'XAUUSD=X', 'GC=F', 'MGC=F'],
}

# Preferred fallback intervals (progressively coarser)
INTERVAL_FALLBACKS = ['1m', '5m', '15m']

# ================================ Utilities ==================================

def log(msg: str, verbose: bool = True):
    if verbose:
        print(f"[XAU-D1-BO] {msg}")


def pip_to_price(pips: float, pip_size: float) -> float:
    return pips * pip_size


def price_to_pips(price_diff: float, pip_size: float) -> float:
    return price_diff / pip_size


# ============================== Data Acquisition =============================

def _infer_period_from_dates(start: Optional[str], end: Optional[str]) -> Optional[str]:
    if not start or not end:
        return None
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    days = (e - s).days
    if days <= 0:
        return None
    return f"{days}d"


def _ensure_tz(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index.tz is None:
        # Localize naive intraday timestamps to UTC if Yahoo omitted tz
        df.index = df.index.tz_localize('UTC')
    return df


def _download_once(symbol: str, start: Optional[str], end: Optional[str], period: Optional[str], interval: str) -> pd.DataFrame:
    df = yf.download(tickers=symbol, start=start, end=end, period=period,
                     interval=interval, auto_adjust=False, progress=False, threads=False, prepost=True)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'Adj Close': 'adj_close', 'Volume': 'volume'
        })
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = df[~df.index.duplicated(keep='first')].sort_index()
        df = _ensure_tz(df)
        df.attrs['interval'] = interval
        return df
    return pd.DataFrame()


def download_data(symbol: str,
                  start: Optional[str] = None,
                  end: Optional[str] = None,
                  period: Optional[str] = None,
                  interval: str = "1m",
                  verbose: bool = True,
                  try_aliases: bool = True,
                  extra_alts: Optional[List[str]] = None) -> Tuple[str, pd.DataFrame]:
    """Download OHLCV using yfinance with robust fallbacks.

    Returns (actual_symbol_used, DataFrame with tz-aware index).
    """
    # If period not provided but dates are, infer one for better yfinance behavior
    if period is None:
        period = _infer_period_from_dates(start, end)

    candidates: List[str] = [symbol]
    # Add built-in aliases
    if try_aliases and symbol in SYMBOL_ALIASES:
        for alt in SYMBOL_ALIASES[symbol]:
            if alt not in candidates:
                candidates.append(alt)
    # Add user-provided extra alts
    if extra_alts:
        for alt in extra_alts:
            if alt not in candidates:
                candidates.append(alt)

    last_err: Optional[Exception] = None

    for sym in candidates:
        for ivl in ([interval] + [i for i in INTERVAL_FALLBACKS if i != interval]):
            try:
                log(f"Trying {sym} at {ivl}...", verbose)
                df = _download_once(sym, start, end, period, ivl)
                if not df.empty:
                    log(f"Downloaded {len(df)} bars for {sym} at {ivl}. TZ={df.index.tz}", verbose)
                    return sym, df
                else:
                    log(f"No data for {sym} at {ivl}.", verbose)
            except Exception as e:
                last_err = e
                log(f"Error for {sym} at {ivl}: {e}", verbose)
                time.sleep(0.7)  # gentle backoff

    # If all attempts failed
    raise RuntimeError(
        f"Failed to download intraday data for {symbol}. "
        f"Tried symbols: {candidates}. Last error: {last_err}"
    )


# ============================== Backtest Engine ===============================
from dataclasses import dataclass as _dataclass

@_dataclass
class PendingOrder:
    side: str  # 'BUY_STOP' or 'SELL_STOP'
    price: float
    sl: float
    tp: float
    placed_at: pd.Timestamp
    comment: str

@_dataclass
class Position:
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    sl: float
    tp: float
    entry_time: pd.Timestamp
    lots: float
    active: bool = True
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    reason: Optional[str] = None  # 'SL'|'TP'
    trail_activated: bool = False

@_dataclass
class TradeRecord:
    ticket: int
    type: str  # 'Buy'/'Sell'
    lots: float
    open_time: pd.Timestamp
    open_price: float
    sl: float
    tp: float
    close_time: pd.Timestamp
    close_price: float
    commission: float
    swap: float
    profit: float
    profit_pips: float
    duration: str
    comment: str


class BreakoutBacktester:
    def __init__(self, params: StrategyInputs, data: pd.DataFrame):
        self.p = params
        self.df = data.copy()
        if self.p.tz:
            self.df.index = self.df.index.tz_convert(self.p.tz)
        self.interval = self.df.attrs.get('interval', self.p.interval)
        self.trades: List[TradeRecord] = []
        self.ticket_seq = 1
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.balance = 0.0  # PnL-only (no initial deposit notion)
        self.positions: List[Position] = []
        self.pending: List[PendingOrder] = []

    def _prev_day_hl(self, day: pd.Timestamp) -> Optional[Tuple[float, float]]:
        day_date = day.normalize()
        prev_start = day_date - pd.Timedelta(days=1)
        prev_end = day_date - pd.Timedelta(seconds=1)
        slice_df = self.df.loc[(self.df.index >= prev_start) & (self.df.index <= prev_end)]
        if slice_df.empty:
            return None
        return float(slice_df['high'].max()), float(slice_df['low'].min())

    def _cancel_all_pendings(self):
        self.pending.clear()

    def _place_daily_pendings(self, day_start: pd.Timestamp):
        hl = self._prev_day_hl(day_start)
        if hl is None:
            return
        prev_high, prev_low = hl
        buffer = pip_to_price(self.p.stop_buffer_pips, self.p.pip_size) if self.p.use_buffer_on_stops else 0.0
        buy_stop = prev_high + buffer
        sell_stop = prev_low - buffer
        buy_sl = buy_stop - pip_to_price(self.p.initial_sl_pips, self.p.pip_size)
        buy_tp = buy_stop + pip_to_price(self.p.initial_tp_pips, self.p.pip_size)
        sell_sl = sell_stop + pip_to_price(self.p.initial_sl_pips, self.p.pip_size)
        sell_tp = sell_stop - pip_to_price(self.p.initial_tp_pips, self.p.pip_size)
        ts = day_start
        self.pending = [
            PendingOrder('BUY_STOP', buy_stop, buy_sl, buy_tp, ts, f"BuyStop prevD1H {prev_high:.5f}"),
            PendingOrder('SELL_STOP', sell_stop, sell_sl, sell_tp, ts, f"SellStop prevD1L {prev_low:.5f}")
        ]

    @staticmethod
    def _bar_first_then_second_path(bar_open: float, bar_close: float) -> Tuple[str, str]:
        return ("HIGH", "LOW") if bar_close >= bar_open else ("LOW", "HIGH")

    def _close_position(self, pos: Position, when: pd.Timestamp, price: float, reason: str):
        pos.active = False
        pos.exit_time = when
        pos.exit_price = price
        pos.reason = reason
        pnl = (price - pos.entry_price) * (1 if pos.side == 'LONG' else -1) * pos.lots
        self.balance += pnl
        duration = str(pos.exit_time - pos.entry_time)
        profit_pips = price_to_pips((price - pos.entry_price) * (1 if pos.side == 'LONG' else -1), self.p.pip_size)
        tr = TradeRecord(
            ticket=self.ticket_seq,
            type='Buy' if pos.side == 'LONG' else 'Sell',
            lots=pos.lots,
            open_time=pos.entry_time,
            open_price=pos.entry_price,
            sl=pos.sl,
            tp=pos.tp,
            close_time=when,
            close_price=price,
            commission=0.0,
            swap=0.0,
            profit=pnl,
            profit_pips=profit_pips,
            duration=duration,
            comment=reason,
        )
        self.ticket_seq += 1
        self.trades.append(tr)

    def _maybe_trail(self, pos: Position, bar_time: pd.Timestamp, bar_high: float, bar_low: float, bar_close: float):
        if pos.side == 'LONG':
            profit_pips = price_to_pips(bar_close - pos.entry_price, self.p.pip_size)
        else:
            profit_pips = price_to_pips(pos.entry_price - bar_close, self.p.pip_size)
        if profit_pips < self.p.trail_activate_pips:
            return
        if not pos.trail_activated:
            if pos.side == 'LONG':
                lock_sl = pos.entry_price + pip_to_price(self.p.trail_lock_pips, self.p.pip_size)
                pos.sl = max(pos.sl, lock_sl)
            else:
                lock_sl = pos.entry_price - pip_to_price(self.p.trail_lock_pips, self.p.pip_size)
                pos.sl = min(pos.sl, lock_sl)
            pos.trail_activated = True
        if pos.side == 'LONG':
            desired = bar_close - pip_to_price(self.p.trail_offset_pips, self.p.pip_size)
            desired = max(desired, pos.entry_price + pip_to_price(self.p.trail_lock_pips, self.p.pip_size))
            if (desired - pos.sl) >= pip_to_price(self.p.trail_step_pips, self.p.pip_size):
                pos.sl = desired
        else:
            desired = bar_close + pip_to_price(self.p.trail_offset_pips, self.p.pip_size)
            desired = min(desired, pos.entry_price - pip_to_price(self.p.trail_lock_pips, self.p.pip_size))
            if (pos.sl - desired) >= pip_to_price(self.p.trail_step_pips, self.p.pip_size):
                pos.sl = desired

    def run(self) -> None:
        idx = self.df.index
        if len(idx) == 0:
            raise RuntimeError("No data to backtest.")
        current_day = idx[0].normalize()
        self._place_daily_pendings(idx[0])
        last_equity_time = None
        for ts, row in self.df.iterrows():
            if ts.normalize() != current_day:
                self._cancel_all_pendings()
                self._place_daily_pendings(ts)
                current_day = ts.normalize()
            o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
            first, second = self._bar_first_then_second_path(o, c)
            triggered: List[Tuple[int, PendingOrder]] = []
            for i, po in enumerate(self.pending):
                if po.side == 'BUY_STOP':
                    if h >= po.price:
                        triggered.append((i, po))
                else:
                    if l <= po.price:
                        triggered.append((i, po))
            for i, po in sorted(triggered, key=lambda x: x[0], reverse=True):
                self.pending.pop(i)
                if po.side == 'BUY_STOP':
                    pos = Position('LONG', po.price, po.sl, po.tp, ts, self.p.lots)
                else:
                    pos = Position('SHORT', po.price, po.sl, po.tp, ts, self.p.lots)
                self.positions.append(pos)
                for phase in [first, second]:
                    if not pos.active:
                        break
                    if pos.side == 'LONG':
                        if phase == 'HIGH' and h >= pos.tp:
                            self._close_position(pos, ts, pos.tp, 'TP')
                        elif phase == 'LOW' and l <= pos.sl:
                            self._close_position(pos, ts, pos.sl, 'SL')
                    else:
                        if phase == 'LOW' and l <= pos.tp:
                            self._close_position(pos, ts, pos.tp, 'TP')
                        elif phase == 'HIGH' and h >= pos.sl:
                            self._close_position(pos, ts, pos.sl, 'SL')
            for pos in [p for p in self.positions if p.active]:
                if pos.side == 'LONG':
                    for phase in [first, second]:
                        if not pos.active:
                            break
                        if phase == 'HIGH' and h >= pos.tp:
                            self._close_position(pos, ts, pos.tp, 'TP')
                        elif phase == 'LOW' and l <= pos.sl:
                            self._close_position(pos, ts, pos.sl, 'SL')
                else:
                    for phase in [first, second]:
                        if not pos.active:
                            break
                        if phase == 'LOW' and l <= pos.tp:
                            self._close_position(pos, ts, pos.tp, 'TP')
                        elif phase == 'HIGH' and h >= pos.sl:
                            self._close_position(pos, ts, pos.sl, 'SL')
            for pos in [p for p in self.positions if p.active]:
                self._maybe_trail(pos, ts, h, l, c)
            if (last_equity_time is None) or (ts - last_equity_time >= pd.Timedelta(minutes=1)):
                self.equity_curve.append((ts, self.balance))
                last_equity_time = ts

    def trades_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=[
                'Ticket', 'Type', 'Lots', 'Open Time', 'Open Price', 'SL', 'TP',
                'Close Time', 'Close Price', 'Commission', 'Swap', 'Profit', 'Profit (pips)',
                'Duration', 'Comment'
            ])
        rows = []
        for tr in self.trades:
            rows.append([
                tr.ticket, tr.type, tr.lots, tr.open_time, tr.open_price, tr.sl, tr.tp,
                tr.close_time, tr.close_price, tr.commission, tr.swap, tr.profit,
                tr.profit_pips, tr.duration, tr.comment
            ])
        df = pd.DataFrame(rows, columns=[
            'Ticket', 'Type', 'Lots', 'Open Time', 'Open Price', 'SL', 'TP',
            'Close Time', 'Close Price', 'Commission', 'Swap', 'Profit', 'Profit (pips)',
            'Duration', 'Comment'
        ])
        return df

    def equity_dataframe(self) -> pd.DataFrame:
        if not self.equity_curve:
            return pd.DataFrame(columns=['Time', 'Equity']).set_index('Time')
        return pd.DataFrame(self.equity_curve, columns=['Time', 'Equity']).set_index('Time')

    def daily_summary(self) -> pd.DataFrame:
        eq = self.equity_dataframe()
        if eq.empty:
            return pd.DataFrame(columns=['Date', 'Net PnL']).set_index('Date')
        daily = eq['Equity'].diff().fillna(0).groupby(eq.index.normalize()).sum()
        return daily.to_frame('Net PnL')

    def metrics(self) -> Dict[str, float]:
        trades = self.trades_dataframe()
        eq = self.equity_dataframe()
        metrics: Dict[str, float] = {}
        net_profit = float(trades['Profit'].sum()) if not trades.empty else 0.0
        gross_profit = float(trades.loc[trades['Profit'] > 0, 'Profit'].sum()) if not trades.empty else 0.0
        gross_loss = float(trades.loc[trades['Profit'] < 0, 'Profit'].sum()) if not trades.empty else 0.0
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else (np.inf if gross_profit > 0 else 0.0)
        total_trades = int(len(trades))
        wins = int((trades['Profit'] > 0).sum()) if not trades.empty else 0
        losses = int((trades['Profit'] < 0).sum()) if not trades.empty else 0
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        expected_payoff = (net_profit / total_trades) if total_trades > 0 else 0.0
        if not eq.empty:
            equity = eq['Equity'].values
            peaks = np.maximum.accumulate(equity)
            drawdowns = peaks - equity
            max_dd = float(np.max(drawdowns))
            peak_at = float(peaks[np.argmax(drawdowns)]) if len(peaks) else 0.0
            max_dd_pct = float(max_dd / peak_at) * 100.0 if peak_at != 0 else 0.0
            abs_dd = float(np.max(np.maximum(0, 0 - equity)))
        else:
            max_dd = 0.0
            max_dd_pct = 0.0
            abs_dd = 0.0
        recovery_factor = (net_profit / max_dd) if max_dd > 0 else np.inf
        if not trades.empty and trades['Profit'].std(ddof=1) > 0:
            sharpe = float(trades['Profit'].mean() / trades['Profit'].std(ddof=1) * math.sqrt(252))
        else:
            sharpe = 0.0
        long_trades = int((trades['Type'] == 'Buy').sum()) if not trades.empty else 0
        short_trades = int((trades['Type'] == 'Sell').sum()) if not trades.empty else 0
        avg_win = float(trades.loc[trades['Profit'] > 0, 'Profit'].mean()) if wins > 0 else 0.0
        avg_loss = float(trades.loc[trades['Profit'] < 0, 'Profit'].mean()) if losses > 0 else 0.0
        largest_win = float(trades['Profit'].max()) if not trades.empty else 0.0
        largest_loss = float(trades['Profit'].min()) if not trades.empty else 0.0
        def _max_streak(is_win_series: pd.Series, target: bool) -> int:
            max_streak = 0
            cur = 0
            for v in is_win_series:
                if bool(v) == target:
                    cur += 1
                    max_streak = max(max_streak, cur)
                else:
                    cur = 0
            return max_streak
        if not trades.empty:
            is_win = trades['Profit'] > 0
            max_consec_wins = _max_streak(is_win, True)
            max_consec_losses = _max_streak(is_win, False)
            avg_trade_duration_seconds = pd.to_timedelta(trades['Duration']).dt.total_seconds().mean()
        else:
            max_consec_wins = 0
            max_consec_losses = 0
            avg_trade_duration_seconds = 0.0
        metrics.update({
            'Net Profit': net_profit,
            'Gross Profit': gross_profit,
            'Gross Loss': gross_loss,
            'Profit Factor': profit_factor,
            'Total Trades': total_trades,
            'Win Rate %': win_rate,
            'Expected Payoff': expected_payoff,
            'Absolute Drawdown': abs_dd,
            'Max Drawdown': max_dd,
            'Max Drawdown %': max_dd_pct,
            'Recovery Factor': recovery_factor,
            'Sharpe Ratio': sharpe,
            'Long Trades': long_trades,
            'Short Trades': short_trades,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Largest Win': largest_win,
            'Largest Loss': largest_loss,
            'Max Consecutive Wins': max_consec_wins,
            'Max Consecutive Losses': max_consec_losses,
            'Average Trade Duration (s)': avg_trade_duration_seconds,
        })
        return metrics


# ============================== Excel Reporting ==============================

def write_excel_report(out_path: str,
                       params: StrategyInputs,
                       trades_df: pd.DataFrame,
                       equity_df: pd.DataFrame,
                       daily_df: pd.DataFrame,
                       metrics: Dict[str, float],
                       candles_used: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine='xlsxwriter', datetime_format='yyyy-mm-dd hh:mm:ss', date_format='yyyy-mm-dd') as xw:
        param_items = [[k, getattr(params, k)] for k in vars(params).keys()]
        pd.DataFrame(param_items, columns=['Parameter', 'Value']).to_excel(xw, sheet_name='Parameters', index=False)
        pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']).to_excel(xw, sheet_name='Metrics', index=False)
        trades_df.to_excel(xw, sheet_name='Trades', index=False)
        eq = equity_df.copy().reset_index().rename(columns={'index': 'Time'})
        eq.to_excel(xw, sheet_name='EquityCurve', index=False)
        dd = daily_df.copy().reset_index().rename(columns={'index': 'Date'})
        dd.to_excel(xw, sheet_name='Daily', index=False)
        cd = candles_used.copy().reset_index().rename(columns={'index': 'Time'})
        cd.to_excel(xw, sheet_name='Candles', index=False)
        wb = xw.book
        ws = xw.sheets['EquityCurve']
        chart = wb.add_chart({'type': 'line'})
        last_row = len(eq) + 1
        chart.add_series({
            'name':       'Equity',
            'categories': ['EquityCurve', 1, 0, last_row, 0],
            'values':     ['EquityCurve', 1, 1, last_row, 1],
        })
        chart.set_title({'name': 'Equity Curve'})
        chart.set_x_axis({'name': 'Time'})
        chart.set_y_axis({'name': 'Equity'})
        ws.insert_chart('E2', chart)
    candles_csv = os.path.splitext(out_path)[0] + '_candles.csv'
    candles_used.to_csv(candles_csv)


# =============================== Main / CLI ================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Daily Breakout/Breakdown backtester on minute data.')
    ap.add_argument('--symbol', required=True, help='Primary symbol (e.g., XAUUSD=X or GC=F)')
    ap.add_argument('--start', help='Start date YYYY-MM-DD (optional if --period used)')
    ap.add_argument('--end', help='End date YYYY-MM-DD (optional if --period used)')
    ap.add_argument('--period', help='Alternative to start/end, e.g., 5d, 60d (works best for intraday)')
    ap.add_argument('--interval', default='1m', help='Desired data interval: 1m (<=7d) or 5m/15m')
    ap.add_argument('--alt', help='Comma-separated alternate symbols to try if the primary fails (e.g., XAU=X,GC=F,MGC=F)')
    ap.add_argument('--no-alias', action='store_true', help='Disable built-in alias fallbacks for known tricky symbols')

    # Strategy params
    ap.add_argument('--lots', type=float, default=0.10)
    ap.add_argument('--pip-size', type=float, default=0.10)
    ap.add_argument('--initial-sl', type=int, default=100)
    ap.add_argument('--initial-tp', type=int, default=100)
    ap.add_argument('--trail-activate', type=int, default=10)
    ap.add_argument('--trail-lock', type=int, default=4)
    ap.add_argument('--trail-offset', type=int, default=10)
    ap.add_argument('--trail-step', type=int, default=10)
    ap.add_argument('--use-buffer', action='store_true')
    ap.add_argument('--buffer-pips', type=int, default=0)
    ap.add_argument('--tz', default=None, help='Timezone to convert data to (e.g., "UTC", "America/New_York").')
    ap.add_argument('--quiet', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet
    symbol = args.symbol
    extra_alts = [s.strip() for s in args.alt.split(',')] if args.alt else None

    actual_symbol, df = download_data(
        symbol,
        start=args.start,
        end=args.end,
        period=args.period,
        interval=args.interval,
        verbose=verbose,
        try_aliases=(not args.no_alias),
        extra_alts=extra_alts,
    )

    # Build params (record actual symbol & interval used)
    p = StrategyInputs(
        symbol=symbol,
        lots=args.lots,
        pip_size=args.pip_size,
        initial_sl_pips=args.initial_sl,
        initial_tp_pips=args.initial_tp,
        trail_activate_pips=args.trail_activate,
        trail_lock_pips=args.trail_lock,
        trail_offset_pips=args.trail_offset,
        trail_step_pips=args.trail_step,
        use_buffer_on_stops=args.use_buffer,
        stop_buffer_pips=args.buffer_pips,
        verbose=verbose,
        interval=df.attrs.get('interval', args.interval),
        tz=args.tz,
        data_symbol=actual_symbol,
    )

    log(f"Data downloaded: {len(df)} bars, symbol_used={actual_symbol}, interval={p.interval}", verbose)

    bt = BreakoutBacktester(p, df)
    bt.run()

    trades_df = bt.trades_dataframe()
    equity_df = bt.equity_dataframe()
    daily_df = bt.daily_summary()
    metrics = bt.metrics()

    start_dt = df.index[0]
    end_dt = df.index[-1]
    start_str = pd.to_datetime(start_dt).strftime('%Y%m%d')
    end_str = pd.to_datetime(end_dt).strftime('%Y%m%d')
    safe_symbol = (actual_symbol or symbol).replace('^', '').replace('=', '').replace('/', '-')
    reports_dir = os.path.join(os.getcwd(), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    out_xlsx = os.path.join(reports_dir, f"Report_{safe_symbol}_{start_str}_{end_str}_{p.interval}.xlsx")

    write_excel_report(out_xlsx, p, trades_df, equity_df, daily_df, metrics, df)

    log(f"Report written to: {out_xlsx}", verbose)
    if trades_df.empty:
        log("No closed trades in the period. Consider extending the period or adjusting parameters.", verbose)


if __name__ == '__main__':
    main()
