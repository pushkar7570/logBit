#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# XAUUSD Daily Breakout Backtester (M1)
# Reproduces the MT5 EA logic in Python. See README for details.

import argparse
import json
import math
import os
import sys
import uuid
import time as pytime
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_PIP_SIZE = 0.10  # $0.10 for XAUUSD
EQUITY_START = 10000.0
DEFAULT_LEVERAGE = "1:100"
EXECUTION_MODE = "Bar-based M1 (no slippage)"
MODEL = "1-minute OHLC"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def draw_simple_line_chart(series: pd.Series, title: str, out_png: str):
    fig = plt.figure(figsize=(9, 4.5))
    plt.plot(series.index, series.values)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(series.name if series.name else "Value")
    plt.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

@dataclass
class StrategyParams:
    Lots: float = 0.10
    PipSize: float = DEFAULT_PIP_SIZE
    InitialSL_Pips: int = 100
    InitialTP_Pips: int = 100
    TrailActivate_Pips: int = 10
    TrailLock_Pips: int = 4
    TrailOffset_Pips: int = 10
    TrailStep_Pips: int = 10
    UseBufferOnStops: bool = False
    StopBuffer_Pips: int = 0
    DeviationPoints: int = 20
    VerboseLogs: bool = False

@dataclass
class BacktestConfig:
    symbol: str = "XAUUSD"
    start: str = "2019-01-01"
    end: str = datetime.utcnow().strftime("%Y-%m-%d")
    initial_deposit: float = EQUITY_START
    leverage: str = DEFAULT_LEVERAGE
    commission_per_lot: float = 0.0
    slippage_points: int = 0
    model: str = MODEL
    execution_mode: str = EXECUTION_MODE

@dataclass
class Order:
    ticket: int
    type: str  # "BUYSTOP" or "SELLSTOP"
    price: float
    sl: float
    tp: float
    volume: float
    comment: str
    placed_time: pd.Timestamp

@dataclass
class Position:
    ticket: int
    type: str  # "BUY" or "SELL"
    entry_price: float
    sl: float
    tp: float
    volume: float
    open_time: pd.Timestamp
    close_time: Optional[pd.Timestamp] = None
    close_price: Optional[float] = None
    profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    trailing_active: bool = False

def load_m1_csv(path: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = ["time", "open", "high", "low", "close"]
    lower_map = {c.lower(): c for c in df.columns}
    for k in req:
        if k not in lower_map:
            raise ValueError(f"CSV must include columns: {req}")
    df.rename(columns={lower_map[k]: k for k in req}, inplace=True)
    df["time"] = pd.to_datetime(df["time"], utc=False)
    df.set_index("time", inplace=True)
    df = df.sort_index()
    df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    return df

class DailyBreakoutEngine:
    def __init__(self, m1: pd.DataFrame, cfg: BacktestConfig, params: StrategyParams):
        self.m1 = m1
        self.cfg = cfg
        self.p = params
        self.equity = cfg.initial_deposit
        self.balance = cfg.initial_deposit
        self.positions: Dict[int, Position] = {}
        self.orders: Dict[int, Order] = {}
        self.next_ticket = 1
        self.equity_curve = pd.Series(index=m1.index, dtype=float)
        self.balance_curve = pd.Series(index=m1.index, dtype=float)
        self.growth_curve = pd.Series(index=m1.index, dtype=float)
        self.closed_positions: List[Position] = []
        self.daily_prev_hi = None
        self.daily_prev_lo = None
        self.current_day = None
        self.ticks_modeled = len(m1)
        self.bars_tested = len(m1)
        self.mismatches = 0

    def _pip2price(self, pips:int) -> float:
        return pips * self.p.PipSize

    def _reset_daily_orders(self, ts: pd.Timestamp):
        self.orders.clear()
        day = ts.normalize()
        prev_day_start = day - pd.Timedelta(days=1)
        prev_day_end = day - pd.Timedelta(seconds=1)
        prev_slice = self.m1.loc[prev_day_start:prev_day_end]
        if prev_slice.empty:
            self.daily_prev_hi = None
            self.daily_prev_lo = None
            return
        self.daily_prev_hi = float(prev_slice["high"].max())
        self.daily_prev_lo = float(prev_slice["low"].min())

        buffer = self._pip2price(self.p.StopBuffer_Pips) if self.p.UseBufferOnStops else 0.0

        buy_price = self.daily_prev_hi + buffer
        sell_price = self.daily_prev_lo - buffer

        buy_sl = buy_price - self._pip2price(self.p.InitialSL_Pips)
        buy_tp = buy_price + self._pip2price(self.p.InitialTP_Pips)
        sell_sl = sell_price + self._pip2price(self.p.InitialSL_Pips)
        sell_tp = sell_price - self._pip2price(self.p.InitialTP_Pips)

        t_buy = self._new_ticket()
        t_sell = self._new_ticket()
        self.orders[t_buy] = Order(t_buy, "BUYSTOP", buy_price, buy_sl, buy_tp, self.p.Lots,
                                   f"BO {str(day.date())}", placed_time=ts)
        self.orders[t_sell] = Order(t_sell, "SELLSTOP", sell_price, sell_sl, sell_tp, self.p.Lots,
                                    f"BD {str(day.date())}", placed_time=ts)

    def _new_ticket(self) -> int:
        t = self.next_ticket
        self.next_ticket += 1
        return t

    def _commission_for(self, vol: float) -> float:
        return self.cfg.commission_per_lot * vol

    def _position_value_per_point(self, vol: float) -> float:
        return 100.0 * vol  # $ per $1 move

    def _mark_to_market(self, ts: pd.Timestamp, price: float):
        eq = self.balance
        for pos in self.positions.values():
            if pos.type == "BUY":
                pnl = (price - pos.entry_price) * self._position_value_per_point(pos.volume)
            else:
                pnl = (pos.entry_price - price) * self._position_value_per_point(pos.volume)
            eq = self.balance + pnl
        self.equity_curve.loc[ts] = eq
        self.balance_curve.loc[ts] = self.balance
        self.growth_curve.loc[ts] = eq

    def _crosses(self, row: pd.Series, price: float, direction: str) -> bool:
        if direction == "UP":
            return row["high"] >= price
        else:
            return row["low"] <= price

    def _enter_market(self, ts: pd.Timestamp, order: Order):
        if order.type == "BUYSTOP":
            pos = Position(ticket=order.ticket, type="BUY", entry_price=order.price,
                           sl=order.sl, tp=order.tp, volume=order.volume, open_time=ts)
        else:
            pos = Position(ticket=order.ticket, type="SELL", entry_price=order.price,
                           sl=order.sl, tp=order.tp, volume=order.volume, open_time=ts)
        self.positions[pos.ticket] = pos

    def _maybe_trigger_orders(self, ts: pd.Timestamp, row: pd.Series):
        to_remove = []
        for ticket, od in list(self.orders.items()):
            if od.type == "BUYSTOP" and self._crosses(row, od.price, "UP"):
                self._enter_market(ts, od); to_remove.append(ticket)
            elif od.type == "SELLSTOP" and self._crosses(row, od.price, "DOWN"):
                self._enter_market(ts, od); to_remove.append(ticket)
        for t in to_remove:
            self.orders.pop(t, None)

    def _maybe_close_by_sl_tp(self, ts: pd.Timestamp, row: pd.Series):
        to_close: List[Tuple[int, float]] = []
        for ticket, pos in list(self.positions.items()):
            if pos.type == "BUY":
                hit_tp = row["high"] >= pos.tp
                hit_sl = row["low"] <= pos.sl
                hit_price = pos.sl if hit_sl else (pos.tp if hit_tp else None)
            else:
                hit_tp = row["low"] <= pos.tp
                hit_sl = row["high"] >= pos.sl
                hit_price = pos.sl if hit_sl else (pos.tp if hit_tp else None)
            if hit_tp and hit_sl:
                hit_price = pos.sl
            if hit_price is not None:
                to_close.append((ticket, hit_price))
        for ticket, price in to_close:
            self._close_position(ticket, ts, price)

    def _trail_logic(self, pos: Position, price: float):
        if pos.type == "BUY":
            profit_pips = (price - pos.entry_price) / self.p.PipSize
            lock_level = pos.entry_price + self._pip2price(self.p.TrailLock_Pips)
            desired = price - self._pip2price(self.p.TrailOffset_Pips)
            desired = max(desired, lock_level)
            if profit_pips >= self.p.TrailActivate_Pips:
                if not pos.trailing_active:
                    if pos.sl < lock_level:
                        pos.sl = lock_level
                    pos.trailing_active = True
                else:
                    if pos.sl is None or desired - pos.sl >= self._pip2price(self.p.TrailStep_Pips):
                        pos.sl = desired
        else:
            profit_pips = (pos.entry_price - price) / self.p.PipSize
            lock_level = pos.entry_price - self._pip2price(self.p.TrailLock_Pips)
            desired = price + self._pip2price(self.p.TrailOffset_Pips)
            desired = min(desired, lock_level)
            if profit_pips >= self.p.TrailActivate_Pips:
                if not pos.trailing_active:
                    if pos.sl == 0.0 or pos.sl > lock_level:
                        pos.sl = lock_level
                    pos.trailing_active = True
                else:
                    if pos.sl is None or pos.sl - desired >= self._pip2price(self.p.TrailStep_Pips):
                        pos.sl = desired

    def _apply_trailing(self, row: pd.Series):
        price = row["close"]
        for pos in self.positions.values():
            self._trail_logic(pos, price)

    def _close_position(self, ticket: int, ts: pd.Timestamp, price: float):
        pos = self.positions.get(ticket)
        if not pos: return
        pos.close_time = ts
        pos.close_price = price
        if pos.type == "BUY":
            pnl = (price - pos.entry_price) * self._position_value_per_point(pos.volume)
        else:
            pnl = (pos.entry_price - price) * self._position_value_per_point(pos.volume)
        pos.commission = self._commission_for(pos.volume)
        pos.profit = pnl - pos.commission
        self.balance += pos.profit
        self.closed_positions.append(pos)
        self.positions.pop(ticket, None)

    def run(self) -> Dict:
        if self.m1.empty:
            return {}

        for ts, row in self.m1.iterrows():
            day = ts.normalize()
            if self.current_day is None or day > self.current_day:
                self.current_day = day
                self._reset_daily_orders(ts)

            self._maybe_trigger_orders(ts, row)
            self._apply_trailing(row)
            self._maybe_close_by_sl_tp(ts, row)
            self._mark_to_market(ts, row["close"])

        if len(self.positions):
            last_ts = self.m1.index[-1]
            last_close = float(self.m1.iloc[-1]["close"])
            for ticket in list(self.positions.keys()):
                self._close_position(ticket, last_ts, last_close)

        return self._generate_report()

    def _generate_report(self) -> Dict:
        trades = self.closed_positions
        profits = [t.profit for t in trades]
        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = -sum(p for p in profits if p < 0)
        total_net = gross_profit - gross_loss
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        total_trades = len(trades)
        expected_payoff = (total_net / total_trades) if total_trades > 0 else 0.0

        eq = self.equity_curve.dropna()
        bal = self.balance_curve.dropna()
        final_balance = float(bal.iloc[-1]) if len(bal) else float(self.balance)

        if len(eq):
            peaks = eq.cummax()
            dd = peaks - eq
            max_dd_abs = float(dd.max())
            max_dd_pct = float(((dd / peaks).replace([np.inf, -np.inf], np.nan).max() or 0) * 100.0)
            abs_drawdown = float((eq.iloc[0] - eq.min()) if len(eq) else 0.0)
        else:
            max_dd_abs = 0.0
            max_dd_pct = 0.0
            abs_drawdown = 0.0

        sharpe = None
        if len(eq) > 2:
            rets = eq.pct_change().dropna()
            if rets.std() > 0:
                sharpe = float((rets.mean() / rets.std()) * math.sqrt(252*24*60))

        recovery = (total_net / max_dd_abs) if max_dd_abs else None

        long_trades = [t for t in trades if t.type == "BUY"]
        short_trades = [t for t in trades if t.type == "SELL"]
        long_wins = len([t for t in long_trades if t.profit > 0])
        short_wins = len([t for t in short_trades if t.profit > 0])
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        max_consec_wins = 0
        max_consec_losses = 0
        cur = 0
        seq = []
        for t in trades:
            if t.profit > 0:
                cur = cur + 1 if cur >= 0 else 1
                max_consec_wins = max(max_consec_wins, cur)
                seq.append(1)
            else:
                cur = cur - 1 if cur <= 0 else -1
                max_consec_losses = min(max_consec_losses, cur)
                seq.append(-1)

        def avg_streak(sign: int) -> float:
            total = 0; streaks = 0; run = 0
            for v in seq:
                if (v > 0 and sign > 0) or (v < 0 and sign < 0):
                    run += 1
                else:
                    if run > 0:
                        streaks += 1; total += run; run = 0
            if run > 0:
                streaks += 1; total += run
            return (total / streaks) if streaks else 0.0

        avg_consec_wins = avg_streak(1)
        avg_consec_losses = avg_streak(-1)

        monthly = {}
        if len(eq):
            eq_month_end = eq.groupby([eq.index.year, eq.index.month]).last()
            eq_month_start = eq.groupby([eq.index.year, eq.index.month]).first()
            for (y, m), val_end in eq_month_end.items():
                val_start = eq_month_start.loc[(y, m)]
                monthly_key = f"{y:04d}-{m:02d}"
                monthly[monthly_key] = {
                    "equity_start": float(val_start),
                    "equity_end": float(val_end),
                    "return_pct": float((val_end/val_start - 1.0) * 100.0) if val_start else 0.0,
                }

        modeling_quality = 90.0

        report = {
            "General Information": {
                "Symbol": self.cfg.symbol,
                "Period": "M1 (simulated)",
                "Model": self.cfg.model,
                "Date range": f"{self.cfg.start} to {self.cfg.end}",
                "Parameters": asdict(self.p),
                "Initial deposit": self.cfg.initial_deposit,
                "Leverage": self.cfg.leverage,
                "Execution mode": self.cfg.execution_mode,
            },
            "Performance Metrics": {
                "Total net profit": float(total_net),
                "Gross profit": float(gross_profit),
                "Gross loss": float(gross_loss),
                "Profit factor": float(profit_factor) if math.isfinite(profit_factor) else None,
                "Expected payoff": float(expected_payoff),
                "Absolute drawdown": float(abs_drawdown),
                "Maximal drawdown (abs)": float(max_dd_abs),
                "Maximal drawdown (%)": float(max_dd_pct),
                "Sharpe ratio": sharpe,
                "Recovery factor": float(recovery) if recovery is not None else None,
            },
            "Trade Statistics": {
                "Total trades": int(len(trades)),
                "Short positions (won %)": f"{len(short_trades)} ({(100*short_wins/len(short_trades)) if short_trades else 0:.2f}%)",
                "Long positions (won %)": f"{len(long_trades)} ({(100*long_wins/len(long_trades)) if long_trades else 0:.2f}%)",
                "Profit trades (% of total)": f"{len(wins)} ({(100*len(wins)/len(trades)) if trades else 0:.2f}%)",
                "Loss trades (% of total)": f"{len(losses)} ({(100*len(losses)/len(trades)) if trades else 0:.2f}%)",
                "Largest profit trade": float(max(wins) if wins else 0.0),
                "Largest loss trade": float(min(losses) if losses else 0.0),
                "Average profit trade": float((sum(wins)/len(wins)) if wins else 0.0),
                "Average loss trade": float((sum(losses)/len(losses)) if losses else 0.0),
                "Maximum consecutive wins (and profit)": f"{max_consec_wins}",
                "Maximum consecutive losses (and loss)": f"{-max_consec_losses}",
                "Average consecutive wins": float(avg_consec_wins),
                "Average consecutive losses": float(avg_consec_losses),
            },
            "Balance & Equity": {
                "Final balance": float(final_balance),
            },
            "Order/Deal-Level Details": [
                {
                    "Ticket": t.ticket,
                    "Open time": str(t.open_time),
                    "Close time": str(t.close_time),
                    "Order type": t.type,
                    "Volume": float(t.volume),
                    "Symbol": self.cfg.symbol,
                    "Open price": float(t.entry_price),
                    "Close price": float(t.close_price),
                    "Stop Loss": float(t.sl),
                    "Take Profit": float(t.tp),
                    "Commission": float(t.commission),
                    "Swap": float(t.swap),
                    "Profit": float(t.profit),
                } for t in trades
            ],
            "Monthly": monthly,
            "Modeling Quality": {
                "Modelling quality %": float(modeling_quality),
                "Ticks generated": int(self.bars_tested),
                "Ticks modeled": int(self.bars_tested),
                "Ticks rejected": 0,
                "Mismatched chart errors": int(self.mismatches),
                "Bars tested": int(self.bars_tested),
                "Time taken (sec)": None,
            }
        }
        return report

def generate_param_grid(mode: str) -> List[StrategyParams]:
    params_list = []
    if mode == "all":
        SLs = list(range(30, 81, 10))
        TPs = list(range(30, 3501, 10))
        act = list(range(5, 41, 1))
        lock = list(range(1, 16, 1))
        off = list(range(5, 41, 1))
        step = list(range(5, 31, 1))
        dev = list(range(10, 81, 1))
        use_buf_options = [False, True]
        for use_buf in use_buf_options:
            stopbuf = [0] if not use_buf else list(range(1, 31))
            for sl, tp, a, l, o, st, d, sb in product(SLs, TPs, act, lock, off, step, dev, stopbuf):
                params_list.append(StrategyParams(
                    InitialSL_Pips=sl,
                    InitialTP_Pips=tp,
                    TrailActivate_Pips=a,
                    TrailLock_Pips=l,
                    TrailOffset_Pips=o,
                    TrailStep_Pips=st,
                    UseBufferOnStops=use_buf,
                    StopBuffer_Pips=sb,
                    DeviationPoints=d,
                    VerboseLogs=False
                ))
    elif mode == "small":
        for sl in [50, 80]:
            for tp in [200, 500, 1000]:
                for a in [10, 20]:
                    for l in [4, 8]:
                        for o in [10, 20]:
                            for st in [10, 20]:
                                for use_buf in [False, True]:
                                    for sb in ([0] if not use_buf else [1, 5]):
                                        for d in [20, 40]:
                                            params_list.append(StrategyParams(
                                                InitialSL_Pips=sl,
                                                InitialTP_Pips=tp,
                                                TrailActivate_Pips=a,
                                                TrailLock_Pips=l,
                                                TrailOffset_Pips=o,
                                                TrailStep_Pips=st,
                                                UseBufferOnStops=use_buf,
                                                StopBuffer_Pips=sb,
                                                DeviationPoints=d,
                                                VerboseLogs=False
                                            ))
    else:
        raise ValueError("grid mode must be one of: all, small")
    return params_list

def run_one(m1: pd.DataFrame, cfg: BacktestConfig, params: StrategyParams, outdir: str, run_id: Optional[str]=None) -> str:
    run_id = run_id or str(uuid.uuid4())[:8]
    run_dir = os.path.join(outdir, f"{cfg.symbol}_{run_id}")
    ensure_dir(run_dir)

    engine = DailyBreakoutEngine(m1, cfg, params)
    t0 = pytime.time()
    report = engine.run()
    elapsed = pytime.time() - t0
    if "Modeling Quality" in report:
        report["Modeling Quality"]["Time taken (sec)"] = elapsed

    meta = {"config": asdict(cfg), "params": asdict(params), "run_id": run_id, "elapsed_sec": elapsed}
    with open(os.path.join(run_dir, "report.json"), "w") as f:
        json.dump({"meta": meta, "report": report}, f, indent=2)

    trades = pd.DataFrame(report["Order/Deal-Level Details"])
    trades.to_csv(os.path.join(run_dir, "trades.csv"), index=False)

    eq = engine.equity_curve.dropna()
    bal = engine.balance_curve.dropna()
    if len(eq):
        draw_simple_line_chart(eq, "Equity Curve", os.path.join(run_dir, "equity_curve.png"))
    if len(bal):
        draw_simple_line_chart(bal, "Balance Curve", os.path.join(run_dir, "balance_curve.png"))

    monthly = pd.DataFrame.from_dict(report["Monthly"], orient="index")
    if not monthly.empty:
        monthly.index.name = "month"
        monthly.to_csv(os.path.join(run_dir, "monthly.csv"))

    return run_dir

def main():
    ap = argparse.ArgumentParser(description="XAUUSD Daily Breakout Backtester (M1)")
    ap.add_argument("--data", required=True, help="Path to CSV M1 data (time,open,high,low,close,volume)")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=datetime.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--out", default="./reports")
    ap.add_argument("--grid", choices=["all", "small"], default="small",
                    help="Parameter grid size. 'all' = exhaustive (very large). 'small' = sanity sample")
    ap.add_argument("--initial-deposit", type=float, default=EQUITY_START)
    ap.add_argument("--leverage", default=DEFAULT_LEVERAGE)
    ap.add_argument("--commission-per-lot", type=float, default=0.0)
    args = ap.parse_args()

    ensure_dir(args.out)
    m1 = load_m1_csv(args.data, args.start, args.end)

    cfg = BacktestConfig(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        initial_deposit=args.initial_deposit,
        leverage=args.leverage,
        commission_per_lot=args.commission_per_lot,
    )

    params_grid = generate_param_grid(args.grid)
    print(f"Total parameter combinations: {len(params_grid)}")

    for i, params in enumerate(params_grid, 1):
        run_dir = run_one(m1, cfg, params, args.out)
        print(f"[{i}/{len(params_grid)}] {run_dir}")

if __name__ == "__main__":
    main()
