# XAUUSD Daily Breakout Backtester (Python, M1)

This program reproduces the **exact strategy** from your MT5 EA:
- Daily Buy Stop at previous day's high, Sell Stop at previous day's low
- Replace pending orders at the start of each new day
- Initial SL/TP in *pips* (pip = `0.10` for gold by default)
- Trailing: activate at +X pips, lock to +Y pips, trail by Z pips with step S

## Data Requirements
Provide **1-minute** CSV with columns: `time,open,high,low,close,volume`.

## Quick Start
```bash
python xau_daily_breakout_backtester.py \
  --data /path/to/M1_XAUUSD.csv \
  --symbol XAUUSD \
  --start 2019-01-01 --end 2025-09-11 \
  --out ./reports \
  --grid small
```

Outputs (per run):
- `report.json` – MT5-like sections: General Info, Performance, Trade Stats, Balance & Equity, Deal Details, Modeling Quality, Monthly.
- `trades.csv` – full deal ledger.
- `equity_curve.png`, `balance_curve.png` – charts.
- `monthly.csv` – month-level equity stats.

## Parameter Grids
Two modes:
- `--grid small` – a small subset for sanity checks.
- `--grid all` – **exhaustive** grid per your spec (enormous).

Ranges used in `--grid all`:
- `InitialSL_Pips`: 30–80 (step 10)
- `InitialTP_Pips`: 30–3500 (step 10)
- `TrailActivate_Pips`: 5–40 (step 1)
- `TrailLock_Pips`: 1–15 (step 1)
- `TrailOffset_Pips`: 5–40 (step 1)
- `TrailStep_Pips`: 5–30 (step 1)
- `UseBufferOnStops`: true/false
- `StopBuffer_Pips`: 0 (only if UseBuffer=false) or 1–30 if true
- `DeviationPoints`: 10–80 (step 1)
- `VerboseLogs`: false

> ⚠️ The full grid is **astronomical**. Start with `--grid small` to validate your data and pipeline, then move to `--grid all` in batches.

## Notes & Assumptions
- Modeling quality approximates **M1 OHLC**. For *every-tick* modeling, use tick data & a tick path simulator.
- PnL per $1 move is modeled as `$100 * lots`, typical for gold. Adjust in `_position_value_per_point` for your broker if different.
- Commission is applied at close as a flat per-lot round-turn amount (`--commission-per-lot`).
