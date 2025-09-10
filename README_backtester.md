# XAUUSD Daily Breakout Backtester (Python, M1) — with Online Data

Now supports **automatic data fetching** via:
- **Dukascopy** (`pip install dukascopy`) — robust minute history for FX/CFDs (use for XAUUSD 2019+)
- **Yahoo Finance** (`pip install yfinance`) — intraday minute data (limited historical depth)

## Quick Start (auto source)
```bash
pip install pandas numpy matplotlib yfinance dukascopy
python xau_daily_breakout_backtester.py \
  --source auto \
  --symbol XAUUSD \
  --start 2019-01-01 --end 2025-09-11 \
  --out ./reports \
  --grid small
```

If both online sources fail or you prefer your own data, use CSV:
```bash
python xau_daily_breakout_backtester.py \
  --source csv \
  --data /path/to/M1_XAUUSD.csv \
  --start 2019-01-01 --end 2025-09-11 \
  --out ./reports \
  --grid small
```

Everything else (strategy rules, parameter grid, and MT5-style reports) remains **exactly the same** as before.
