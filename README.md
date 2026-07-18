Pattern Trader – AI-Powered Forex Signal System
An advanced, self‑improving Forex trading system that uses per‑pair machine learning models (XGBoost, Random Forest, SVM+HMM) to generate high‑confidence signals. Built with Flask, Supabase, and yfinance, it automatically retrains models weekly and supports live execution via MetaTrader (PyTrader).

🚀 Features
Pair‑specific models – each currency pair gets its own best‑performing ML model (XGBoost, RF, SVM+HMM, or rule‑based).

Automatic retraining – models are retrained every 7 days with fresh data, stored in Supabase.

Manual retrain – one‑click retraining from the web GUI.

Live price feeds – uses Twelve Data API (with fallback to yfinance and PyTrader).

Execution – integrates with MetaTrader via PyTrader (optional).

Web UI – clean dashboard to view signals, trade, copy trade plans, and toggle auto‑trading.

Backtested – rigorously tested on 2022–2025 data; best models selected per pair.

📊 Backtest Results (2022–2025, daily data)
Pair	Best Model	Win Rate	Total PnL	Val. Accuracy
AUDUSD	Rule	33.3%	+5.92%	–
EURUSD	XGBoost	42.9%	+6.86%	0.56
GBPUSD	Random Forest	57.1%	+10.82%	0.58
USDCAD	XGBoost	44.4%	+5.53%	0.47
Conclusion: One‑size‑fits‑all fails. Pair‑tailored models significantly improve performance.

🛠️ Tech Stack
Backend: Flask, Python 3.13

Data: yfinance, pandas, pandas_ta

ML: scikit‑learn, XGBoost, hmmlearn

Database: Supabase (PostgreSQL) – stores models and trade logs

Price Feeds: Twelve Data API (primary), yfinance (fallback)

Execution: PyTrader (MT4/MT5 integration) – optional

Deployment: Render (or any WSGI server)