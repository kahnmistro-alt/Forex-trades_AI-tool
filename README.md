Pattern Trader – CNN-LSTM Forex Signal System
A production‑ready forex trading signal system powered by a hybrid CNN‑LSTM deep learning model.
The system detects chart patterns, generates trade signals (BUY/SELL/HOLD) with confidence scores, and supports automated execution via MetaTrader 4/5.

Overview
This system is designed to be a complete pipeline for algorithmic forex trading. It:

Ingests real‑time OHLCV data from yfinance or Twelve Data.

Engineers a rich set of technical, macro‑economic, and commodity features.

Uses a CNN‑LSTM neural network to capture both geometric chart patterns (head & shoulders, triangles, flags) and temporal dependencies (price momentum, reversal sequences).

Applies confidence calibration (Platt scaling) to produce probabilities in the range of 0.7–0.9.

Outputs actionable signals with stop‑loss and take‑profit levels based on volatility.

Logs all trades to Supabase for performance monitoring and adaptive retraining.

Provides a web dashboard for monitoring signals, confidence, and trade history.

Note: This version uses only the CNN‑LSTM model. All fallback systems (SVM, HMM, rule‑based candlestick/chart pattern recognisers) have been removed. The system relies entirely on deep learning for signal generation.

Features
✅ CNN‑LSTM hybrid model – detects both spatial and temporal patterns in forex data.

✅ Real‑time data ingestion – OHLCV from yfinance or Twelve Data.

✅ Feature engineering – 30+ technical indicators + macro (bond spreads, VIX) + commodities.

✅ Confidence calibration – Platt scaling for reliable probability estimates.

✅ Automated trade execution – integrated with MT4/MT5 via PyTrader.

✅ Trade logging & performance tracking – Supabase database.

✅ Periodic retraining – model updates every 6 hours with fresh data.

✅ Web dashboard – built with Flask, JavaScript, and CSS.

✅ Backtesting framework – walk‑forward validation to compare strategies.

Architecture
text
┌─────────────────────┐
│   Data Ingestion    │
│ (yfinance / Twelve) │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Feature Engineering │
│  (pandas_ta + macro) │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   CNN‑LSTM Model    │
│  (TensorFlow/Keras) │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Confidence         │
│  Calibration        │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Signal Decision    │
│  (BUY/SELL/HOLD)    │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Execution Layer    │
│  (MT4/MT5 via       │
│   PyTrader)         │
└─────────────────────┘
The system is modular – each component can be modified or replaced independently.