# PatternTrader – ML-Powered Forex Trading System

A self‑learning algorithmic trading system that combines rule‑based pattern detection with a Random Forest classifier to generate high‑confidence forex signals and execute trades automatically via MetaTrader 4.

## Features

- **Machine Learning** – Random Forest model trained on OHLCV + technical indicators (RSI, MACD, ATR). 83% test accuracy.
- **Auto‑Retraining** – retrains every 10 minutes with fresh data; model persisted to Supabase as base64 blob.
- **Hybrid Signal Logic** – rule‑based detection (candlestick patterns, support/resistance, breakouts) combined with ML predictions.
- **Live MT4 Integration** – uses PyTrader EA to fetch real‑time bid/ask and execute trades.
- **Adaptive Confidence** – min confidence threshold adjusts based on win rate (feedback loop via Supabase).
- **Persistent Storage** – trades, config, and ML models stored in Supabase (PostgreSQL).

## Quick Start

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt