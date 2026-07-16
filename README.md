# Pattern Trader – ML‑Enhanced Forex Trading System

A hybrid trading system that combines classical candlestick/chart patterns with SVM+HMM machine learning to generate high‑confidence signals and execute trades via MetaTrader 4.

## Key Features

- **Hybrid Signal Logic** – SVM classifies direction, HMM detects market regime; combined with rule‑based candlestick and chart patterns.
- **Adaptive Confidence** – minimum confidence threshold adjusts based on win rate (feedback loop via Supabase).
- **Trend Filter** – only trades in the direction of the 200‑period SMA.
- **Dynamic Confidence** – threshold adjusts with ATR volatility, avoiding trades during erratic periods.
- **Best‑of‑Breed Parameters** – tuned via walk‑forward backtesting (`risk_atr=1.0`, `min_confidence=0.6`, reward:risk = 3:1).
- **Focus on Top Pairs** – auto‑trade limited to USDJPY and GBPUSD (the best performers from backtesting).

## Backtest Insights

- **SVM+HMM** outperformed pure rule‑based detection on all tested pairs.
- **USDJPY** and **GBPUSD** showed the most consistent profitability over a 1‑year period.
- **Trend filter** and **dynamic confidence** reduced drawdowns significantly.
- **Optimal parameters** were found through a 9‑iteration walk‑forward grid search.

## Quick Start

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt