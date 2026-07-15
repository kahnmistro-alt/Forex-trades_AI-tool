📈 Multi‑Pair Auto ARIMA Forex Signals
A professional Flask web application that generates live trading signals for multiple currency pairs using Auto ARIMA time series forecasting, DTW cycle similarity, and ATR‑based risk management (fixed 3:1 reward:risk). The dashboard is fully responsive – works on both desktop and mobile devices.

https://via.placeholder.com/800x400?text=ARIMA+Trader+Dashboard
(Screenshot placeholder – your actual UI will look similar)

✨ Features
✅ Multi‑pair support – Monitor any number of Forex pairs (Yahoo Finance symbols like EURUSD=X)

✅ Auto ARIMA per pair – Each pair gets its own optimally tuned ARIMA model (cached for performance)

✅ Cycle pattern matching – Uses Dynamic Time Warping (DTW) to find similar historical patterns

✅ 3:1 Risk:Reward – Fixed reward‑to‑risk ratio with dynamic ATR stops

✅ Live signals – BUY, SELL, or HOLD with confidence score

✅ Copy‑paste ready trade plans – One‑click copy of TP/SL levels

✅ Interactive charts – Plotly charts with signal markers

✅ Responsive design – Works on mobile, tablet, and desktop

✅ Caching – ARIMA models cached to avoid retraining on every refresh

🛠️ Technology Stack
Backend: Flask (Python)

Data: yfinance (Yahoo Finance)

Forecasting: pmdarima (Auto ARIMA)

Pattern matching: dtaidistance (DTW), scikit‑learn

Frontend: HTML5, CSS3, JavaScript, Plotly.js

Icons: Font Awesome

📦 Installation
1. Clone or download the project
Place the four files in the following structure:

text
forex_signals/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
2. Create a virtual environment (recommended)
bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
3. Install dependencies
bash
pip install -r requirements.txt
requirements.txt content:

text
flask==2.3.3
yfinance==0.2.28
pandas==2.0.3
numpy==1.24.3
pmdarima==2.0.4
scikit-learn==1.3.0
dtaidistance==2.3.10
plotly==5.17.0
4. Run the app
bash
python app.py
Open your browser and go to: http://127.0.0.1:5000

🚀 How to Use
Set your currency pairs – In the sidebar, enter Yahoo Finance symbols (e.g., EURUSD=X, GBPUSD=X). Separate multiple pairs with commas.

Choose interval – 1h (1 hour) or 1d (daily).

Adjust ATR period – Controls the sensitivity of the stop loss (typical 14).

Set risk in ATR units – Stop loss distance = ATR × risk_mult. Reward is always 3× risk.

Click “Refresh Signals” – The app will:

Download historical data (last year for training, current year for recent)

Add technical features (ATR, volatility)

Train an Auto ARIMA model for each pair (cached)

Compute cycle similarity via DTW

Generate signals (BUY/SELL/HOLD)

Read the summary table – Shows all pairs with signal, confidence, TP/SL levels.

Copy trade plans – For BUY/SELL signals, click Copy Plan and paste into your trading platform.

Inspect charts – Expand any pair to see price action + signal arrow.

⚙️ How It Works (Brief)
Training data: Previous full calendar year (e.g., 2025‑01‑01 to 2025‑12‑31)

Recent data: Current year up to today

ARIMA: Auto‑selects the best (p,d,q)(P,D,Q)m parameters using AIC

Cycle signal: DTW compares the last 48 bars against the training history; similarity score influences final signal

Final decision: Weighted combination of ARIMA forecast return and cycle similarity → BUY if >0.7, SELL if <‑0.7, else HOLD

Risk management: Stop loss = ATR × risk_mult, Take profit = Stop loss × 3

📱 Responsive Design
Desktop: Two‑column layout with sticky settings sidebar

Tablet / Mobile: Settings stack on top, tables scroll horizontally, trade cards become vertical

Touch‑friendly buttons and readable font sizes

🧪 Customisation
Adding more pairs
Simply edit the textarea in the sidebar. Example:

text
EURUSD=X, GBPUSD=X, AUDJPY=X, XAUUSD=X
Note: Use Yahoo Finance symbols. For Forex, append =X (e.g., EURUSD=X). For commodities, use GC=F (gold) or CL=F (oil), but ensure yfinance supports them.

Changing the risk/reward ratio
In app.py, change the reward_ratio variable (currently hardcoded to 3.0). You can also expose it in the frontend if desired.

Adjusting ARIMA parameters
Inside get_or_train_model() in app.py, modify the auto_arima arguments (e.g., max_p, max_q, seasonal, m).

Cache duration
Data is cached by yfinance (5 minutes). Model cache lasts until the underlying data changes (hash‑based). To force retraining, restart the Flask server.

🐞 Troubleshooting
Issue	Solution
No data returned	Check your internet connection. Ensure the pair symbol is correct (e.g., EURUSD=X).
'Series' object has no attribute 'tobytes'	This error has been fixed in the provided app.py. Use the latest version.
Models retrain every time	The cache uses a hash of the training series. If data changes (e.g., new day), the hash changes and retraining occurs. This is intended.
Slow first load	Training ARIMA for multiple pairs can take 20‑60 seconds. Subsequent refreshes are much faster (cached).
📄 License
This project is for educational purposes only. Trading Forex involves substantial risk. Past performance does not guarantee future results. Use at your own risk.

🙌 Acknowledgements
pmdarima – Auto ARIMA

dtaidistance – DTW

yfinance – Free market data

Happy Trading!
Always backtest and manage your risk.