# backtest.py – with CNN-primary signal (rule as filter)
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hmmlearn import hmm
import pandas_datareader.data as web
from cnn_lstm_model import CNNLSTMClassifier
from candlestick_patterns import detect_candlestick_patterns, get_pattern_signal
from chart_patterns import detect_chart_patterns

# ---------- Configuration ----------
PAIRS = [
    'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X',
    'USDCHF=X', 'EURGBP=X', 'EURJPY=X', 'NZDUSD=X',
    'GBPJPY=X', 'USDJPY=X'
]
START_DATE = '2015-01-01'
END_DATE = '2025-07-16'
INTERVAL = '1d'
INITIAL_BALANCE = 10000
LOT_SIZE = 0.01
SPREAD = 0.0001
COMMISSION = 5.0

RISK_ATR = 1.0
MIN_CONFIDENCE = 0.7
REWARD_RATIO = 3.0

INITIAL_TRAIN_BARS = 1500
TEST_BARS = 300
STEP = 300

FEATURES = [
    'open', 'high', 'low', 'close',
    'open_prev_1', 'high_prev_1', 'low_prev_1', 'close_prev_1',
    'open_prev_2', 'high_prev_2', 'low_prev_2', 'close_prev_2',
    'rsi_14', 'macd', 'atr_14',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
    'roc_10', 'roc_20',
    'returns_std_10', 'returns_skew_10', 'returns_kurt_10',
    'stoch_k', 'stoch_d', 'williams_r', 'cci_20', 'adx_14',
    'volatility_20', 'volatility_ratio',
    'crude_oil', 'gold', 'agri',
    'spread_us_de', 'spread_us_uk', 'spread_us_au', 'spread_us_ca',
    'spread_us_ch', 'spread_de_uk', 'spread_de_jp', 'spread_us_nz',
    'spread_uk_jp', 'spread_us_jp',
    'vix'
]

_commodity_cache = {}
_macro_cache = {}

def fetch_commodity_data(start_date, end_date):
    global _commodity_cache
    cache_key = f"{start_date}_{end_date}"
    if cache_key in _commodity_cache:
        return _commodity_cache[cache_key]
    try:
        symbols = {'crude_oil': 'CL=F', 'gold': 'GC=F', 'agri': 'DBA'}
        combined = pd.DataFrame()
        for name, symbol in symbols.items():
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False, timeout=60)
                if not data.empty:
                    price_series = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
                    if isinstance(price_series, pd.DataFrame):
                        price_series = price_series.iloc[:, 0]
                    price_series.name = name
                    if combined.empty:
                        combined = price_series.to_frame()
                    else:
                        combined = combined.join(price_series, how='outer')
            except Exception as e:
                print(f"⚠️ Could not fetch {name}: {e}")
                combined[name] = np.nan
        if combined.empty:
            return pd.DataFrame()
        combined = combined.ffill().dropna(how='all')
        _commodity_cache[cache_key] = combined
        return combined
    except Exception as e:
        print(f"❌ Commodity fetch error: {e}")
        return pd.DataFrame()

def fetch_macro_data(start_date, end_date):
    global _macro_cache
    cache_key = f"{start_date}_{end_date}"
    if cache_key in _macro_cache:
        return _macro_cache[cache_key]
    try:
        fred_symbols = {
            'us10y': 'DGS10',
            'de10y': 'IRLTLT01DEM156N',
            'uk10y': 'IRLTLT01GBM156N',
            'au10y': 'IRLTLT01AUM156N',
            'ca10y': 'IRLTLT01CAM156N',
            'ch10y': 'IRLTLT01CHM156N',
            'nz10y': 'IRLTLT01NZM156N',
            'jp10y': 'IRLTLT01JPM156N',
        }
        combined = pd.DataFrame()
        for name, fred_code in fred_symbols.items():
            try:
                data = web.DataReader(fred_code, 'fred', start_date, end_date)
                data = data.rename(columns={fred_code: name})
                if combined.empty:
                    combined = data
                else:
                    combined = combined.join(data, how='outer')
            except Exception as e:
                print(f"⚠️ Could not fetch {name}: {e}")
                combined[name] = np.nan
        vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False, timeout=60)
        if not vix_data.empty:
            vix_series = vix_data['Adj Close'] if 'Adj Close' in vix_data.columns else vix_data['Close']
            if isinstance(vix_series, pd.DataFrame):
                vix_series = vix_series.iloc[:, 0]
            vix_series.name = 'vix'
            if combined.empty:
                combined = vix_series.to_frame()
            else:
                combined = combined.join(vix_series, how='outer')
        if combined.empty:
            return pd.DataFrame()
        required = ['us10y','de10y','uk10y','au10y','ca10y','ch10y','nz10y','jp10y','vix']
        for c in required:
            if c not in combined.columns:
                combined[c] = np.nan
        combined = combined.ffill().bfill()
        combined['spread_us_de'] = combined['us10y'] - combined['de10y']
        combined['spread_us_uk'] = combined['us10y'] - combined['uk10y']
        combined['spread_us_au'] = combined['us10y'] - combined['au10y']
        combined['spread_us_ca'] = combined['us10y'] - combined['ca10y']
        combined['spread_us_ch'] = combined['us10y'] - combined['ch10y']
        combined['spread_de_uk'] = combined['de10y'] - combined['uk10y']
        combined['spread_de_jp'] = combined['de10y'] - combined['jp10y']
        combined['spread_us_nz'] = combined['us10y'] - combined['nz10y']
        combined['spread_uk_jp'] = combined['uk10y'] - combined['jp10y']
        combined['spread_us_jp'] = combined['us10y'] - combined['jp10y']
        spread_cols = [c for c in combined.columns if c.startswith('spread_')]
        combined = combined.dropna(subset=spread_cols, how='all')
        _macro_cache[cache_key] = combined
        return combined
    except Exception as e:
        print(f"❌ Macro fetch error: {e}")
        return pd.DataFrame()

def fetch_data(pair, start, end, interval):
    print(f"Fetching {pair} {interval}...")
    tickers = [pair, pair.replace('=X', '')]
    data = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end, interval=interval,
                               progress=False, auto_adjust=False, timeout=60,
                               multi_level_index=False)
            if not data.empty:
                break
        except:
            continue
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [c.lower() for c in data.columns]
    required = ['open','high','low','close']
    for c in required:
        if c not in data.columns:
            data[c] = data.get('close', np.nan)
    return data[['open','high','low','close']]

def add_features(df):
    if df.empty or len(df) < 20:
        return pd.DataFrame()
    df = df.copy()
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9'] if macd is not None else 0
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    sma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = sma + 2*std
    df['bb_middle'] = sma
    df['bb_lower'] = sma - 2*std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_width'] = df['bb_width'].replace([np.inf, -np.inf], np.nan)
    df['roc_10'] = ta.roc(df['close'], length=10)
    df['roc_20'] = ta.roc(df['close'], length=20)
    ret = df['close'].pct_change()
    df['returns_std_10'] = ret.rolling(10).std()
    df['returns_skew_10'] = ret.rolling(10).skew()
    df['returns_kurt_10'] = ret.rolling(10).kurt()
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    if stoch is not None:
        df['stoch_k'] = stoch.get('STOCHk_14_3_3', np.nan)
        df['stoch_d'] = stoch.get('STOCHd_14_3_3', np.nan)
    else:
        df['stoch_k'] = df['stoch_d'] = np.nan
    df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
    df['cci_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx_14'] = adx['ADX_14'] if adx is not None else np.nan
    df['volatility_20'] = ret.rolling(20).std()
    df['volatility_ratio'] = df['volatility_20'] / df['volatility_20'].rolling(10).mean()
    df['volatility_ratio'] = df['volatility_ratio'].replace([np.inf, -np.inf], np.nan)
    for lag in [1,2]:
        df[f'open_prev_{lag}'] = df['open'].shift(lag)
        df[f'high_prev_{lag}'] = df['high'].shift(lag)
        df[f'low_prev_{lag}'] = df['low'].shift(lag)
        df[f'close_prev_{lag}'] = df['close'].shift(lag)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = df[c].ffill()
    df = df.replace([np.inf, -np.inf], np.nan)
    nan_frac = df[FEATURES].isna().mean()
    bad = nan_frac[nan_frac > 0.95].index.tolist()
    if bad:
        df = df.drop(columns=bad)
    usable = [c for c in FEATURES if c in df.columns]
    df = df.dropna(subset=usable)
    return df

def prepare_ml_data(df):
    cols = [c for c in FEATURES if c in df.columns]
    X = df[cols].values
    future_ret = df['close'].shift(-1) / df['close'] - 1
    y = np.where(future_ret > 0.001, 1, np.where(future_ret < -0.001, -1, 0))
    X = X[:-1]; y = y[:-1]
    return X, y, df.index[:-1]

def train_svm_hmm(X, y):
    mask = y != 0
    X_bin = X[mask]; y_bin = (y[mask] > 0).astype(int)
    if len(X_bin) < 50:
        return None, None, None, 0.0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bin)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bin, test_size=0.2, random_state=42)
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    X_all_scaled = scaler.transform(X)
    hmm_model = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=100, random_state=42)
    hmm_model.fit(X_all_scaled)
    return svm, hmm_model, scaler, acc

def predict_svm_hmm(df, svm, hmm_model, scaler):
    try:
        cols = [c for c in FEATURES if c in df.columns]
        X = df[cols].values[-1:].reshape(1, -1)
        X_scaled = scaler.transform(X)
        prob = svm.predict_proba(X_scaled)[0]
        pred = svm.predict(X_scaled)[0]
        conf = prob[1] if pred == 1 else prob[0]
        state_probs = hmm_model.predict_proba(X_scaled)[0]
        dominant = np.argmax(state_probs)
        if pred == 1:
            if dominant == 2: conf = min(1.0, conf+0.15)
            elif dominant == 0: conf = max(0.0, conf-0.15)
        else:
            if dominant == 0: conf = min(1.0, conf+0.15)
            elif dominant == 2: conf = max(0.0, conf-0.15)
        return 'BUY' if pred==1 else 'SELL', conf
    except:
        return 'HOLD', 0.0

def detect_trend(df, ma_long=200):
    if len(df) < ma_long:
        return 'neutral'
    sma = df['close'].rolling(ma_long).mean().iloc[-1]
    curr = df['close'].iloc[-1]
    if curr > sma: return 'uptrend'
    elif curr < sma: return 'downtrend'
    else: return 'neutral'

def compute_rule_signal(df, min_confidence=0.0):
    candle_patterns = detect_candlestick_patterns(df)
    candle_signal, candle_conf = get_pattern_signal(candle_patterns)
    chart_patterns = detect_chart_patterns(df, lookback=40) if len(df)>=40 else []
    trend = detect_trend(df, ma_long=200)
    signal = 'HOLD'; conf = 0.0
    if candle_signal != 'HOLD':
        signal = candle_signal; conf = candle_conf
        for pat in chart_patterns:
            if pat in ['double_bottom','inverse_head_shoulders','ascending_triangle'] and signal=='BUY':
                conf = min(1.0, conf+0.15)
            elif pat in ['double_top','head_shoulders','descending_triangle'] and signal=='SELL':
                conf = min(1.0, conf+0.15)
        if signal=='BUY' and trend=='uptrend': conf = min(1.0, conf+0.1)
        elif signal=='SELL' and trend=='downtrend': conf = min(1.0, conf+0.1)
    else:
        if chart_patterns:
            bullish = ['double_bottom','inverse_head_shoulders','ascending_triangle','falling_wedge']
            bearish = ['double_top','head_shoulders','descending_triangle','rising_wedge']
            b = sum(1 for p in chart_patterns if p in bullish)
            be = sum(1 for p in chart_patterns if p in bearish)
            if b > be:
                signal = 'BUY'; conf = 0.5 + 0.3*(b/(b+be))
            elif be > b:
                signal = 'SELL'; conf = 0.5 + 0.3*(be/(b+be))
            else:
                signal = 'HOLD'; conf = 0.0
            if signal=='BUY' and trend=='uptrend': conf = min(1.0, conf+0.2)
            elif signal=='SELL' and trend=='downtrend': conf = min(1.0, conf+0.2)
    if conf >= min_confidence and signal != 'HOLD':
        return signal, conf
    else:
        return 'HOLD', 0.0

def compute_hybrid_svmhmm(df, svm, hmm_model, scaler, min_confidence=0.7):
    rule_signal, rule_conf = compute_rule_signal(df, min_confidence=0.0)
    ml_signal, ml_conf = predict_svm_hmm(df, svm, hmm_model, scaler)
    # Original hybrid: rule dominates if present
    if rule_signal != 'HOLD' and ml_signal != 'HOLD' and rule_signal == ml_signal:
        conf = min(1.0, (rule_conf+ml_conf)/2 + 0.1)
        return rule_signal, conf
    elif rule_signal != 'HOLD':
        conf = rule_conf * 0.8
        return rule_signal, conf
    elif ml_signal != 'HOLD':
        conf = ml_conf * 0.8
        return ml_signal, conf
    else:
        return 'HOLD', 0.0

# ---------- CNN+LSTM helpers (with PRIMARY signal) ----------
def create_sequences(df, feature_cols, window_size=60):
    data = df[feature_cols].values
    close = df['close'].values
    X_seq, y_seq = [], []
    for i in range(window_size, len(data)):
        future_ret = close[i] / close[i-1] - 1
        if abs(future_ret) <= 0.001:
            continue
        X_seq.append(data[i-window_size:i])
        y_seq.append(1 if future_ret > 0 else 0)
    if len(X_seq) == 0:
        return None, None
    return np.array(X_seq), np.array(y_seq)

def train_cnn_lstm(df, feature_cols):
    X_seq, y_seq = create_sequences(df, feature_cols, window_size=60)
    if X_seq is None or len(X_seq) < 100:
        print(f"Not enough sequences for CNN+LSTM: {len(X_seq) if X_seq is not None else 0}")
        return None
    model = CNNLSTMClassifier(window_size=60, n_features=len(feature_cols),
                              epochs=50, batch_size=64, calibration=None)
    model.fit(X_seq, y_seq)
    return model

def predict_cnn_lstm(df, model, feature_cols):
    if len(df) < 60:
        return 'HOLD', 0.0
    last = df[feature_cols].iloc[-60:].values
    if np.isnan(last).any():
        return 'HOLD', 0.0
    last = last.reshape(1, 60, -1)
    prob = model.predict_proba(last)[0]
    up = prob[1]
    if up > 0.5:
        return 'BUY', up
    else:
        return 'SELL', 1-up

# NEW: CNN-primary with rule as filter
def compute_cnn_primary_signal(df, cnn_model, feature_cols, min_confidence=0.7):
    # Get ML signal (CNN)
    ml_signal, ml_conf = predict_cnn_lstm(df, cnn_model, feature_cols)
    
    # Get rule signal
    rule_signal, rule_conf = compute_rule_signal(df, min_confidence=0.0)
    
    # If ML is HOLD, we HOLD
    if ml_signal == 'HOLD':
        return 'HOLD', 0.0
    
    # If rule agrees, boost confidence and return ML signal
    if rule_signal == ml_signal:
        conf = min(1.0, (ml_conf + rule_conf) / 2 + 0.1)
        return ml_signal, conf
    
    # If rule disagrees (opposite signal)
    if rule_signal != 'HOLD' and rule_signal != ml_signal:
        # Only if ML confidence is high enough, still go with ML but penalize
        if ml_conf >= 0.6:
            conf = ml_conf * 0.85  # penalty for conflict
            return ml_signal, conf
        else:
            return 'HOLD', 0.0  # reject
    
    # If rule is HOLD (no strong opinion), go with ML with slight penalty
    conf = ml_conf * 0.9
    return ml_signal, conf

# ---------- SL/TP helpers ----------
def adjust_sl_tp_sim(price, sl, tp, digits=5, stop_level=10):
    point = 10 ** -digits
    min_dist = max(stop_level * point, 20 * point)
    price = round(price, digits); sl = round(sl, digits); tp = round(tp, digits)
    if sl < price:
        if price - sl < min_dist: sl = price - min_dist
        if tp < price or tp - price < min_dist: tp = price + min_dist*3
    else:
        if sl - price < min_dist: sl = price + min_dist
        if tp > price or price - tp < min_dist: tp = price - min_dist*3
    return sl, tp

def compute_tp_sl(price, atr, signal, risk_atr=1.0, reward_ratio=3.0):
    risk = atr * risk_atr
    if signal == 'BUY':
        sl = price - risk
        tp = price + risk * reward_ratio
    else:
        sl = price + risk
        tp = price - risk * reward_ratio
    return tp, sl

# ---------- Walk-forward engine (updated) ----------
def run_walk_forward(pair, start, end, interval,
                     initial_train, test_bars, step,
                     risk_atr, reward_ratio, base_conf,
                     model_type='rule',
                     use_trend_filter=True, use_dynamic_conf=True):
    df = fetch_data(pair, start, end, interval)
    if df.empty:
        return None
    macro_df = fetch_macro_data(start, end)
    commodity_df = fetch_commodity_data(start, end)
    if not macro_df.empty:
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        macro_df.index = pd.to_datetime(macro_df.index)
        df = df.merge(macro_df, left_on='Date', right_index=True, how='left')
        for c in macro_df.columns:
            if c in df.columns:
                df[c] = df[c].ffill()
        df = df.set_index('Date')
    else:
        for c in macro_df.columns if not macro_df.empty else []:
            df[c] = np.nan
    if not commodity_df.empty:
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        commodity_df.index = pd.to_datetime(commodity_df.index)
        df = df.merge(commodity_df, left_on='Date', right_index=True, how='left')
        for c in ['crude_oil','gold','agri']:
            if c in df.columns:
                df[c] = df[c].ffill()
        df = df.set_index('Date')
    else:
        for c in ['crude_oil','gold','agri']:
            df[c] = np.nan

    df = add_features(df)
    if df.empty or len(df) < initial_train + test_bars:
        print(f"Not enough data for {pair} (need {initial_train+test_bars}, have {len(df)})")
        return None

    nan_frac = df[FEATURES].isna().mean()
    bad = nan_frac[nan_frac > 0.95].index.tolist()
    if bad:
        df = df.drop(columns=bad)
    usable_features = [c for c in FEATURES if c in df.columns]
    df = df.dropna(subset=usable_features)
    if df.empty or len(df) < initial_train + test_bars:
        print(f"After cleaning, insufficient data for {pair}")
        return None

    results = []
    train_start = 0
    train_end = initial_train
    test_start = train_end
    test_end = test_start + test_bars
    iteration = 1

    while test_end <= len(df):
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        print(f"\n=== {pair} Iteration {iteration} ===")
        print(f"Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} bars)")
        print(f"Test: {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} bars)")

        if model_type == 'rule':
            signal_func = compute_rule_signal
            model_obj = None
            acc_val = 0.0
        elif model_type == 'svm_hmm':
            X_train, y_train, _ = prepare_ml_data(train_df)
            svm, hmm_model, scaler, acc = train_svm_hmm(X_train, y_train)
            if svm is None:
                print("Skipping – SVM+HMM training failed")
                train_end += step; test_start += step; test_end += step; iteration += 1
                continue
            def signal_func_svm(df, min_confidence):
                return compute_hybrid_svmhmm(df, svm, hmm_model, scaler, min_confidence=min_confidence)
            signal_func = signal_func_svm
            model_obj = (svm, hmm_model, scaler)
            acc_val = acc
        elif model_type == 'cnn_lstm':
            # This is the original hybrid (rule-dominated) – keep for comparison
            cnn_model = train_cnn_lstm(train_df, usable_features)
            if cnn_model is None:
                print("Skipping – CNN+LSTM training failed")
                train_end += step; test_start += step; test_end += step; iteration += 1
                continue
            def signal_func_cnn_hybrid(df, min_confidence):
                # Original hybrid logic from compute_hybrid_cnn_lstm (rule heavy)
                rule_signal, rule_conf = compute_rule_signal(df, min_confidence=0.0)
                ml_signal, ml_conf = predict_cnn_lstm(df, cnn_model, usable_features)
                if rule_signal != 'HOLD' and ml_signal != 'HOLD' and rule_signal == ml_signal:
                    conf = min(1.0, (rule_conf+ml_conf)/2 + 0.1)
                    return rule_signal, conf
                elif rule_signal != 'HOLD':
                    conf = rule_conf * 0.8
                    return rule_signal, conf
                elif ml_signal != 'HOLD':
                    conf = ml_conf * 0.8
                    return ml_signal, conf
                else:
                    return 'HOLD', 0.0
            signal_func = signal_func_cnn_hybrid
            model_obj = cnn_model
            acc_val = 0.0
        elif model_type == 'cnn_primary':
            # NEW: CNN-primary with rule as filter
            cnn_model = train_cnn_lstm(train_df, usable_features)
            if cnn_model is None:
                print("Skipping – CNN+LSTM training failed")
                train_end += step; test_start += step; test_end += step; iteration += 1
                continue
            def signal_func_cnn_primary(df, min_confidence):
                return compute_cnn_primary_signal(df, cnn_model, usable_features, min_confidence=min_confidence)
            signal_func = signal_func_cnn_primary
            model_obj = cnn_model
            acc_val = 0.0
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        trades = []
        in_position = False
        entry_price = 0.0; sl = 0.0; tp = 0.0; signal = 'HOLD'
        balance = INITIAL_BALANCE

        train_atr_median = train_df['atr_14'].median()
        for i in range(60, len(test_df)):
            window = test_df.iloc[:i+1]
            current_price = test_df['close'].iloc[i]
            current_atr = test_df['atr_14'].iloc[i]

            if use_dynamic_conf:
                atr_ratio = current_atr / train_atr_median if train_atr_median > 0 else 1.0
                conf_threshold = base_conf + 0.1*(atr_ratio-1.0)
                conf_threshold = max(0.5, min(0.9, conf_threshold))
            else:
                conf_threshold = base_conf

            if use_trend_filter:
                trend = detect_trend(window, ma_long=200)
            else:
                trend = None

            if in_position:
                if signal == 'BUY':
                    if current_price >= tp:
                        exit_price = current_price - SPREAD
                        pnl = (exit_price - entry_price) / entry_price * 100
                        trades[-1]['exit_price'] = exit_price
                        trades[-1]['pnl'] = pnl
                        trades[-1]['exit_time'] = test_df.index[i]
                        trades[-1]['net_pnl'] = pnl - (COMMISSION*LOT_SIZE*2/balance*100)
                        balance += (pnl/100)*balance - (COMMISSION*LOT_SIZE*2)
                        in_position = False
                    elif current_price <= sl:
                        exit_price = current_price - SPREAD
                        pnl = (exit_price - entry_price) / entry_price * 100
                        trades[-1]['exit_price'] = exit_price
                        trades[-1]['pnl'] = pnl
                        trades[-1]['exit_time'] = test_df.index[i]
                        trades[-1]['net_pnl'] = pnl - (COMMISSION*LOT_SIZE*2/balance*100)
                        balance += (pnl/100)*balance - (COMMISSION*LOT_SIZE*2)
                        in_position = False
                else:
                    if current_price <= tp:
                        exit_price = current_price + SPREAD
                        pnl = (entry_price - exit_price) / entry_price * 100
                        trades[-1]['exit_price'] = exit_price
                        trades[-1]['pnl'] = pnl
                        trades[-1]['exit_time'] = test_df.index[i]
                        trades[-1]['net_pnl'] = pnl - (COMMISSION*LOT_SIZE*2/balance*100)
                        balance += (pnl/100)*balance - (COMMISSION*LOT_SIZE*2)
                        in_position = False
                    elif current_price >= sl:
                        exit_price = current_price + SPREAD
                        pnl = (entry_price - exit_price) / entry_price * 100
                        trades[-1]['exit_price'] = exit_price
                        trades[-1]['pnl'] = pnl
                        trades[-1]['exit_time'] = test_df.index[i]
                        trades[-1]['net_pnl'] = pnl - (COMMISSION*LOT_SIZE*2/balance*100)
                        balance += (pnl/100)*balance - (COMMISSION*LOT_SIZE*2)
                        in_position = False

            if not in_position:
                sig, conf = signal_func(window, min_confidence=conf_threshold)
                if sig != 'HOLD':
                    # Apply trend filter (if enabled)
                    if use_trend_filter and trend is not None and trend != 'neutral':
                        if (sig == 'BUY' and trend != 'uptrend') or (sig == 'SELL' and trend != 'downtrend'):
                            continue
                    # Also apply rule filter for cnn_primary: we already did it in the signal function
                    tp_price, sl_price = compute_tp_sl(current_price, current_atr, sig,
                                                       risk_atr=risk_atr, reward_ratio=reward_ratio)
                    digits = 5 if not pair.startswith(('USDJPY','EURJPY','GBPJPY')) else 3
                    sl_price, tp_price = adjust_sl_tp_sim(current_price, sl_price, tp_price, digits=digits)
                    entry_price = current_price + SPREAD if sig == 'BUY' else current_price - SPREAD
                    signal = sig
                    sl = sl_price
                    tp = tp_price
                    in_position = True
                    trades.append({
                        'entry_time': test_df.index[i],
                        'entry_price': entry_price,
                        'signal': signal,
                        'sl': sl,
                        'tp': tp,
                        'exit_price': np.nan,
                        'pnl': 0.0,
                        'net_pnl': 0.0,
                        'exit_time': np.nan,
                        'confidence': conf
                    })

        if trades:
            df_trades = pd.DataFrame(trades)
            wins = df_trades[df_trades['net_pnl'] > 0]
            losses = df_trades[df_trades['net_pnl'] <= 0]
            win_rate = len(wins) / len(df_trades) * 100
            total_pnl = df_trades['net_pnl'].sum()
            print(f"Trades: {len(df_trades)}, Win rate: {win_rate:.1f}%, PnL: {total_pnl:.2f}%")
        else:
            win_rate = 0; total_pnl = 0
            print("No trades.")

        results.append({
            'iteration': iteration,
            'trades': len(trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'model_acc': acc_val if model_type=='svm_hmm' else 0.0
        })

        train_end += step
        test_start += step
        test_end += step
        iteration += 1

    if not results:
        return None
    return pd.DataFrame(results)

def run_multi_pairs(pairs, start, end, interval,
                    initial_train, test_bars, step,
                    risk_atr, reward_ratio, base_conf,
                    model_type='rule'):
    all_results = {}
    for pair in pairs:
        print(f"\n{'='*60}")
        print(f"Running {model_type.upper()} on {pair}")
        print('='*60)
        summary = run_walk_forward(
            pair=pair, start=start, end=end, interval=interval,
            initial_train=initial_train, test_bars=test_bars, step=step,
            risk_atr=risk_atr, reward_ratio=reward_ratio, base_conf=base_conf,
            model_type=model_type,
            use_trend_filter=True, use_dynamic_conf=True
        )
        if summary is not None:
            all_results[pair] = summary
    return all_results

if __name__ == '__main__':
    print("==== COMPARING RULE-ONLY vs SVM+HMM vs CNN+LSTM (hybrid) vs CNN_PRIMARY ====")
    print(f"Using {INTERVAL} data from {START_DATE} to {END_DATE}")
    print(f"Train: {INITIAL_TRAIN_BARS} bars, Test: {TEST_BARS} bars, Step: {STEP}")
    print(f"Parameters: risk_atr={RISK_ATR}, min_confidence={MIN_CONFIDENCE}, reward_ratio={REWARD_RATIO}")
    print("Trend filter and dynamic confidence ENABLED\n")

    # Run all four models
    rule_results = run_multi_pairs(
        pairs=PAIRS,
        start=START_DATE, end=END_DATE, interval=INTERVAL,
        initial_train=INITIAL_TRAIN_BARS, test_bars=TEST_BARS, step=STEP,
        risk_atr=RISK_ATR, reward_ratio=REWARD_RATIO, base_conf=MIN_CONFIDENCE,
        model_type='rule'
    )

    svm_results = run_multi_pairs(
        pairs=PAIRS,
        start=START_DATE, end=END_DATE, interval=INTERVAL,
        initial_train=INITIAL_TRAIN_BARS, test_bars=TEST_BARS, step=STEP,
        risk_atr=RISK_ATR, reward_ratio=REWARD_RATIO, base_conf=MIN_CONFIDENCE,
        model_type='svm_hmm'
    )

    cnn_hybrid_results = run_multi_pairs(
        pairs=PAIRS,
        start=START_DATE, end=END_DATE, interval=INTERVAL,
        initial_train=INITIAL_TRAIN_BARS, test_bars=TEST_BARS, step=STEP,
        risk_atr=RISK_ATR, reward_ratio=REWARD_RATIO, base_conf=MIN_CONFIDENCE,
        model_type='cnn_lstm'   # original hybrid (rule-dominated)
    )

    cnn_primary_results = run_multi_pairs(
        pairs=PAIRS,
        start=START_DATE, end=END_DATE, interval=INTERVAL,
        initial_train=INITIAL_TRAIN_BARS, test_bars=TEST_BARS, step=STEP,
        risk_atr=RISK_ATR, reward_ratio=REWARD_RATIO, base_conf=MIN_CONFIDENCE,
        model_type='cnn_primary'  # new primary CNN with rule as filter
    )

    print("\n\n========== FINAL COMPARISON ==========")
    comp_data = []
    for pair in PAIRS:
        if pair in rule_results and pair in svm_results and pair in cnn_hybrid_results and pair in cnn_primary_results:
            rule_df = rule_results[pair]
            svm_df = svm_results[pair]
            hybrid_df = cnn_hybrid_results[pair]
            primary_df = cnn_primary_results[pair]
            comp_data.append({
                'Pair': pair.replace('=X', ''),
                'Rule Win Rate': f"{rule_df['win_rate'].mean():.1f}%",
                'Rule PnL': f"{rule_df['total_pnl'].sum():.2f}%",
                'Rule Trades': f"{rule_df['trades'].mean():.1f}",
                'SVM+HMM Win Rate': f"{svm_df['win_rate'].mean():.1f}%",
                'SVM+HMM PnL': f"{svm_df['total_pnl'].sum():.2f}%",
                'SVM+HMM Trades': f"{svm_df['trades'].mean():.1f}",
                'CNN Hybrid Win Rate': f"{hybrid_df['win_rate'].mean():.1f}%",
                'CNN Hybrid PnL': f"{hybrid_df['total_pnl'].sum():.2f}%",
                'CNN Hybrid Trades': f"{hybrid_df['trades'].mean():.1f}",
                'CNN Primary Win Rate': f"{primary_df['win_rate'].mean():.1f}%",
                'CNN Primary PnL': f"{primary_df['total_pnl'].sum():.2f}%",
                'CNN Primary Trades': f"{primary_df['trades'].mean():.1f}",
                'SVM Acc': f"{svm_df['model_acc'].mean():.2f}" if 'model_acc' in svm_df else 'N/A'
            })
    comp_df = pd.DataFrame(comp_data)
    print(comp_df.to_string(index=False))