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

from candlestick_patterns import detect_candlestick_patterns, get_pattern_signal
from chart_patterns import detect_chart_patterns

# ---------- Configuration ----------
PAIRS = ['EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X']   # USDJPY excluded
START_DATE = '2024-07-17'
END_DATE = '2025-07-16'
INTERVAL = '1h'
INITIAL_BALANCE = 10000
LOT_SIZE = 0.01
SPREAD = 0.0001
COMMISSION = 5.0

RISK_ATR = 1.0
MIN_CONFIDENCE = 0.6
REWARD_RATIO = 3.0

INITIAL_TRAIN_BARS = 600
TEST_BARS = 600
STEP = 600

FEATURES = [
    'open', 'high', 'low', 'close',
    'open_prev_1', 'high_prev_1', 'low_prev_1', 'close_prev_1',
    'open_prev_2', 'high_prev_2', 'low_prev_2', 'close_prev_2',
    'rsi_14', 'macd', 'atr_14',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
    'roc_10', 'roc_20',
    'returns_std_10', 'returns_skew_10', 'returns_kurt_10'
]

# ---------- Data Fetch & Feature Engineering ----------
def fetch_data(pair, start, end, interval):
    print(f"Fetching {pair} {interval} from {start} to {end}...")
    tickers = [pair, pair.replace('=X', '')]
    data = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end, interval=interval,
                               progress=False, auto_adjust=False, timeout=60,
                               multi_level_index=False)
            if not data.empty:
                print(f"✅ Data fetched for {ticker}")
                break
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            continue
    if data.empty:
        print("1h data empty, trying daily...")
        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start, end=end, interval='1d',
                                   progress=False, auto_adjust=False, timeout=60,
                                   multi_level_index=False)
                if not data.empty:
                    print(f"✅ Daily data for {ticker}")
                    break
            except Exception:
                continue
    if data.empty:
        print("All attempts failed.")
        return pd.DataFrame()
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [c.lower() for c in data.columns]
    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in data.columns:
            data[col] = data.get('close', np.nan)
    data = data[['open', 'high', 'low', 'close']]
    print(f"Data shape: {data.shape}")
    return data

def add_features(df):
    if df.empty:
        return df
    df = df.copy()
    if len(df) < 20:
        return pd.DataFrame()
    
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        df['macd'] = macd_df.get('MACD_12_26_9', macd_df.get('MACD', 0))
    else:
        df['macd'] = 0
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    sma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = sma + 2 * std
    df['bb_middle'] = sma
    df['bb_lower'] = sma - 2 * std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_width'] = df['bb_width'].replace([np.inf, -np.inf], np.nan)
    df['roc_10'] = ta.roc(df['close'], length=10)
    df['roc_20'] = ta.roc(df['close'], length=20)
    ret = df['close'].pct_change()
    df['returns_std_10'] = ret.rolling(10).std()
    df['returns_skew_10'] = ret.rolling(10).skew()
    df['returns_kurt_10'] = ret.rolling(10).kurt()
    for lag in [1, 2]:
        df[f'open_prev_{lag}'] = df['open'].shift(lag)
        df[f'high_prev_{lag}'] = df['high'].shift(lag)
        df[f'low_prev_{lag}'] = df['low'].shift(lag)
        df[f'close_prev_{lag}'] = df['close'].shift(lag)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES)
    print(f"After feature engineering: {df.shape[0]} rows")
    return df

def prepare_ml_data(df):
    X = df[FEATURES].values
    future_ret = df['close'].shift(-1) / df['close'] - 1
    y = np.where(future_ret > 0.001, 1, np.where(future_ret < -0.001, -1, 0))
    X = X[:-1]
    y = y[:-1]
    return X, y, df.index[:-1]

# ---------- SVM + HMM (fixed) ----------
def train_svm_hmm(X, y):
    mask = y != 0
    X_bin = X[mask]
    y_bin = (y[mask] > 0).astype(int)
    if len(X_bin) < 50:
        return None, None, None, 0.0  # 4 values: svm, hmm, scaler, acc
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bin)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bin, test_size=0.2, random_state=42)

    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    print(f"✅ SVM trained with accuracy: {acc:.2f}")

    X_all_scaled = scaler.transform(X)
    hmm_model = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=100, random_state=42)
    hmm_model.fit(X_all_scaled)
    print("✅ HMM trained.")
    return svm, hmm_model, scaler, acc

def predict_svm_hmm(df, svm, hmm_model, scaler):
    try:
        X = df[FEATURES].values[-1:].reshape(1, -1)
        X_scaled = scaler.transform(X)
        prob = svm.predict_proba(X_scaled)[0]
        pred = svm.predict(X_scaled)[0]
        conf = prob[1] if pred == 1 else prob[0]
        state_probs = hmm_model.predict_proba(X_scaled)[0]
        dominant_state = np.argmax(state_probs)
        if pred == 1:
            if dominant_state == 2:  # uptrend
                confidence = min(1.0, conf + 0.15)
            elif dominant_state == 0:
                confidence = max(0.0, conf - 0.15)
            else:
                confidence = conf
        else:
            if dominant_state == 0:  # downtrend
                confidence = min(1.0, conf + 0.15)
            elif dominant_state == 2:
                confidence = max(0.0, conf - 0.15)
            else:
                confidence = conf
        signal = 'BUY' if pred == 1 else 'SELL'
        return signal, confidence
    except:
        return 'HOLD', 0.0

# ---------- Rule-based (pure patterns) ----------
def detect_trend(df, ma_long=200):
    if len(df) < ma_long:
        return 'neutral'
    sma_long = df['close'].rolling(ma_long).mean().iloc[-1]
    current = df['close'].iloc[-1]
    if current > sma_long:
        return 'uptrend'
    elif current < sma_long:
        return 'downtrend'
    else:
        return 'neutral'

def compute_rule_signal(df, min_confidence=0.6):
    candle_patterns = detect_candlestick_patterns(df)
    candle_signal, candle_conf = get_pattern_signal(candle_patterns)
    chart_patterns = detect_chart_patterns(df, lookback=40) if len(df) >= 40 else []
    trend = detect_trend(df, ma_long=200)
    signal = 'HOLD'
    confidence = 0.0
    if candle_signal != 'HOLD':
        signal = candle_signal
        confidence = candle_conf
        for pat in chart_patterns:
            if pat in ['double_bottom', 'inverse_head_shoulders', 'ascending_triangle'] and signal == 'BUY':
                confidence = min(1.0, confidence + 0.15)
            elif pat in ['double_top', 'head_shoulders', 'descending_triangle'] and signal == 'SELL':
                confidence = min(1.0, confidence + 0.15)
        if signal == 'BUY' and trend == 'uptrend':
            confidence = min(1.0, confidence + 0.1)
        elif signal == 'SELL' and trend == 'downtrend':
            confidence = min(1.0, confidence + 0.1)
    else:
        if chart_patterns:
            bullish_pats = ['double_bottom', 'inverse_head_shoulders', 'ascending_triangle', 'falling_wedge']
            bearish_pats = ['double_top', 'head_shoulders', 'descending_triangle', 'rising_wedge']
            bullish = sum(1 for p in chart_patterns if p in bullish_pats)
            bearish = sum(1 for p in chart_patterns if p in bearish_pats)
            if bullish > bearish:
                signal = 'BUY'
                confidence = 0.5 + 0.3 * (bullish / (bullish + bearish + 1e-6))
            elif bearish > bullish:
                signal = 'SELL'
                confidence = 0.5 + 0.3 * (bearish / (bullish + bearish + 1e-6))
            else:
                signal = 'HOLD'
                confidence = 0.0
            if signal == 'BUY' and trend == 'uptrend':
                confidence = min(1.0, confidence + 0.2)
            elif signal == 'SELL' and trend == 'downtrend':
                confidence = min(1.0, confidence + 0.2)
    if confidence >= min_confidence and signal != 'HOLD':
        return signal, confidence
    else:
        return 'HOLD', 0.0

# ---------- Hybrid signal (for SVM+HMM) ----------
def compute_hybrid_signal_svmhmm(df, svm, hmm_model, scaler, min_confidence=0.6):
    rule_signal, rule_conf = compute_rule_signal(df, min_confidence=0.0)
    ml_signal, ml_conf = predict_svm_hmm(df, svm, hmm_model, scaler)

    if rule_signal != 'HOLD' and ml_signal != 'HOLD':
        avg_conf = (rule_conf + ml_conf) / 2
        if rule_signal == ml_signal:
            confidence = min(1.0, avg_conf + 0.1)
            signal = rule_signal
        else:
            confidence = max(0.0, avg_conf - 0.1)
            signal = 'HOLD'
    elif rule_signal != 'HOLD' and ml_signal == 'HOLD':
        confidence = rule_conf * 0.8
        signal = rule_signal
    elif ml_signal != 'HOLD' and rule_signal == 'HOLD':
        confidence = ml_conf * 0.8
        signal = ml_signal
    else:
        signal = 'HOLD'
        confidence = 0.0
    if confidence >= min_confidence and signal != 'HOLD':
        return signal, confidence
    else:
        return 'HOLD', 0.0

# ---------- SL/TP helpers ----------
def adjust_sl_tp_sim(price, sl, tp, digits=5, stop_level=10):
    point = 10 ** -digits
    min_dist = max(stop_level * point, 20 * point)
    price = round(price, digits)
    sl = round(sl, digits)
    tp = round(tp, digits)
    if sl < price:
        if price - sl < min_dist:
            sl = price - min_dist
        if tp < price or tp - price < min_dist:
            tp = price + min_dist
    else:
        if sl - price < min_dist:
            sl = price + min_dist
        if tp > price or price - tp < min_dist:
            tp = price - min_dist
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

# ---------- Walk‑Forward Engine ----------
def run_walk_forward(pair, start, end, interval,
                     initial_train, test_bars, step,
                     risk_atr, reward_ratio, base_conf,
                     model_type='rule',
                     use_trend_filter=True, use_dynamic_conf=True):
    df = fetch_data(pair, start, end, interval)
    if df.empty:
        return None
    df = add_features(df)
    if df.empty or len(df) < initial_train + test_bars:
        print(f"Not enough data: {len(df)} bars, need {initial_train + test_bars}")
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

        print(f"\n=== Iteration {iteration} ===")
        print(f"Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} bars)")
        print(f"Test: {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} bars)")

        if model_type == 'svm_hmm':
            X_train, y_train, _ = prepare_ml_data(train_df)
            svm, hmm_model, scaler, acc = train_svm_hmm(X_train, y_train)
            if svm is None:
                print("Skipping – insufficient samples")
                train_end += step
                test_start += step
                test_end += step
                iteration += 1
                continue
            signal_func = compute_hybrid_signal_svmhmm
            ml_model = (svm, hmm_model, scaler)
            acc_val = acc
        else:  # rule only
            svm = None; hmm_model = None; scaler = None; acc_val = 0.0
            signal_func = None
            ml_model = None

        trades = []
        in_position = False
        entry_price = 0.0
        sl = 0.0
        tp = 0.0
        signal = 'HOLD'
        balance = INITIAL_BALANCE

        train_atr_median = train_df['atr_14'].median()
        train_atr_std = train_df['atr_14'].std()

        for i in range(60, len(test_df)):
            window = test_df.iloc[:i+1]
            current_price = test_df['close'].iloc[i]
            current_atr = test_df['atr_14'].iloc[i]

            if use_dynamic_conf:
                atr_ratio = current_atr / train_atr_median if train_atr_median > 0 else 1.0
                conf_threshold = base_conf + 0.1 * (atr_ratio - 1.0)
                conf_threshold = max(0.4, min(0.9, conf_threshold))
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
                        trades[-1]['net_pnl'] = pnl - (COMMISSION * LOT_SIZE * 2 / balance * 100)
                        balance += (pnl / 100) * balance - (COMMISSION * LOT_SIZE * 2)
                        in_position = False
                    elif current_price <= sl:
                        exit_price = current_price - SPREAD
                        pnl = (exit_price - entry_price) / entry_price * 100
                        trades[-1]['exit_price'] = exit_price
                        trades[-1]['pnl'] = pnl
                        trades[-1]['exit_time'] = test_df.index[i]
                        trades[-1]['net_pnl'] = pnl - (COMMISSION * LOT_SIZE * 2 / balance * 100)
                        balance += (pnl / 100) * balance - (COMMISSION * LOT_SIZE * 2)
                        in_position = False
                else:
                    if current_price <= tp:
                        exit_price = current_price + SPREAD
                        pnl = (entry_price - exit_price) / entry_price * 100
                        trades[-1]['exit_price'] = exit_price
                        trades[-1]['pnl'] = pnl
                        trades[-1]['exit_time'] = test_df.index[i]
                        trades[-1]['net_pnl'] = pnl - (COMMISSION * LOT_SIZE * 2 / balance * 100)
                        balance += (pnl / 100) * balance - (COMMISSION * LOT_SIZE * 2)
                        in_position = False
                    elif current_price >= sl:
                        exit_price = current_price + SPREAD
                        pnl = (entry_price - exit_price) / entry_price * 100
                        trades[-1]['exit_price'] = exit_price
                        trades[-1]['pnl'] = pnl
                        trades[-1]['exit_time'] = test_df.index[i]
                        trades[-1]['net_pnl'] = pnl - (COMMISSION * LOT_SIZE * 2 / balance * 100)
                        balance += (pnl / 100) * balance - (COMMISSION * LOT_SIZE * 2)
                        in_position = False

            if not in_position:
                if model_type == 'svm_hmm':
                    sig, conf = signal_func(window, svm, hmm_model, scaler, min_confidence=conf_threshold)
                else:
                    sig, conf = compute_rule_signal(window, min_confidence=conf_threshold)

                if sig != 'HOLD':
                    if use_trend_filter and trend is not None and trend != 'neutral':
                        if (sig == 'BUY' and trend != 'uptrend') or (sig == 'SELL' and trend != 'downtrend'):
                            continue

                    tp_price, sl_price = compute_tp_sl(current_price, current_atr, sig,
                                                       risk_atr=risk_atr, reward_ratio=reward_ratio)
                    digits = 5 if not pair.startswith(('USDJPY', 'EURJPY', 'GBPJPY')) else 3
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
            avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
            avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0
            print(f"Trades: {len(df_trades)}, Win rate: {win_rate:.1f}%, PnL: {total_pnl:.2f}%")
        else:
            win_rate = 0
            total_pnl = 0
            avg_win = 0
            avg_loss = 0
            print("No trades.")

        results.append({
            'iteration': iteration,
            'trades': len(trades) if trades else 0,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'model_acc': acc_val if model_type == 'svm_hmm' else 0.0
        })

        train_end += step
        test_start += step
        test_end += step
        iteration += 1

    df_summary = pd.DataFrame(results)
    print("\n=== Walk‑Forward Summary ===")
    print(df_summary[['iteration', 'trades', 'win_rate', 'total_pnl']].to_string(index=False))
    return df_summary

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
            pair=pair,
            start=start,
            end=end,
            interval=interval,
            initial_train=initial_train,
            test_bars=test_bars,
            step=step,
            risk_atr=risk_atr,
            reward_ratio=reward_ratio,
            base_conf=base_conf,
            model_type=model_type,
            use_trend_filter=True,
            use_dynamic_conf=True
        )
        if summary is not None:
            all_results[pair] = summary
    return all_results

if __name__ == '__main__':
    print("==== COMPARING RULE-ONLY vs SVM+HMM (excluding USDJPY) ====")
    print("Parameters: risk_atr=1.0, min_confidence=0.6, reward_ratio=3.0")
    print("Trend filter and dynamic confidence ENABLED\n")

    # Run rule-only
    rule_results = run_multi_pairs(
        pairs=PAIRS,
        start=START_DATE,
        end=END_DATE,
        interval=INTERVAL,
        initial_train=INITIAL_TRAIN_BARS,
        test_bars=TEST_BARS,
        step=STEP,
        risk_atr=RISK_ATR,
        reward_ratio=REWARD_RATIO,
        base_conf=MIN_CONFIDENCE,
        model_type='rule'
    )

    # Run SVM+HMM
    svm_results = run_multi_pairs(
        pairs=PAIRS,
        start=START_DATE,
        end=END_DATE,
        interval=INTERVAL,
        initial_train=INITIAL_TRAIN_BARS,
        test_bars=TEST_BARS,
        step=STEP,
        risk_atr=RISK_ATR,
        reward_ratio=REWARD_RATIO,
        base_conf=MIN_CONFIDENCE,
        model_type='svm_hmm'
    )

    # Final comparison table
    print("\n\n========== FINAL COMPARISON ==========")
    comp_data = []
    for pair in PAIRS:
        if pair in rule_results and pair in svm_results:
            rule_df = rule_results[pair]
            svm_df = svm_results[pair]
            comp_data.append({
                'Pair': pair.replace('=X', ''),
                'Rule Win Rate': f"{rule_df['win_rate'].mean():.1f}%",
                'Rule PnL': f"{rule_df['total_pnl'].sum():.2f}%",
                'Rule Trades': f"{rule_df['trades'].mean():.1f}",
                'SVM+HMM Win Rate': f"{svm_df['win_rate'].mean():.1f}%",
                'SVM+HMM PnL': f"{svm_df['total_pnl'].sum():.2f}%",
                'SVM+HMM Trades': f"{svm_df['trades'].mean():.1f}",
                'SVM Acc': f"{svm_df['model_acc'].mean():.2f}"
            })
    comp_df = pd.DataFrame(comp_data)
    print(comp_df.to_string(index=False))