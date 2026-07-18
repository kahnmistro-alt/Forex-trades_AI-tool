import os
import json
import time
import threading
import requests
import traceback
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from dotenv import load_dotenv
from candlestick_patterns import detect_candlestick_patterns, get_pattern_signal
from chart_patterns import detect_chart_patterns
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

try:
    from pytrader_api import Pytrader_API
    PYTRADER_AVAILABLE = True
except ImportError as e:
    PYTRADER_AVAILABLE = False
    print(f"PyTrader module not found: {e}. Auto-trading disabled.")

app = Flask(__name__)

# ---------- Best parameters from backtest ----------
RISK_ATR = 1.0
MIN_CONFIDENCE = 0.6
REWARD_RATIO = 3.0
AUTO_TRADE_PAIRS = ['USDJPY', 'GBPUSD']

PYTRADER_SERVER = os.environ.get('PYTRADER_SERVER', 'localhost')
PYTRADER_PORT = int(os.environ.get('PYTRADER_PORT', 1122))
PYTRADER_AUTH_CODE = os.environ.get('PYTRADER_AUTH_CODE', 'None')
TRADE_VOLUME = float(os.environ.get('TRADE_VOLUME', 0.01))

# ---------- Auto-retraining configuration ----------
AUTO_RETRAIN_ENABLED = os.environ.get('AUTO_RETRAIN_ENABLED', 'True').lower() == 'true'
RETRAIN_INTERVAL_DAYS = int(os.environ.get('RETRAIN_INTERVAL_DAYS', 7))
RETRAIN_DATA_DAYS = int(os.environ.get('RETRAIN_DATA_DAYS', 730))

# ---------- Twelve Data ----------
TWELVE_DATA_API_KEY = os.environ.get('TWELVE_DATA_API_KEY')
TWELVE_DATA_CACHE_TTL = int(os.environ.get('TWELVE_DATA_CACHE_TTL', 180))
TWELVE_BASE = 'https://api.twelvedata.com/quote'

_live_price_cache = {}
_twelve_request_count = 0

def get_live_price_twelve(symbol):
    if not TWELVE_DATA_API_KEY:
        return None
    if len(symbol) == 6 and symbol.isalpha():
        base = symbol[:3]
        quote = symbol[3:]
        symbol_formatted = f"{base}/{quote}"
    else:
        symbol_formatted = symbol
    cache_key = symbol
    now = time.time()
    if cache_key in _live_price_cache and (now - _live_price_cache[cache_key]['timestamp'] < TWELVE_DATA_CACHE_TTL):
        return _live_price_cache[cache_key]['price']
    try:
        url = f"{TWELVE_BASE}?symbol={symbol_formatted}&apikey={TWELVE_DATA_API_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'close' in data and data['close'] is not None:
                price = float(data['close'])
                _live_price_cache[cache_key] = {'price': price, 'timestamp': now}
                global _twelve_request_count
                _twelve_request_count += 1
                if _twelve_request_count % 50 == 0:
                    print(f"ℹ️ Twelve Data requests so far: {_twelve_request_count}")
                return price
        else:
            print(f"⚠️ Twelve Data error: {resp.status_code} for {symbol_formatted}")
    except Exception as e:
        print(f"⚠️ Twelve Data exception for {symbol}: {e}")
    return None

# ---------- Supabase ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set!")

if not SUPABASE_URL.endswith('/'):
    SUPABASE_URL += '/'
if 'rest/v1' not in SUPABASE_URL:
    SUPABASE_URL = SUPABASE_URL.rstrip('/') + '/rest/v1/'

def supabase_request(method, endpoint, data=None):
    url = SUPABASE_URL + endpoint
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    try:
        if method == 'GET':
            resp = requests.get(url, headers=headers)
        elif method == 'POST':
            resp = requests.post(url, json=data, headers=headers)
        elif method == 'PATCH':
            resp = requests.patch(url, json=data, headers=headers)
        elif method == 'DELETE':
            resp = requests.delete(url, headers=headers)
        else:
            return False, f"Unsupported method {method}"
        if resp.status_code in (200, 201, 204):
            return True, resp.json() if resp.text else {}
        else:
            return False, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, str(e)

# ---------- PyTrader state ----------
pytrader = None
pytrader_connected = False
last_connection_attempt = 0
connection_attempt_interval = 60

# ---------- Auto‑Trade state ----------
auto_trade_enabled = False
auto_trade_thread = None
auto_trade_lock = threading.Lock()
auto_trade_pairs = AUTO_TRADE_PAIRS

# ---------- ML Model Manager (per‑pair) ----------
class SupabaseClientWrapper:
    def __init__(self, supabase_url, service_key):
        self.base_url = supabase_url
        self.key = service_key

    def table(self, table_name):
        return TableWrapper(self, table_name)

class TableWrapper:
    def __init__(self, client, table_name):
        self.client = client
        self.table_name = table_name
        self.select_fields = '*'
        self.filters = {}
        self.order_by = None
        self.limit_val = None
        self.order_desc = False

    def select(self, fields):
        self.select_fields = fields
        return self

    def eq(self, column, value):
        self.filters[column] = value
        return self

    def order(self, column, desc=False):
        self.order_by = column
        self.order_desc = desc
        return self

    def limit(self, n):
        self.limit_val = n
        return self

    def execute(self):
        url = self.client.base_url + self.table_name
        params = []
        if self.select_fields != '*':
            params.append(f"select={self.select_fields}")
        for col, val in self.filters.items():
            params.append(f"{col}=eq.{val}")
        if self.order_by:
            params.append(f"order={self.order_by}.{'desc' if self.order_desc else 'asc'}")
        if self.limit_val:
            params.append(f"limit={self.limit_val}")
        if params:
            url += '?' + '&'.join(params)

        headers = {
            "apikey": self.client.key,
            "Authorization": f"Bearer {self.client.key}",
            "Content-Type": "application/json"
        }
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            class Response:
                def __init__(self, data):
                    self.data = data
            return Response(resp.json())
        else:
            class Response:
                def __init__(self):
                    self.data = []
            return Response()

    def insert(self, data):
        url = self.client.base_url + self.table_name
        headers = {
            "apikey": self.client.key,
            "Authorization": f"Bearer {self.client.key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        try:
            resp = requests.post(url, json=data, headers=headers)
            if resp.status_code in (200, 201):
                try:
                    json_data = resp.json()
                except:
                    json_data = []
                class Response:
                    def __init__(self, data):
                        self.data = data
                return Response(json_data)
            else:
                class Response:
                    def __init__(self):
                        self.data = []
                return Response()
        except Exception as e:
            class Response:
                def __init__(self):
                    self.data = []
            return Response()

supabase_wrapper = SupabaseClientWrapper(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ---------- Per-Pair Model Configuration ----------
PAIR_MODEL_TYPES = {
    'AUDUSD': 'rule',
    'EURUSD': 'xgboost',
    'GBPUSD': 'rf',
    'USDCAD': 'xgboost',
}

from pattern_model import PairModelManager
pair_model_manager = PairModelManager(supabase_wrapper)

for pair, model_type in PAIR_MODEL_TYPES.items():
    if model_type != 'rule':
        pair_model_manager.load_model(pair, model_type)
    else:
        pair_model_manager.models[pair] = {'type': 'rule', 'model': None, 'scaler': None}

print("✅ Per‑pair models loaded.")

# ---------- Data Fetch & Feature Functions ----------
def get_date_ranges():
    now = datetime.now()
    recent_end = now.strftime("%Y-%m-%d")
    recent_start = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    return recent_start, recent_end

def fetch_data(pair, start, end, interval):
    """
    Fetch data for a given pair. Tries both the provided symbol and the version with '=X'.
    """
    print(f"📊 Fetching {pair} ({interval}) from {start} to {end}...")
    symbols_to_try = [pair]
    if '=X' not in pair:
        symbols_to_try.append(pair + '=X')
    else:
        symbols_to_try.append(pair.replace('=X', ''))
    data = pd.DataFrame()
    for ticker in symbols_to_try:
        try:
            data = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                timeout=60,
                auto_adjust=False,
                threads=True,
                ignore_tz=True
            )
            if not data.empty:
                print(f"✅ {pair}: {len(data)} bars ({interval}) using {ticker}")
                if 'Adj Close' in data.columns:
                    data = data.drop(columns=['Adj Close'])
                data.columns = ['open', 'high', 'low', 'close', 'volume']
                return data
        except Exception as e:
            print(f"⚠️ {ticker} failed: {e}")
            continue
    # Fallback to daily if interval not '1d'
    if interval != '1d':
        print(f"📊 {pair}: falling back to daily...")
        for ticker in symbols_to_try:
            try:
                data = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval='1d',
                    progress=False,
                    timeout=60,
                    auto_adjust=False
                )
                if not data.empty:
                    print(f"✅ {pair}: {len(data)} daily bars using {ticker}")
                    if 'Adj Close' in data.columns:
                        data = data.drop(columns=['Adj Close'])
                    data.columns = ['open', 'high', 'low', 'close', 'volume']
                    return data
            except Exception as e:
                continue
    print(f"❌ No data for {pair} after all attempts.")
    return pd.DataFrame()

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
    df = df.dropna()
    return df

# ---------- Pattern detection (rule-based) ----------
def detect_support_resistance(df, window=20):
    try:
        high = df['high']
        low = df['low']
        piv_high = high[(high.shift(1) < high) & (high.shift(-1) < high)]
        piv_low = low[(low.shift(1) > low) & (low.shift(-1) > low)]
        if len(piv_high) == 0 or len(piv_low) == 0:
            return None, None
        current = df['close'].iloc[-1]
        support = piv_low[piv_low < current].max() if any(piv_low < current) else None
        resistance = piv_high[piv_high > current].min() if any(piv_high > current) else None
        return support, resistance
    except:
        return None, None

def detect_trend(df, ma_short=20, ma_long=50):
    try:
        if len(df) < ma_long:
            return 'neutral'
        sma_short = df['close'].rolling(ma_short).mean().iloc[-1]
        sma_long = df['close'].rolling(ma_long).mean().iloc[-1]
        short_slope = df['close'].rolling(ma_short).mean().diff(5).iloc[-1]
        if sma_short > sma_long and short_slope > 0:
            return 'uptrend'
        elif sma_short < sma_long and short_slope < 0:
            return 'downtrend'
        else:
            return 'neutral'
    except:
        return 'neutral'

def detect_breakout(df, lookback=20, threshold=0.002):
    try:
        high = df['high'].iloc[-lookback:-1].max()
        low = df['low'].iloc[-lookback:-1].min()
        curr_close = df['close'].iloc[-1]
        if curr_close > high * (1 + threshold):
            return 'breakout_up'
        elif curr_close < low * (1 - threshold):
            return 'breakout_down'
        else:
            return None
    except:
        return None

def compute_pattern_signal(df):
    candle_patterns = detect_candlestick_patterns(df)
    candle_signal, candle_conf = get_pattern_signal(candle_patterns)
    chart_patterns = detect_chart_patterns(df, lookback=40) if len(df) >= 40 else []
    support, resistance = detect_support_resistance(df)
    trend = detect_trend(df)
    breakout = detect_breakout(df)
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
        else:
            signal = 'HOLD'
            confidence = 0.0
    if signal != 'HOLD':
        if signal == 'BUY' and trend == 'downtrend':
            signal = 'HOLD'
            confidence = 0.0
        elif signal == 'SELL' and trend == 'uptrend':
            signal = 'HOLD'
            confidence = 0.0
    return signal, confidence

# ---------- SL/TP calculations ----------
def get_live_entry_price(pair, signal):
    symbol = pair.replace('=X', '')
    price = get_live_price_twelve(symbol)
    if price is not None:
        return price, True
    global pytrader, pytrader_connected
    if not pytrader_connected:
        if not connect_to_mt4():
            return None, False
    if pair not in pytrader.instrument_conversion_list:
        pytrader.instrument_conversion_list[pair] = pair
    try:
        quote = pytrader.Get_last_ask_bid(pair)
        if quote is not None:
            if signal == 'BUY':
                return quote['ask'], True
            elif signal == 'SELL':
                return quote['bid'], True
    except Exception as e:
        print(f"⚠️ PyTrader error for {pair}: {e}")
    return None, False

def get_valid_lot_size(pair, requested_volume):
    global pytrader, pytrader_connected
    if not pytrader_connected:
        if not connect_to_mt4():
            return requested_volume
    if pair not in pytrader.instrument_conversion_list:
        pytrader.instrument_conversion_list[pair] = pair
    info = pytrader.Get_instrument_info(pair)
    if info is None:
        min_lot, max_lot, step = 0.01, 10.0, 0.01
    else:
        min_lot = info.get('min_lotsize', 0.01)
        max_lot = info.get('max_lotsize', 10.0)
        step = info.get('lot_step', 0.01)
    valid_vol = max(min_lot, min(max_lot, requested_volume))
    valid_vol = round(valid_vol / step) * step
    if valid_vol < min_lot:
        valid_vol = min_lot
    return valid_vol

def adjust_sl_tp(pair, price, sl, tp):
    global pytrader, pytrader_connected
    if not pytrader_connected:
        if not connect_to_mt4():
            return sl, tp
    if pair not in pytrader.instrument_conversion_list:
        pytrader.instrument_conversion_list[pair] = pair
    info = pytrader.Get_instrument_info(pair)
    if info is None:
        stop_level = 10
        digits = 5
    else:
        stop_level = info.get('stop_level', 10)
        digits = info.get('digits', 5)
    price = round(price, digits)
    sl = round(sl, digits)
    tp = round(tp, digits)
    point = 10 ** -digits
    min_dist = max(stop_level * point, 50 * point)
    if sl < price:  # BUY
        if price - sl < min_dist:
            sl = price - min_dist
        if tp < price or tp - price < min_dist:
            tp = price + min_dist * 3
    else:  # SELL
        if sl - price < min_dist:
            sl = price + min_dist
        if tp > price or price - tp < min_dist:
            tp = price - min_dist * 3
    sl = round(sl, digits)
    tp = round(tp, digits)
    if not ((sl < price and tp > price) or (sl > price and tp < price)):
        if sl < price:
            sl = price - min_dist
            tp = price + min_dist * 3
        else:
            sl = price + min_dist
            tp = price - min_dist * 3
        sl = round(sl, digits)
        tp = round(tp, digits)
    return sl, tp

def validate_sl_tp(pair, price, sl, tp):
    global pytrader, pytrader_connected
    if not pytrader_connected:
        if not connect_to_mt4():
            return False, "MT4 not connected"
    if pair not in pytrader.instrument_conversion_list:
        pytrader.instrument_conversion_list[pair] = pair
    info = pytrader.Get_instrument_info(pair)
    if info is None:
        stop_level = 10
        digits = 5
    else:
        stop_level = info.get('stop_level', 10)
        digits = info.get('digits', 5)
    point = 10 ** -digits
    min_dist = max(stop_level * point, 50 * point)
    price = round(price, digits)
    sl = round(sl, digits)
    tp = round(tp, digits)
    if sl < price and tp > price:
        if price - sl < min_dist:
            return False, f"SL too close (min {min_dist:.5f})"
        if tp - price < min_dist:
            return False, f"TP too close (min {min_dist:.5f})"
    elif sl > price and tp < price:
        if sl - price < min_dist:
            return False, f"SL too close (min {min_dist:.5f})"
        if price - tp < min_dist:
            return False, f"TP too close (min {min_dist:.5f})"
    else:
        return False, "SL and TP on same side of price"
    return True, "OK"

def compute_tp_sl(price, atr, signal, risk_atr=RISK_ATR, reward_ratio=REWARD_RATIO):
    risk = atr * risk_atr
    if signal == 'BUY':
        sl = price - risk
        tp = price + risk * reward_ratio
    else:
        sl = price + risk
        tp = price - risk * reward_ratio
    return sl, tp

# ---------- Supabase helpers ----------
def table_exists(table_name):
    ok, _ = supabase_request('GET', table_name + '?limit=1')
    return ok

def init_db():
    if not table_exists('trades') or not table_exists('config'):
        print("\n⚠️  Tables 'trades' and/or 'config' are missing.")
        print("Please create them manually in your Supabase SQL Editor with:")
        print("""
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    pair TEXT,
    signal TEXT,
    entry_price FLOAT,
    tp FLOAT,
    sl FLOAT,
    result TEXT,
    pnl FLOAT
);

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value FLOAT
);

INSERT INTO config (key, value) VALUES ('min_confidence', 0.6)
ON CONFLICT (key) DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_trades_result ON trades(result);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
        """)
        print("The app will continue with default confidence (0.6).")
    else:
        print("✅ Tables 'trades' and 'config' already exist.")

    if not table_exists('pair_models'):
        print("\n⚠️  Table 'pair_models' does not exist.")
        print("Please create it manually in your Supabase SQL Editor with:")
        print("""
CREATE TABLE IF NOT EXISTS pair_models (
    id SERIAL PRIMARY KEY,
    pair TEXT NOT NULL,
    model_type TEXT NOT NULL,
    model_blob TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    version TEXT,
    metrics JSONB
);
CREATE INDEX IF NOT EXISTS idx_pair_model ON pair_models(pair, model_type, created_at DESC);
        """)
    else:
        print("✅ Table 'pair_models' already exists.")

def get_min_confidence():
    ok, result = supabase_request('GET', 'config?key=eq.min_confidence')
    if ok and result:
        return result[0].get('value', 0.6)
    return 0.6

def update_min_confidence(new_val):
    supabase_request('PATCH', 'config?key=eq.min_confidence', {'value': new_val})

def log_trade(pair, signal, price, tp, sl, result, pnl):
    data = {
        'pair': pair,
        'signal': signal,
        'entry_price': price,
        'tp': tp,
        'sl': sl,
        'result': result,
        'pnl': pnl,
        'timestamp': datetime.now().isoformat()
    }
    ok, _ = supabase_request('POST', 'trades', data)
    if not ok:
        print(f"Could not log trade: {_}")

def update_trade_result(trade_id, result, pnl):
    supabase_request('PATCH', f'trades?id=eq.{trade_id}', {'result': result, 'pnl': pnl})

def update_confidence_threshold():
    try:
        ok, rows = supabase_request('GET', 'trades?result=neq.pending&select=result')
        if not ok or len(rows) < 5:
            return
        wins = sum(1 for r in rows if r['result'] == 'win')
        win_rate = wins / len(rows)
        cur_val = get_min_confidence()
        if win_rate > 0.6:
            new_val = max(0.4, cur_val - 0.02)
        elif win_rate < 0.4:
            new_val = min(0.8, cur_val + 0.02)
        else:
            new_val = cur_val
        update_min_confidence(new_val)
    except Exception as e:
        print(f"Error updating confidence threshold: {e}")

# ---------- Main process_pair (with per-pair model) ----------
def process_pair(pair, interval, atr_period, risk_mult, reward_ratio):
    recent_start, recent_end = get_date_ranges()
    print(f"🔍 Processing {pair}...")
    try:
        df = fetch_data(pair, recent_start, recent_end, interval)
        if df.empty or len(df) < 60:
            return None
        df = add_features(df)
        if df.empty:
            return None
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df = df.dropna()
        if df.empty:
            return None

        # ---- Use per-pair model or fallback to rule ----
        ml_signal, ml_conf, model_type = pair_model_manager.predict(pair, df)
        if model_type != 'rule' and ml_signal is not None and ml_conf >= MIN_CONFIDENCE:
            signal = ml_signal
            confidence = ml_conf
            details = {'trend': detect_trend(df), 'support': None, 'resistance': None, 'breakout': None}
        else:
            signal, confidence = compute_pattern_signal(df)
            details = {
                'trend': detect_trend(df),
                'support': None,
                'resistance': None,
                'breakout': None
            }

        current_atr = df['atr'].iloc[-1]
        live_price, live_ok = get_live_entry_price(pair, signal)
        if live_ok:
            reference_price = live_price
        else:
            reference_price = df['close'].iloc[-1]

        if signal != 'HOLD':
            raw_sl, raw_tp = compute_tp_sl(reference_price, current_atr, signal,
                                           risk_atr=risk_mult, reward_ratio=reward_ratio)
            adjusted_sl, adjusted_tp = adjust_sl_tp(pair, reference_price, raw_sl, raw_tp)
            if adjusted_sl is None or adjusted_tp is None:
                can_trade = False
                reason = "Cannot adjust SL/TP to valid levels"
                sl, tp = raw_sl, raw_tp
            else:
                sl, tp = adjusted_sl, adjusted_tp
                can_trade, reason = validate_sl_tp(pair, reference_price, sl, tp)
        else:
            sl, tp = None, None
            can_trade = False
            reason = "No signal"

        chart_data = {
            "dates": [str(d) for d in df.index[-100:]],
            "prices": df['close'].iloc[-100:].tolist(),
            "signal_point": {
                "date": str(df.index[-1]),
                "price": reference_price,
                "signal": signal
            } if signal != "HOLD" else None
        }

        return {
            "pair": pair.replace("=X", ""),
            "signal": signal,
            "confidence": round(confidence, 3) if confidence else 0.0,
            "price": round(reference_price, 5),
            "tp": round(tp, 5) if tp else None,
            "sl": round(sl, 5) if sl else None,
            "atr": round(current_atr, 5),
            "can_trade": can_trade,
            "can_trade_reason": reason,
            "pattern_details": details,
            "trend": details['trend'],
            "chart": chart_data,
            "model_used": model_type,
            "error": None
        }
    except Exception as e:
        print(f"❌ Error processing {pair}: {e}")
        traceback.print_exc()
        return {"pair": pair, "error": str(e)[:100]}

# ---------- Execute trade ----------
def execute_trade(pair, signal, price, tp, sl, volume=TRADE_VOLUME):
    global pytrader, pytrader_connected
    if not PYTRADER_AVAILABLE:
        return {"success": False, "error": "PyTrader not available"}
    if not pytrader_connected:
        if not connect_to_mt4():
            return {"success": False, "error": "Could not connect to MT4"}
    if pair not in pytrader.instrument_conversion_list:
        pytrader.instrument_conversion_list[pair] = pair
    live_price, live_ok = get_live_entry_price(pair, signal)
    if not live_ok:
        ref_price = price
    else:
        ref_price = live_price
    adjusted_volume = get_valid_lot_size(pair, volume)
    adjusted_sl, adjusted_tp = adjust_sl_tp(pair, ref_price, sl, tp)
    if adjusted_sl is None or adjusted_tp is None:
        return {"success": False, "error": "Invalid SL/TP after adjustment"}
    valid, reason = validate_sl_tp(pair, ref_price, adjusted_sl, adjusted_tp)
    if not valid:
        return {"success": False, "error": f"SL/TP invalid: {reason}"}
    try:
        ordertype = "buy" if signal.upper() == "BUY" else "sell"
        ticket = pytrader.Open_order(
            instrument=pair,
            ordertype=ordertype,
            volume=adjusted_volume,
            openprice=0.0,
            slippage=5,
            magicnumber=0,
            stoploss=adjusted_sl,
            takeprofit=adjusted_tp,
            comment="Pattern",
            market=False
        )
        if ticket == -1:
            error_msg = pytrader.order_return_message or "Unknown error"
            return {"success": False, "error": f"Order failed: {error_msg}"}
        return {
            "success": True,
            "order_id": ticket,
            "message": f"{signal} {pair} executed, ticket {ticket}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ---------- Auto-Trade ----------
def connect_to_mt4():
    global pytrader, pytrader_connected, last_connection_attempt
    if not PYTRADER_AVAILABLE:
        return False
    now = time.time()
    if now - last_connection_attempt < connection_attempt_interval and not pytrader_connected:
        return False
    last_connection_attempt = now

    default_pairs = [
        'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD',
        'EURCHF', 'EURGBP', 'AUDCHF', 'NZDCHF', 'GBPNZD'
    ]
    instrument_lookup = {pair: pair for pair in default_pairs}

    try:
        pytrader = Pytrader_API()
        pytrader.debug = False
        success = pytrader.Connect(
            server=PYTRADER_SERVER,
            port=PYTRADER_PORT,
            instrument_lookup=instrument_lookup,
            authorization_code=PYTRADER_AUTH_CODE
        )
        if success:
            pytrader_connected = True
            print("✅ Connected to MT4 via PyTrader.")
            return True
        else:
            print("❌ PyTrader connection failed.")
            pytrader_connected = False
            return False
    except Exception as e:
        print(f"❌ PyTrader connection error: {e}")
        pytrader_connected = False
        return False

def auto_trade_iteration():
    if not auto_trade_enabled:
        return
    print(f"[Auto-Trade] Running at {datetime.now().strftime('%H:%M:%S')}")
    interval = '1h'
    atr_period = 14
    risk_mult = RISK_ATR
    reward_ratio = REWARD_RATIO
    volume = TRADE_VOLUME
    pairs = auto_trade_pairs
    for pair in pairs:
        yf_pair = pair if '=X' in pair else pair + '=X'
        res = process_pair(yf_pair, interval, atr_period, risk_mult, reward_ratio)
        if res and res.get('signal') in ('BUY', 'SELL') and res.get('can_trade'):
            trade_res = execute_trade(
                pair=res['pair'],
                signal=res['signal'],
                price=res['price'],
                tp=res['tp'],
                sl=res['sl'],
                volume=volume
            )
            if trade_res.get('success'):
                log_trade(res['pair'], res['signal'], res['price'],
                          res['tp'], res['sl'], 'pending', 0.0)
                print(f"[Auto-Trade] ✅ {res['signal']} {res['pair']} executed")
            else:
                print(f"[Auto-Trade] ❌ {res['pair']} failed: {trade_res.get('error')}")

def auto_trade_loop():
    while True:
        if auto_trade_enabled:
            auto_trade_iteration()
        time.sleep(60)

def start_auto_trade_thread():
    global auto_trade_thread
    if auto_trade_thread is None or not auto_trade_thread.is_alive():
        auto_trade_thread = threading.Thread(target=auto_trade_loop, daemon=True)
        auto_trade_thread.start()
        print("🚀 Auto-trade thread started (runs every 60s).")

# ---------- Auto-Retraining ----------
auto_retrain_running = False
auto_retrain_thread = None

def retrain_all_pairs():
    print(f"🔄 Starting auto-retraining at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=RETRAIN_DATA_DAYS)).strftime('%Y-%m-%d')
    results = {}
    for pair, model_type in PAIR_MODEL_TYPES.items():
        if model_type == 'rule':
            print(f"⏭️ Skipping {pair} (rule-based)")
            continue
        print(f"🔄 Retraining {pair} with {model_type}...")
        try:
            # Ensure the pair has '=X' for Yahoo Finance
            yf_pair = pair if '=X' in pair else pair + '=X'
            df = fetch_data(yf_pair, start, end, '1d')
            if df.empty:
                print(f"❌ No data for {pair}")
                results[pair] = {'success': False, 'error': 'No data'}
                continue
            df = add_features(df)
            if df.empty:
                print(f"❌ Feature engineering failed for {pair}")
                results[pair] = {'success': False, 'error': 'Feature engineering failed'}
                continue
            model, scaler, acc = pair_model_manager.train_model(df, model_type)
            if model is None:
                print(f"❌ Training failed for {pair}")
                results[pair] = {'success': False, 'error': 'Training failed'}
                continue
            pair_model_manager.save_model(pair, model_type, model, scaler, {'accuracy': acc})
            print(f"✅ Saved new {model_type} model for {pair} with accuracy {acc:.2f}")
            results[pair] = {'success': True, 'accuracy': acc}
        except Exception as e:
            print(f"❌ Error retraining {pair}: {e}")
            results[pair] = {'success': False, 'error': str(e)}
    print(f"🔄 Auto-retraining completed. Results: {results}")
    return results

def auto_retrain_loop():
    global auto_retrain_running
    while AUTO_RETRAIN_ENABLED:
        if auto_retrain_running:
            time.sleep(RETRAIN_INTERVAL_DAYS * 24 * 3600)
        else:
            auto_retrain_running = True
            retrain_all_pairs()
            time.sleep(RETRAIN_INTERVAL_DAYS * 24 * 3600)

def start_auto_retrain_thread():
    global auto_retrain_thread
    if not AUTO_RETRAIN_ENABLED:
        print("ℹ️ Auto-retraining is disabled by environment variable.")
        return
    if auto_retrain_thread is None or not auto_retrain_thread.is_alive():
        auto_retrain_thread = threading.Thread(target=auto_retrain_loop, daemon=True)
        auto_retrain_thread.start()
        print(f"🚀 Auto-retraining thread started (interval: {RETRAIN_INTERVAL_DAYS} days).")

# ---------- Flask Routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/signals', methods=['POST'])
def get_signals():
    data = request.get_json()
    DEFAULT_PAIRS = 'EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDCAD=X'
    pairs_raw = data.get('pairs', DEFAULT_PAIRS)
    pair_list = [p.strip().upper() for p in pairs_raw.split(',') if p.strip()]
    interval = data.get('interval', '1h')
    atr_period = int(data.get('atr_period', 14))
    risk_mult = float(data.get('risk_mult', RISK_ATR))
    reward_ratio = REWARD_RATIO
    update_confidence_threshold()
    results = []
    for pair in pair_list:
        res = process_pair(pair, interval, atr_period, risk_mult, reward_ratio)
        if res:
            results.append(res)
    return jsonify(results)

@app.route('/api/autotrade', methods=['POST'])
def auto_trade():
    data = request.get_json()
    DEFAULT_PAIRS = 'EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDCAD=X'
    pairs_raw = data.get('pairs', DEFAULT_PAIRS)
    pair_list = [p.strip().upper() for p in pairs_raw.split(',') if p.strip()]
    interval = data.get('interval', '1h')
    atr_period = int(data.get('atr_period', 14))
    risk_mult = float(data.get('risk_mult', RISK_ATR))
    volume = float(data.get('volume', TRADE_VOLUME))
    reward_ratio = REWARD_RATIO
    update_confidence_threshold()
    results = []
    for pair in pair_list:
        res = process_pair(pair, interval, atr_period, risk_mult, reward_ratio)
        if res:
            if res['signal'] in ('BUY', 'SELL') and res.get('can_trade', False):
                trade_res = execute_trade(
                    pair=res['pair'],
                    signal=res['signal'],
                    price=res['price'],
                    tp=res['tp'],
                    sl=res['sl'],
                    volume=volume
                )
                res['trade'] = trade_res
                if trade_res['success']:
                    log_trade(res['pair'], res['signal'], res['price'],
                              res['tp'], res['sl'], 'pending', 0.0)
            else:
                res['trade'] = {"success": False, "error": "No valid signal or invalid SL/TP"}
            results.append(res)
    return jsonify(results)

@app.route('/api/auto_trade_status', methods=['GET', 'POST'])
def auto_trade_status():
    global auto_trade_enabled
    if request.method == 'POST':
        data = request.get_json()
        enabled = data.get('enabled', False)
        with auto_trade_lock:
            auto_trade_enabled = enabled
            if enabled:
                start_auto_trade_thread()
        return jsonify({"enabled": auto_trade_enabled})
    else:
        return jsonify({"enabled": auto_trade_enabled})

@app.route('/api/auto_trade_pairs', methods=['POST'])
def set_auto_trade_pairs():
    global auto_trade_pairs
    data = request.get_json()
    pairs_raw = data.get('pairs', '')
    if pairs_raw:
        pair_list = [p.strip().upper() for p in pairs_raw.split(',') if p.strip()]
        allowed = ['USDJPY', 'GBPUSD']
        auto_trade_pairs = [p.replace('=X', '') for p in pair_list if p.replace('=X', '') in allowed]
        if not auto_trade_pairs:
            auto_trade_pairs = ['USDJPY', 'GBPUSD']
    else:
        auto_trade_pairs = ['USDJPY', 'GBPUSD']
    return jsonify({"pairs": auto_trade_pairs})

@app.route('/api/update_trade', methods=['POST'])
def update_trade():
    data = request.get_json()
    trade_id = data.get('trade_id')
    result = data.get('result')
    pnl = data.get('pnl', 0.0)
    if not trade_id:
        return jsonify({"success": False, "error": "trade_id required"})
    try:
        update_trade_result(trade_id, result, pnl)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    data = request.get_json()
    pair = data.get('pair')
    model_type = data.get('model_type')
    if not pair:
        return jsonify({"success": False, "error": "pair required"})
    if model_type is None:
        model_type = PAIR_MODEL_TYPES.get(pair)
        if model_type is None:
            return jsonify({"success": False, "error": "No model type configured for this pair"})
    if model_type == 'rule':
        return jsonify({"success": False, "error": "Rule-based model cannot be retrained"})
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=RETRAIN_DATA_DAYS)).strftime('%Y-%m-%d')
    # Ensure pair has '=X' for Yahoo
    yf_pair = pair if '=X' in pair else pair + '=X'
    df = fetch_data(yf_pair, start, end, '1d')
    if df.empty:
        return jsonify({"success": False, "error": "No data fetched"})
    df = add_features(df)
    if df.empty:
        return jsonify({"success": False, "error": "Feature engineering failed"})
    model, scaler, acc = pair_model_manager.train_model(df, model_type)
    if model is None:
        return jsonify({"success": False, "error": "Training failed"})
    pair_model_manager.save_model(pair, model_type, model, scaler, {'accuracy': acc})
    return jsonify({"success": True, "accuracy": acc, "model_type": model_type})

@app.route('/api/retrain_all', methods=['POST'])
def retrain_all():
    try:
        results = retrain_all_pairs()
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/retrain_status', methods=['GET'])
def retrain_status():
    return jsonify({
        "enabled": AUTO_RETRAIN_ENABLED,
        "interval_days": RETRAIN_INTERVAL_DAYS,
        "running": auto_retrain_running
    })

# ---------- Startup ----------
if __name__ == '__main__':
    init_db()
    print(f"✅ Database ready. Min confidence: {get_min_confidence()}")
    if PYTRADER_AVAILABLE:
        connect_to_mt4()
    if TWELVE_DATA_API_KEY:
        test_price = get_live_price_twelve('EURUSD')
        if test_price:
            print(f"✅ Twelve Data works: EURUSD = {test_price}")
        else:
            print("❌ Twelve Data failed – check API key")
    else:
        print("ℹ️ Twelve Data API key not set – using PyTrader/yfinance for prices.")
    test_pair = 'EURUSD=X'
    test_data = fetch_data(test_pair, (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'), '1h')
    if not test_data.empty:
        print(f"✅ yfinance works: fetched {len(test_data)} bars for {test_pair}")
    else:
        print(f"❌ yfinance failed for {test_pair}. Check internet connection and package.")
    start_auto_retrain_thread()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)