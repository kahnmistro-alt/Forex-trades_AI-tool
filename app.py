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
from pattern_model import PatternModel
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
MIN_CONFIDENCE = 0.7
REWARD_RATIO = 3.0
AUTO_TRADE_PAIRS = ['USDJPY', 'GBPUSD']

PYTRADER_SERVER = os.environ.get('PYTRADER_SERVER', 'localhost')
PYTRADER_PORT = int(os.environ.get('PYTRADER_PORT', 1122))
PYTRADER_AUTH_CODE = os.environ.get('PYTRADER_AUTH_CODE', 'None')
TRADE_VOLUME = float(os.environ.get('TRADE_VOLUME', 0.01))

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

# Test Supabase write
try:
    test_data = {
        'pair': 'TEST',
        'signal': 'BUY',
        'entry_price': 1.0,
        'tp': 1.1,
        'sl': 0.9,
        'result': 'pending',
        'pnl': 0.0
    }
    ok, _ = supabase_request('POST', 'trades', test_data)
    if ok:
        print("✅ Supabase write test successful")
    else:
        print("❌ Supabase write test failed – but continuing.")
except Exception as e:
    print(f"❌ Supabase write test exception: {e}")

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

# ---------- ML Model ----------
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

    def order(self, column, desc=False):
        self.order_by = column
        self.order_desc = desc
        return self

    def limit(self, n):
        self.limit_val = n
        return self

    def execute(self):
        url = self.client.base_url + self.table_name
        params = {}
        if self.select_fields != '*':
            params['select'] = self.select_fields
        if self.order_by:
            params['order'] = f'{self.order_by}.desc' if self.order_desc else f'{self.order_by}.asc'
        if self.limit_val:
            params['limit'] = self.limit_val
        headers = {
            "apikey": self.client.key,
            "Authorization": f"Bearer {self.client.key}",
            "Content-Type": "application/json"
        }
        resp = requests.get(url, params=params, headers=headers)
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
pattern_model = PatternModel(supabase_wrapper)

# ---------- Connection ----------
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

# ---------- Volume & SL/TP ----------
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

# ---------- Supabase Helpers ----------
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
    pnl FLOAT,
    source TEXT,
    ml_conf FLOAT,
    rule_conf FLOAT
);

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value FLOAT
);

INSERT INTO config (key, value) VALUES ('min_confidence', 0.7)
ON CONFLICT (key) DO NOTHING;

INSERT INTO config (key, value) VALUES ('rule_weight', 0.3)
ON CONFLICT (key) DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_trades_result ON trades(result);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
        """)
        print("The app will continue with default confidence (0.7).")
    else:
        print("✅ Tables 'trades' and 'config' already exist.")

    if not table_exists('pattern_models'):
        print("\n⚠️  Table 'pattern_models' does not exist.")
        print("Please create it manually in your Supabase SQL Editor with:")
        print("""
CREATE TABLE IF NOT EXISTS pattern_models (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    version TEXT,
    model_blob TEXT
);
        """)
    else:
        print("✅ Table 'pattern_models' already exists.")

def get_min_confidence():
    ok, result = supabase_request('GET', 'config?key=eq.min_confidence')
    if ok and result:
        return result[0].get('value', 0.7)
    return 0.7

def update_min_confidence(new_val):
    supabase_request('PATCH', 'config?key=eq.min_confidence', {'value': new_val})

def load_rule_weight():
    global _rule_weight
    ok, result = supabase_request('GET', 'config?key=eq.rule_weight')
    if ok and result:
        _rule_weight = result[0].get('value', 0.3)
    else:
        _rule_weight = 0.3
    print(f"Loaded rule_weight: {_rule_weight:.2f}")

def log_trade(pair, signal, price, tp, sl, result, pnl, source='hybrid', ml_conf=0.0, rule_conf=0.0):
    data = {
        'pair': pair,
        'signal': signal,
        'entry_price': price,
        'tp': tp,
        'sl': sl,
        'result': result,
        'pnl': pnl,
        'timestamp': datetime.now().isoformat(),
        'source': source,
        'ml_conf': ml_conf,
        'rule_conf': rule_conf
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

# ---------- Dynamic Fusion Weight ----------
_rule_weight = 0.3
_rule_weight_lock = threading.Lock()
_RULE_WEIGHT_UPDATE_INTERVAL = 3600  # 1 hour

def update_rule_weight():
    """Adjust rule_weight based on recent performance of ML vs rule signals."""
    global _rule_weight
    try:
        ok, trades = supabase_request('GET', 'trades?limit=50&order=timestamp.desc')
        if not ok or len(trades) < 20:
            return
        ml_trades = [t for t in trades if t.get('source') == 'ml' and t['result'] != 'pending']
        rule_trades = [t for t in trades if t.get('source') == 'rule' and t['result'] != 'pending']
        if len(ml_trades) < 10 or len(rule_trades) < 10:
            return
        ml_wr = sum(1 for t in ml_trades if t['result'] == 'win') / len(ml_trades)
        rule_wr = sum(1 for t in rule_trades if t['result'] == 'win') / len(rule_trades)
        with _rule_weight_lock:
            if rule_wr > ml_wr:
                _rule_weight = min(1.0, _rule_weight + 0.05)
            elif ml_wr > rule_wr:
                _rule_weight = max(0.0, _rule_weight - 0.05)
            supabase_request('PATCH', 'config?key=eq.rule_weight', {'value': _rule_weight})
        print(f"Updated rule_weight: {_rule_weight:.2f} (ML WR: {ml_wr:.2f}, Rule WR: {rule_wr:.2f})")
    except Exception as e:
        print(f"Error updating rule_weight: {e}")

def rule_weight_update_loop():
    while True:
        time.sleep(_RULE_WEIGHT_UPDATE_INTERVAL)
        update_rule_weight()

def start_rule_weight_thread():
    thread = threading.Thread(target=rule_weight_update_loop, daemon=True)
    thread.start()
    print("🚀 Rule‑weight update thread started.")

# ---------- Pattern Recognition ----------
def get_date_ranges():
    now = datetime.now()
    recent_end = now.strftime("%Y-%m-%d")
    recent_start = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    return recent_start, recent_end

def fetch_data(pair, start, end, interval):
    print(f"📊 Fetching {pair} ({interval}) from {start} to {end}...")
    data = pd.DataFrame()
    try:
        data = yf.download(
            pair,
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
            print(f"✅ {pair}: {len(data)} bars ({interval})")
            if 'Adj Close' in data.columns:
                data = data.drop(columns=['Adj Close'])
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            return data
    except Exception as e:
        print(f"❌ {pair} ({interval}) error: {e}")
    try:
        print(f"📊 {pair}: falling back to daily...")
        data = yf.download(
            pair,
            start=start,
            end=end,
            interval='1d',
            progress=False,
            timeout=60
        )
        if not data.empty:
            print(f"✅ {pair}: {len(data)} daily bars")
            if 'Adj Close' in data.columns:
                data = data.drop(columns=['Adj Close'])
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            return data
    except Exception as e:
        print(f"❌ {pair} daily fallback error: {e}")
    try:
        print(f"📊 {pair}: trying with period='1mo'...")
        data = yf.download(
            pair,
            period='1mo',
            interval='1h',
            progress=False,
            timeout=60
        )
        if not data.empty:
            print(f"✅ {pair}: {len(data)} bars (period='1mo')")
            if 'Adj Close' in data.columns:
                data = data.drop(columns=['Adj Close'])
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            return data
    except Exception as e:
        print(f"❌ {pair} period fallback error: {e}")
    print(f"❌ No data for {pair} after all attempts.")
    return pd.DataFrame()

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

# ---------- Profit Calculation Helpers ----------
def get_pip_value(pair, volume=1.0):
    """Return dollars per pip for a given pair and volume (standard lot = 1.0)."""
    global pytrader, pytrader_connected
    if pytrader_connected and pair in pytrader.instrument_conversion_list:
        info = pytrader.Get_instrument_info(pair)
        if info:
            tick_size = info.get('tick_size')
            tick_value = info.get('tick_value')  # per standard lot
            if tick_size and tick_value:
                # dollars per point * volume
                return tick_value / tick_size * volume
    # Fallback approximations
    if pair.endswith('JPY'):
        pip_size = 0.01
    else:
        pip_size = 0.0001
    # Standard lot (1.0) = $10 per pip, so dollars per pip = 10 * volume
    return 10.0 * volume

def compute_profit(entry, exit_price, pair, volume):
    """Return dollar profit/loss for a trade from entry to exit."""
    global pytrader, pytrader_connected
    if pytrader_connected and pair in pytrader.instrument_conversion_list:
        info = pytrader.Get_instrument_info(pair)
        if info:
            tick_size = info.get('tick_size')
            tick_value = info.get('tick_value')  # per standard lot
            if tick_size and tick_value:
                diff = exit_price - entry
                profit = (diff / tick_size) * tick_value * volume
                return profit
    # Fallback
    if pair.endswith('JPY'):
        pip_size = 0.01
    else:
        pip_size = 0.0001
    pips = (exit_price - entry) / pip_size
    profit = pips * 10.0 * volume
    return profit

# ---------- Model Retraining ----------
_retraining_lock = threading.Lock()
_retraining_thread = None
_auto_retrain_enabled = True
_RETRAIN_INTERVAL_HOURS = 6
_RETRAIN_PAIRS = ['EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X']

def retrain_model():
    """Fetch fresh data from the start of the year and retrain the SVM+HMM model."""
    with _retraining_lock:
        print("🔄 Starting model retraining...")
        try:
            all_dfs = []
            now = datetime.now()
            start_of_year = datetime(now.year, 1, 1)  # Jan 1 of current year

            for pair in _RETRAIN_PAIRS:
                df = fetch_data(pair, start_of_year.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d'), '1h')
                if not df.empty:
                    # Keep timestamp as column, reset index
                    df = df.reset_index()
                    all_dfs.append(df)

            if not all_dfs:
                print("❌ No data for retraining.")
                return False

            combined = pd.concat(all_dfs, ignore_index=True)
            combined = combined.drop(columns=['index']) if 'index' in combined.columns else combined

            success = pattern_model.train(combined)
            if success:
                print("✅ Model retrained successfully.")
            else:
                print("❌ Retraining failed.")
            return success
        except Exception as e:
            print(f"❌ Retraining error: {e}")
            traceback.print_exc()
            return False

def auto_retrain_loop():
    while True:
        if _auto_retrain_enabled:
            retrain_model()
        time.sleep(_RETRAIN_INTERVAL_HOURS * 3600)

def start_auto_retrain_thread():
    global _retraining_thread
    if _retraining_thread is None or not _retraining_thread.is_alive():
        _retraining_thread = threading.Thread(target=auto_retrain_loop, daemon=True)
        _retraining_thread.start()
        print("🚀 Auto‑retrain thread started.")

# ---------- Signal Computation (ML + Rule) ----------
def compute_pattern_signal(df):
    """
    Returns (ml_signal, ml_conf, rule_signal, rule_conf, details)
    """
    # ML prediction
    ml_signal, ml_conf = pattern_model.predict_pattern(df)

    # ---- Rule-based (candlestick + chart patterns) ----
    candle_patterns = detect_candlestick_patterns(df)
    candle_signal, candle_conf = get_pattern_signal(candle_patterns)
    chart_patterns = detect_chart_patterns(df, lookback=40) if len(df) >= 40 else []
    support, resistance = detect_support_resistance(df)
    trend = detect_trend(df)
    breakout = detect_breakout(df)

    rule_signal = 'HOLD'
    rule_conf = 0.0

    if candle_signal != 'HOLD':
        rule_signal = candle_signal
        rule_conf = candle_conf
        for pat in chart_patterns:
            if pat in ['double_bottom', 'inverse_head_shoulders', 'ascending_triangle'] and rule_signal == 'BUY':
                rule_conf = min(1.0, rule_conf + 0.15)
            elif pat in ['double_top', 'head_shoulders', 'descending_triangle'] and rule_signal == 'SELL':
                rule_conf = min(1.0, rule_conf + 0.15)
        if rule_signal == 'BUY' and trend == 'uptrend':
            rule_conf = min(1.0, rule_conf + 0.1)
        elif rule_signal == 'SELL' and trend == 'downtrend':
            rule_conf = min(1.0, rule_conf + 0.1)
    else:
        if chart_patterns:
            bullish_pats = ['double_bottom', 'inverse_head_shoulders', 'ascending_triangle', 'falling_wedge']
            bearish_pats = ['double_top', 'head_shoulders', 'descending_triangle', 'rising_wedge']
            bullish = sum(1 for p in chart_patterns if p in bullish_pats)
            bearish = sum(1 for p in chart_patterns if p in bearish_pats)
            if bullish > bearish:
                rule_signal = 'BUY'
                rule_conf = 0.5 + 0.3 * (bullish / (bullish + bearish + 1e-6))
            elif bearish > bullish:
                rule_signal = 'SELL'
                rule_conf = 0.5 + 0.3 * (bearish / (bullish + bearish + 1e-6))
            else:
                rule_signal = 'HOLD'
                rule_conf = 0.0
            if rule_signal == 'BUY' and trend == 'uptrend':
                rule_conf = min(1.0, rule_conf + 0.2)
            elif rule_signal == 'SELL' and trend == 'downtrend':
                rule_conf = min(1.0, rule_conf + 0.2)
        else:
            rule_signal = 'HOLD'
            rule_conf = 0.0

    # Trend filter for both
    if rule_signal != 'HOLD':
        if rule_signal == 'BUY' and trend == 'downtrend':
            rule_signal = 'HOLD'
            rule_conf = 0.0
        elif rule_signal == 'SELL' and trend == 'uptrend':
            rule_signal = 'HOLD'
            rule_conf = 0.0

    details = {
        'trend': trend,
        'support': support,
        'resistance': resistance,
        'breakout': breakout
    }
    return ml_signal, ml_conf, rule_signal, rule_conf, details

def compute_tp_sl(price, atr, signal, risk_atr=RISK_ATR, reward_ratio=REWARD_RATIO):
    risk = atr * risk_atr
    if signal == 'BUY':
        sl = price - risk
        tp = price + risk * reward_ratio
    else:
        sl = price + risk
        tp = price - risk * reward_ratio
    return sl, tp

def process_pair(pair, interval, atr_period, risk_mult, reward_ratio):
    recent_start, recent_end = get_date_ranges()
    print(f"🔍 Processing {pair}...")
    try:
        df = fetch_data(pair, recent_start, recent_end, interval)
        if df.empty or len(df) < 60:
            return None
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
        df = df.dropna()
        if df.empty:
            return None

        # Get ML + rule signals
        ml_signal, ml_conf, rule_signal, rule_conf, details = compute_pattern_signal(df)

        # Fuse using dynamic weight
        with _rule_weight_lock:
            rw = _rule_weight
        final_signal, final_conf = pattern_model.fuse_signals(ml_signal, ml_conf, rule_signal, rule_conf, rw)

        # --- Enforce minimum confidence (0.7) ---
        min_conf = get_min_confidence()
        if final_conf < min_conf:
            final_signal = 'HOLD'
            final_conf = 0.0

        # Determine source for logging
        if final_signal == 'HOLD':
            source = 'hold'
        elif final_signal == ml_signal and final_signal != 'HOLD':
            source = 'ml'
        elif final_signal == rule_signal and final_signal != 'HOLD':
            source = 'rule'
        else:
            source = 'hybrid'

        current_atr = df['atr'].iloc[-1]
        live_price, live_ok = get_live_entry_price(pair, final_signal)
        if live_ok:
            reference_price = live_price
        else:
            reference_price = df['close'].iloc[-1]

        # Compute TP/SL only if signal is not HOLD
        if final_signal != 'HOLD':
            raw_sl, raw_tp = compute_tp_sl(reference_price, current_atr, final_signal,
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
            can_trade = False
            reason = "No signal"
            sl, tp = None, None

        # Profit calculation
        volume_for_profit = TRADE_VOLUME
        tp_profit = None
        sl_loss = None
        if final_signal != 'HOLD' and can_trade and sl is not None and tp is not None:
            tp_profit = compute_profit(reference_price, tp, pair, volume_for_profit)
            sl_loss = compute_profit(reference_price, sl, pair, volume_for_profit)

        chart_data = {
            "dates": [str(d) for d in df.index[-100:]],
            "prices": df['close'].iloc[-100:].tolist(),
            "signal_point": {
                "date": str(df.index[-1]),
                "price": reference_price,
                "signal": final_signal
            } if final_signal != "HOLD" else None
        }

        return {
            "pair": pair.replace("=X", ""),
            "signal": final_signal,
            "confidence": round(final_conf, 3) if final_signal != 'HOLD' else 0.0,
            "price": round(reference_price, 5),
            "tp": round(tp, 5) if tp is not None else None,
            "sl": round(sl, 5) if sl is not None else None,
            "atr": round(current_atr, 5),
            "can_trade": can_trade,
            "can_trade_reason": reason,
            "pattern_details": details,
            "trend": details['trend'],
            "chart": chart_data,
            "tp_profit": round(tp_profit, 2) if tp_profit is not None else None,
            "sl_loss": round(sl_loss, 2) if sl_loss is not None else None,
            "source": source,
            "ml_conf": round(ml_conf, 3),
            "rule_conf": round(rule_conf, 3),
            "error": None
        }
    except Exception as e:
        print(f"❌ Error processing {pair}: {e}")
        traceback.print_exc()
        return {"pair": pair, "error": str(e)[:100]}

# ---------- Trade Execution ----------
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
                log_trade(
                    res['pair'], res['signal'], res['price'],
                    res['tp'], res['sl'], 'pending', 0.0,
                    source=res.get('source', 'hybrid'),
                    ml_conf=res.get('ml_conf', 0.0),
                    rule_conf=res.get('rule_conf', 0.0)
                )
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
                    log_trade(
                        res['pair'], res['signal'], res['price'],
                        res['tp'], res['sl'], 'pending', 0.0,
                        source=res.get('source', 'hybrid'),
                        ml_conf=res.get('ml_conf', 0.0),
                        rule_conf=res.get('rule_conf', 0.0)
                    )
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
def retrain_endpoint():
    """Manually trigger model retraining."""
    try:
        success = retrain_model()
        return jsonify({'success': success, 'message': 'Retraining completed' if success else 'Retraining failed'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

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
    # Load rule_weight from DB
    load_rule_weight()
    # Start background threads
    start_auto_retrain_thread()
    start_rule_weight_thread()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)