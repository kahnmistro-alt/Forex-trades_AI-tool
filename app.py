import os
import json
import time
import threading
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from dotenv import load_dotenv
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

PYTRADER_SERVER = os.environ.get('PYTRADER_SERVER', 'localhost')
PYTRADER_PORT = int(os.environ.get('PYTRADER_PORT', 1122))
PYTRADER_AUTH_CODE = os.environ.get('PYTRADER_AUTH_CODE', 'None')
TRADE_VOLUME = float(os.environ.get('TRADE_VOLUME', 0.01))

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
auto_trade_pairs = ['EURUSD', 'AUDCHF', 'NZDCHF', 'GBPNZD', 'USDCAD']  # Demo only

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
        'EURCHF', 'EURGBP', 'AUDCHF', 'NZDCHF', 'GBPNZD',
        'EURNOK', 'USDCHF', 'USDSGD', 'EURDKK', 'USDHKD'
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
    global pytrader, pytrader_connected
    if not pytrader_connected:
        if not connect_to_mt4():
            return None, False
    if pair not in pytrader.instrument_conversion_list:
        pytrader.instrument_conversion_list[pair] = pair
    quote = pytrader.Get_last_ask_bid(pair)
    if quote is None:
        return None, False
    if signal == 'BUY':
        return quote['ask'], True
    elif signal == 'SELL':
        return quote['bid'], True
    else:
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
    min_dist = max(stop_level * point, 20 * point)
    if sl < price:  # BUY
        if price - sl < min_dist:
            sl = price - min_dist
        if tp < price or tp - price < min_dist:
            tp = price + min_dist
    else:  # SELL
        if sl - price < min_dist:
            sl = price + min_dist
        if tp > price or price - tp < min_dist:
            tp = price - min_dist
    sl = round(sl, digits)
    tp = round(tp, digits)
    if (sl < price and tp > price) or (sl > price and tp < price):
        return sl, tp
    else:
        return None, None

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
    min_dist = max(stop_level * point, 20 * point)
    price = round(price, digits)
    sl = round(sl, digits)
    tp = round(tp, digits)
    if sl < price and tp > price:  # BUY
        if price - sl < min_dist:
            return False, f"SL too close (min {min_dist:.5f})"
        if tp - price < min_dist:
            return False, f"TP too close (min {min_dist:.5f})"
    elif sl > price and tp < price:  # SELL
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

# ---------- Pattern Detection ----------
def get_date_ranges():
    now = datetime.now()
    cy = now.year
    train_start = f"{cy-1}-01-01"
    train_end = f"{cy-1}-12-31"
    recent_start = f"{cy}-01-01"
    recent_end = now.strftime("%Y-%m-%d")
    return train_start, train_end, recent_start, recent_end

def fetch_data(pair, start, end, interval):
    try:
        data = yf.download(pair, start=start, end=end, interval=interval,
                           progress=False, timeout=30)
        if data.empty:
            data = yf.download(pair, start=start, end=end, interval='1d',
                               progress=False, timeout=30)
    except Exception as e:
        print(f"Error downloading {pair}: {e}")
        return pd.DataFrame()
    if 'Adj Close' in data.columns:
        data = data.drop(columns=['Adj Close'])
    if data.empty:
        return data
    data.columns = ['open', 'high', 'low', 'close', 'volume']
    return data

def detect_support_resistance(df, window=20, tolerance=0.005):
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

def detect_candlestick_patterns(df):
    patterns = {}
    cdl = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name='all')
    if cdl is not None:
        last = cdl.iloc[-1]
        for col in last.index:
            val = last[col]
            if val == 100:
                patterns[col] = 'bullish'
            elif val == -100:
                patterns[col] = 'bearish'
    if len(df) >= 2:
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        if (prev['close'] < prev['open'] and curr['close'] > curr['open'] and
            curr['open'] < prev['close'] and curr['close'] > prev['open']):
            patterns['CDL_ENGULFING_BULL'] = 'bullish'
        if (prev['close'] > prev['open'] and curr['close'] < curr['open'] and
            curr['open'] > prev['close'] and curr['close'] < prev['open']):
            patterns['CDL_ENGULFING_BEAR'] = 'bearish'
    return patterns

def detect_trend(df, ma_short=20, ma_long=50):
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

def detect_breakout(df, lookback=20, threshold=0.002):
    high = df['high'].iloc[-lookback:-1].max()
    low = df['low'].iloc[-lookback:-1].min()
    curr_close = df['close'].iloc[-1]
    if curr_close > high * (1 + threshold):
        return 'breakout_up'
    elif curr_close < low * (1 - threshold):
        return 'breakout_down'
    else:
        return None

def compute_pattern_signal(df):
    support, resistance = detect_support_resistance(df)
    patterns = detect_candlestick_patterns(df)
    trend = detect_trend(df)
    breakout = detect_breakout(df)

    bullish_count = sum(1 for p, d in patterns.items() if d == 'bullish')
    bearish_count = sum(1 for p, d in patterns.items() if d == 'bearish')

    if bullish_count > bearish_count:
        primary = 'BUY'
        pattern_strength = bullish_count / (bullish_count + bearish_count + 1e-6)
    elif bearish_count > bullish_count:
        primary = 'SELL'
        pattern_strength = bearish_count / (bullish_count + bearish_count + 1e-6)
    else:
        primary = 'HOLD'
        pattern_strength = 0.0

    if primary == 'BUY' and trend == 'uptrend':
        confidence = 0.7 + 0.3 * pattern_strength
    elif primary == 'SELL' and trend == 'downtrend':
        confidence = 0.7 + 0.3 * pattern_strength
    elif primary == 'BUY' and (support is not None and df['close'].iloc[-1] < support * 1.01):
        confidence = 0.6 + 0.2 * pattern_strength
    elif primary == 'SELL' and (resistance is not None and df['close'].iloc[-1] > resistance * 0.99):
        confidence = 0.6 + 0.2 * pattern_strength
    else:
        confidence = 0.3 * pattern_strength

    if breakout == 'breakout_up' and primary == 'BUY':
        confidence = min(1.0, confidence + 0.2)
    elif breakout == 'breakout_down' and primary == 'SELL':
        confidence = min(1.0, confidence + 0.2)

    confidence = max(0, min(1, confidence))

    # ML prediction
    ml_signal, ml_confidence = pattern_model.predict_pattern(df)
    if primary != 'HOLD' and ml_signal != 'HOLD':
        if primary == ml_signal:
            confidence = min(1.0, confidence + 0.1)
        else:
            confidence = max(0.0, confidence - 0.1)
    elif primary == 'HOLD' and ml_signal != 'HOLD' and ml_confidence > 0.7:
        primary = ml_signal
        confidence = ml_confidence * 0.8

    min_conf = get_min_confidence()
    if confidence >= min_conf:
        signal = primary
    else:
        signal = 'HOLD'

    details = {
        'patterns': patterns,
        'trend': trend,
        'support': support,
        'resistance': resistance,
        'breakout': breakout,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count
    }
    return signal, confidence, details

def compute_tp_sl(price, atr, signal, risk_atr=1.0, reward_ratio=3.0):
    risk = atr * risk_atr
    if signal == 'BUY':
        sl = price - risk
        tp = price + risk * reward_ratio
    else:
        sl = price + risk
        tp = price - risk * reward_ratio
    return tp, sl

def process_pair(pair, interval, atr_period, risk_mult, reward_ratio):
    train_start, train_end, recent_start, recent_end = get_date_ranges()
    try:
        df = fetch_data(pair, recent_start, recent_end, interval)
        if df.empty or len(df) < 60:
            return None

        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df = df.dropna()
        if df.empty:
            return None

        signal, confidence, details = compute_pattern_signal(df)
        current_atr = df['atr'].iloc[-1]

        live_price, live_ok = get_live_entry_price(pair, signal)
        if live_ok:
            reference_price = live_price
        else:
            reference_price = df['close'].iloc[-1]

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
            "confidence": round(confidence, 3),
            "price": round(reference_price, 5),
            "tp": round(tp, 5),
            "sl": round(sl, 5),
            "atr": round(current_atr, 5),
            "can_trade": can_trade,
            "can_trade_reason": reason,
            "pattern_details": details,
            "trend": details['trend'],
            "chart": chart_data,
            "error": None
        }
    except Exception as e:
        return {"pair": pair, "error": str(e)[:100]}

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
    risk_mult = 1.0
    reward_ratio = 3.0
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

# ---------- ML Auto-training ----------
training_lock = threading.Lock()
last_training_time = None

def train_model_background():
    global last_training_time
    with training_lock:
        print(f"[ML] Starting model training at {datetime.now()}")
        try:
            pair = 'EURUSD=X'
            end = datetime.now().strftime('%Y-%m-%d')
            start = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            df = fetch_data(pair, start, end, '1h')
            if df.empty:
                print("[ML] No data fetched for training.")
                return
            success = pattern_model.train(df, labels=None)
            if success:
                last_training_time = datetime.now()
                print(f"[ML] Model training completed at {last_training_time}")
            else:
                print("[ML] Training failed.")
        except Exception as e:
            print(f"[ML] Training error: {e}")

def schedule_training():
    train_model_background()
    threading.Timer(600, schedule_training).start()

# ---------- Flask Routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/signals', methods=['POST'])
def get_signals():
    data = request.get_json()
    DEFAULT_PAIRS = ('EURUSD=X, AUDCHF=X, NZDCHF=X, GBPNZD=X, USDCAD=X, '
                     'GBPUSD=X, USDJPY=X, AUDUSD=X, EURGBP=X, EURJPY=X, '
                     'USDCHF=X, NZDUSD=X, AUDJPY=X, EURAUD=X, GBPJPY=X, '
                     'EURCHF=X, CADJPY=X, AUDNZD=X, EURNZD=X, CHFJPY=X, '
                     'GBPCHF=X, GBPAUD=X, EURCAD=X, USDCNY=X, USDHKD=X, '
                     'USDSGD=X, USDSEK=X, USDNOK=X, USDDKK=X, EURNOK=X, '
                     'EURSEK=X, EURDKK=X, AUDCAD=X, NZDCAD=X, CADCHF=X')
    pairs_raw = data.get('pairs', DEFAULT_PAIRS)
    pair_list = [p.strip().upper() for p in pairs_raw.split(',') if p.strip()]
    interval = data.get('interval', '1h')
    atr_period = int(data.get('atr_period', 14))
    risk_mult = float(data.get('risk_mult', 1.0))
    reward_ratio = 3.0

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
    DEFAULT_PAIRS = ('EURUSD=X, AUDCHF=X, NZDCHF=X, GBPNZD=X, USDCAD=X, '
                     'GBPUSD=X, USDJPY=X, AUDUSD=X, EURGBP=X, EURJPY=X, '
                     'USDCHF=X, NZDUSD=X, AUDJPY=X, EURAUD=X, GBPJPY=X, '
                     'EURCHF=X, CADJPY=X, AUDNZD=X, EURNZD=X, CHFJPY=X, '
                     'GBPCHF=X, GBPAUD=X, EURCAD=X, USDCNY=X, USDHKD=X, '
                     'USDSGD=X, USDSEK=X, USDNOK=X, USDDKK=X, EURNOK=X, '
                     'EURSEK=X, EURDKK=X, AUDCAD=X, NZDCAD=X, CADCHF=X')
    pairs_raw = data.get('pairs', DEFAULT_PAIRS)
    pair_list = [p.strip().upper() for p in pairs_raw.split(',') if p.strip()]
    interval = data.get('interval', '1h')
    atr_period = int(data.get('atr_period', 14))
    risk_mult = float(data.get('risk_mult', 1.0))
    volume = float(data.get('volume', TRADE_VOLUME))
    reward_ratio = 3.0

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
        auto_trade_pairs = [p.replace('=X', '') for p in pair_list]
    else:
        auto_trade_pairs = []
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

# ---------- Startup ----------
if __name__ == '__main__':
    init_db()
    print(f"✅ Database ready. Min confidence: {get_min_confidence()}")
    if PYTRADER_AVAILABLE:
        connect_to_mt4()
    schedule_training()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)