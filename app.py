# app.py – Lightweight SVM+HMM signal generation (no CNN)
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
from pattern_model import PatternModel
import pandas_datareader.data as web
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

# ---------- Parameters ----------
RISK_ATR = 1.0
MIN_CONFIDENCE = 0.7
REWARD_RATIO = 1.5                     # 1:1.5 ratio (SL:TP)
RISK_PER_TRADE = 0.01
BALANCE = 10000.0
ALL_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF',
             'EURGBP', 'EURJPY', 'NZDUSD', 'GBPJPY', 'USDJPY']
AUTO_TRADE_PAIRS = ALL_PAIRS

PYTRADER_SERVER = os.environ.get('PYTRADER_SERVER', 'localhost')
PYTRADER_PORT = int(os.environ.get('PYTRADER_PORT', 1122))
PYTRADER_AUTH_CODE = os.environ.get('PYTRADER_AUTH_CODE', 'None')
TRADE_VOLUME = float(os.environ.get('TRADE_VOLUME', 0.01))
BALANCE = float(os.environ.get('BALANCE', BALANCE))

# ---------- Fixed volume & fixed dollar profit ----------
FIXED_VOLUME = 1.0                     # always trade 1 lot
TARGET_PROFIT_DOLLARS = 100.0          # profit target

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

# ---------- ML Model (SVM+HMM) ----------
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
        params = {}
        if self.select_fields != '*':
            params['select'] = self.select_fields
        for col, val in self.filters.items():
            params[col] = f'eq.{val}'
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

    def update(self, data):
        url = self.client.base_url + self.table_name
        params = {}
        for col, val in self.filters.items():
            params[col] = f'eq.{val}'
        if params:
            url += '?' + '&'.join(f'{k}={v}' for k, v in params.items())
        headers = {
            "apikey": self.client.key,
            "Authorization": f"Bearer {self.client.key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        try:
            resp = requests.patch(url, json=data, headers=headers)
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

def get_live_bid_ask(pair):
    """Return (bid, ask) or (None, None) if not available."""
    global pytrader, pytrader_connected
    if not pytrader_connected:
        if not connect_to_mt4():
            return None, None
    if pair not in pytrader.instrument_conversion_list:
        pytrader.instrument_conversion_list[pair] = pair
    try:
        quote = pytrader.Get_last_ask_bid(pair)
        if quote is not None:
            return quote['bid'], quote['ask']
    except Exception as e:
        print(f"⚠️ PyTrader bid/ask error for {pair}: {e}")
    return None, None

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

def adjust_sl_tp(pair, price, sl, tp, ratio=REWARD_RATIO):
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
        sl_distance = price - sl
        min_tp_distance = max(sl_distance * ratio, min_dist)
        if tp < price + min_tp_distance:
            tp = price + min_tp_distance
    else:  # SELL
        if sl - price < min_dist:
            sl = price + min_dist
        sl_distance = sl - price
        min_tp_distance = max(sl_distance * ratio, min_dist)
        if tp > price - min_tp_distance:
            tp = price - min_tp_distance

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

# ---------- Position Sizing (kept for compatibility) ----------
def compute_volume(pair, entry, sl, balance=BALANCE, risk_per_trade=RISK_PER_TRADE):
    if pair.endswith('JPY'):
        pip_size = 0.01
    else:
        pip_size = 0.0001
    sl_pips = abs(entry - sl) / pip_size
    if sl_pips <= 0:
        return TRADE_VOLUME
    pip_value = get_pip_value(pair, volume=1.0)
    risk_amount = balance * risk_per_trade
    volume = risk_amount / (sl_pips * pip_value)
    return get_valid_lot_size(pair, volume)

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

def log_trade(pair, signal, price, tp, sl, result, pnl, source='svm_hmm', ml_conf=0.0, rule_conf=0.0):
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

# ---------- Macro Data Cache ----------
_macro_cache = {}
_macro_lock = threading.Lock()

def fetch_macro_data(start_date, end_date):
    global _macro_cache
    with _macro_lock:
        cache_key = f"{start_date}_{end_date}"
        if cache_key in _macro_cache:
            print("ℹ️ Using cached macro data.")
            return _macro_cache[cache_key]
        try:
            print("📊 Fetching macro data (yields & VIX)...")
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
                    print(f"⚠️ Could not fetch {name} ({fred_code}): {e}")
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
                print("⚠️ No macro data fetched.")
                return pd.DataFrame()

            required_cols = ['us10y', 'de10y', 'uk10y', 'au10y', 'ca10y', 'ch10y', 'nz10y', 'jp10y', 'vix']
            for col in required_cols:
                if col not in combined.columns:
                    combined[col] = np.nan

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

            spread_cols = [col for col in combined.columns if col.startswith('spread_')]
            combined = combined.dropna(subset=spread_cols, how='all')

            _macro_cache[cache_key] = combined
            print(f"✅ Macro data fetched: {len(combined)} rows")
            return combined
        except Exception as e:
            print(f"❌ Error fetching macro data: {e}")
            traceback.print_exc()
            return pd.DataFrame()

# ---------- Commodity Data Cache ----------
_commodity_cache = {}
_commodity_lock = threading.Lock()

def fetch_commodity_data(start_date, end_date):
    global _commodity_cache
    with _commodity_lock:
        cache_key = f"{start_date}_{end_date}"
        if cache_key in _commodity_cache:
            print("ℹ️ Using cached commodity data.")
            return _commodity_cache[cache_key]
        try:
            print("📊 Fetching commodity data...")
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
                    print(f"⚠️ Could not fetch {name} ({symbol}): {e}")
                    combined[name] = np.nan
            if combined.empty:
                print("⚠️ No commodity data fetched.")
                return pd.DataFrame()
            combined = combined.ffill()
            combined = combined.dropna(how='all')
            _commodity_cache[cache_key] = combined
            print(f"✅ Commodity data fetched: {len(combined)} rows")
            return combined
        except Exception as e:
            print(f"❌ Error fetching commodity data: {e}")
            traceback.print_exc()
            return pd.DataFrame()

# ---------- Data Fetching ----------
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
    print(f"❌ No data for {pair} after all attempts.")
    return pd.DataFrame()

# ---------- Profit Calculation Helpers ----------
def get_pip_value(pair, volume=1.0):
    global pytrader, pytrader_connected
    if pytrader_connected and pair in pytrader.instrument_conversion_list:
        info = pytrader.Get_instrument_info(pair)
        if info:
            tick_size = info.get('tick_size')
            tick_value = info.get('tick_value')
            if tick_size and tick_value:
                return tick_value / tick_size * volume
    if pair.endswith('JPY'):
        pip_size = 0.01
    else:
        pip_size = 0.0001
    return 10.0 * volume

def compute_profit(entry, exit_price, pair, volume):
    global pytrader, pytrader_connected
    if pytrader_connected and pair in pytrader.instrument_conversion_list:
        info = pytrader.Get_instrument_info(pair)
        if info:
            tick_size = info.get('tick_size')
            tick_value = info.get('tick_value')
            if tick_size and tick_value:
                diff = exit_price - entry
                profit = (diff / tick_size) * tick_value * volume
                return profit
    if pair.endswith('JPY'):
        pip_size = 0.01
    else:
        pip_size = 0.0001
    pips = (exit_price - entry) / pip_size
    profit = pips * 10.0 * volume
    return profit

# ---------- Model Retraining (Full retrain from January 1) ----------
_retraining_lock = threading.Lock()
_retraining_thread = None
_auto_retrain_enabled = True
_RETRAIN_INTERVAL_HOURS = 24   # once per day
_RETRAIN_PAIRS = [
    'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X',
    'USDCHF=X', 'EURGBP=X', 'EURJPY=X', 'NZDUSD=X',
    'GBPJPY=X', 'USDJPY=X'
]

def retrain_model():
    with _retraining_lock:
        print("🔄 Starting model training (full retrain from year-to-date)...")
        try:
            now = datetime.now()
            start_of_year = datetime(now.year, 1, 1).strftime('%Y-%m-%d')
            end = now.strftime('%Y-%m-%d')
            print(f"📅 Training period: {start_of_year} to {end}")
            all_dfs = []

            macro_df = fetch_macro_data(start_of_year, end)
            if macro_df.empty:
                print("⚠️ Macro data empty, trying to fetch anyway...")
            commodity_df = fetch_commodity_data(start_of_year, end)
            if commodity_df.empty:
                print("⚠️ Commodity data empty, trying to fetch anyway...")

            for pair in _RETRAIN_PAIRS:
                df = fetch_data(pair, start_of_year, end, '1h')
                if not df.empty:
                    df = df.reset_index()
                    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
                    df['Date'] = pd.to_datetime(df['Date'])

                    if not macro_df.empty:
                        macro_df.index = pd.to_datetime(macro_df.index)
                        df = df.merge(macro_df, left_on='Date', right_index=True, how='left')
                        for col in macro_df.columns:
                            if col in df.columns:
                                df[col] = df[col].ffill()

                    if not commodity_df.empty:
                        commodity_df.index = pd.to_datetime(commodity_df.index)
                        df = df.merge(commodity_df, left_on='Date', right_index=True, how='left')
                        for col in ['crude_oil', 'gold', 'agri']:
                            if col in df.columns:
                                df[col] = df[col].ffill()

                    all_dfs.append(df)

            if not all_dfs:
                print("❌ No data for training.")
                return False

            combined = pd.concat(all_dfs, ignore_index=True)
            if 'Date' in combined.columns:
                combined = combined.drop(columns=['Date'])
            if 'index' in combined.columns:
                combined = combined.drop(columns=['index'])

            X, _, _, _ = pattern_model.prepare_features(combined)
            if X is None or len(X) < 60:
                print("⚠️ Not enough data for training, skipping.")
                return False

            success = pattern_model.train(combined)
            if success:
                print("✅ SVM+HMM model trained successfully from year-to-date data.")
            else:
                print("❌ Training failed.")
            return success
        except Exception as e:
            print(f"❌ Training error: {e}")
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
        print("🚀 Auto‑retrain thread started (once per day, full YTD retrain).")

# ---------- Fixed profit / fixed volume SL/TP computation (1:1.5 ratio) ----------
def compute_tp_sl(price, pair, signal, volume=FIXED_VOLUME,
                  target_profit=TARGET_PROFIT_DOLLARS,
                  ratio=REWARD_RATIO):
    """
    Compute SL and TP prices for a given ratio and fixed profit target.
    TP distance is set to achieve the target profit with fixed volume.
    SL distance = TP distance / ratio.
    """
    pip_value = get_pip_value(pair, volume=1.0)
    if pair.endswith('JPY'):
        pip_size = 0.01
    else:
        pip_size = 0.0001

    # Distance (in price units) needed for target profit
    tp_pips = target_profit / (pip_value * volume)
    tp_distance = tp_pips * pip_size

    # SL distance = TP / ratio
    sl_distance = tp_distance / ratio

    if signal == 'BUY':
        sl = price - sl_distance
        tp = price + tp_distance
    else:  # SELL
        sl = price + sl_distance
        tp = price - tp_distance

    return sl, tp

# ---------- Trade Readiness Check ----------
def is_trade_ready(pair, price, signal, atr, df,
                   spread_threshold=0.0002,
                   max_price_dev=0.005,
                   min_atr_ratio=0.3,
                   max_atr_ratio=3.0,
                   lookback=20):
    bid, ask = get_live_bid_ask(pair)
    if bid is not None and ask is not None:
        current_spread = ask - bid
        if current_spread > spread_threshold:
            return False, f"Spread too wide: {current_spread:.5f}"

    if len(df) > 0:
        last_close = df['close'].iloc[-1]
        if last_close != 0:
            dev = abs(price - last_close) / last_close
            if dev > max_price_dev:
                return False, f"Price deviated {dev:.3%} > {max_price_dev:.3%}"

    if 'atr' not in df.columns:
        return False, "ATR column missing in DataFrame"
    atr_series = df['atr'].dropna()
    if len(atr_series) >= lookback:
        avg_atr = atr_series.iloc[-lookback:].mean()
        if avg_atr > 0:
            ratio = atr / avg_atr
            if ratio < min_atr_ratio or ratio > max_atr_ratio:
                return False, f"ATR ratio {ratio:.2f} outside [{min_atr_ratio}, {max_atr_ratio}]"

    return True, "Ready"

def process_pair(pair, interval, atr_period, risk_mult, reward_ratio):
    recent_start, recent_end = get_date_ranges()
    print(f"🔍 Processing {pair}...")
    try:
        df = fetch_data(pair, recent_start, recent_end, interval)
        if df.empty or len(df) < 60:
            return None

        macro_start = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
        macro_end = datetime.now().strftime('%Y-%m-%d')
        macro_df = fetch_macro_data(macro_start, macro_end)
        commodity_df = fetch_commodity_data(macro_start, macro_end)

        if not macro_df.empty:
            latest = macro_df.iloc[-1]
            for col in macro_df.columns:
                df[col] = latest.get(col, np.nan)
            df = df.ffill()

        if not commodity_df.empty:
            latest = commodity_df.iloc[-1]
            for col in ['crude_oil', 'gold', 'agri']:
                df[col] = latest.get(col, np.nan)
            df = df.ffill()

        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
        df = df.dropna()
        if df.empty:
            return None

        signal, conf = pattern_model.predict_pattern(df)
        print(f"🔍 {pair} raw confidence: {conf:.4f}")

        min_conf = get_min_confidence()
        if conf < min_conf:
            signal = 'HOLD'
            conf = 0.0

        current_atr = df['atr'].iloc[-1]
        live_price, live_ok = get_live_entry_price(pair, signal)
        if live_ok:
            reference_price = live_price
        else:
            reference_price = df['close'].iloc[-1]

        # Compute SL/TP using fixed volume and 1:1.5 ratio
        if signal != 'HOLD':
            raw_sl, raw_tp = compute_tp_sl(reference_price, pair, signal,
                                           volume=FIXED_VOLUME,
                                           target_profit=TARGET_PROFIT_DOLLARS,
                                           ratio=REWARD_RATIO)
            adjusted_sl, adjusted_tp = adjust_sl_tp(pair, reference_price, raw_sl, raw_tp, ratio=REWARD_RATIO)
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

        volume_for_profit = FIXED_VOLUME
        tp_profit = None
        sl_loss = None
        if signal != 'HOLD' and can_trade and sl is not None and tp is not None:
            tp_profit = compute_profit(reference_price, tp, pair, volume_for_profit)
            sl_loss = compute_profit(reference_price, sl, pair, volume_for_profit)

        ready, ready_reason = False, "Not applicable"
        if signal != 'HOLD' and can_trade:
            ready, ready_reason = is_trade_ready(pair, reference_price, signal, current_atr, df)

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
            "confidence": round(conf, 3) if signal != 'HOLD' else 0.0,
            "price": round(reference_price, 5),
            "tp": round(tp, 5) if tp is not None else None,
            "sl": round(sl, 5) if sl is not None else None,
            "atr": round(current_atr, 5),
            "can_trade": can_trade,
            "can_trade_reason": reason,
            "trade_ready": ready,
            "trade_ready_reason": ready_reason,
            "chart": chart_data,
            "tp_profit": round(tp_profit, 2) if tp_profit is not None else None,
            "sl_loss": round(sl_loss, 2) if sl_loss is not None else None,
            "source": "svm_hmm",
            "ml_conf": round(conf, 3),
            "rule_conf": 0.0,
            "error": None
        }
    except Exception as e:
        print(f"❌ Error processing {pair}: {e}")
        traceback.print_exc()
        return {"pair": pair, "error": str(e)[:100]}

# ---------- Trade Execution (fixed volume) ----------
def execute_trade(pair, signal, price, tp, sl, volume=TRADE_VOLUME):
    global pytrader, pytrader_connected
    if not PYTRADER_AVAILABLE:
        return {"success": False, "error": "PyTrader not available"}
    if not pytrader_connected:
        if not connect_to_mt4():
            return {"success": False, "error": "Could not connect to MT4"}
    if pair not in pytrader.instrument_conversion_list:
        pytrader.instrument_conversion_list[pair] = pair

    # Override volume with fixed volume
    volume = FIXED_VOLUME
    adjusted_volume = get_valid_lot_size(pair, volume)

    live_price, live_ok = get_live_entry_price(pair, signal)
    if not live_ok:
        ref_price = price
    else:
        ref_price = live_price

    # Re‑compute SL/TP with fixed volume & 1:1.5 ratio
    raw_sl, raw_tp = compute_tp_sl(ref_price, pair, signal,
                                   volume=FIXED_VOLUME,
                                   target_profit=TARGET_PROFIT_DOLLARS,
                                   ratio=REWARD_RATIO)
    adjusted_sl, adjusted_tp = adjust_sl_tp(pair, ref_price, raw_sl, raw_tp, ratio=REWARD_RATIO)
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
            comment="SVM_HMM",
            market=False
        )
        if ticket == -1:
            error_msg = pytrader.order_return_message or "Unknown error"
            return {"success": False, "error": f"Order failed: {error_msg}"}
        return {
            "success": True,
            "order_id": ticket,
            "message": f"{signal} {pair} executed, ticket {ticket}",
            "volume": adjusted_volume
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ---------- Auto-Trade ----------
def has_pending_trade(pair):
    ok, rows = supabase_request('GET', f'trades?pair=eq.{pair}&result=eq.pending')
    if ok and rows:
        return len(rows) > 0
    return False

def auto_trade_iteration():
    if not auto_trade_enabled:
        return
    print(f"[Auto-Trade] Running at {datetime.now().strftime('%H:%M:%S')}")
    interval = '1h'
    atr_period = 14
    risk_mult = RISK_ATR
    reward_ratio = REWARD_RATIO
    volume = FIXED_VOLUME
    pairs = auto_trade_pairs
    for pair in pairs:
        yf_pair = pair if '=X' in pair else pair + '=X'
        res = process_pair(yf_pair, interval, atr_period, risk_mult, reward_ratio)
        if res and res.get('signal') in ('BUY', 'SELL'):
            if not res.get('can_trade', False):
                print(f"[Auto-Trade] ⏳ {pair} signal exists but can_trade=False: {res.get('can_trade_reason')}")
                continue
            if not res.get('trade_ready', False):
                print(f"[Auto-Trade] ⏳ {pair} signal ready but timing not perfect: {res.get('trade_ready_reason')}")
                continue
            if has_pending_trade(pair):
                print(f"[Auto-Trade] ⏭️ {pair} already has a pending trade – skipping")
                continue
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
                    source='svm_hmm',
                    ml_conf=res.get('ml_conf', 0.0),
                    rule_conf=0.0
                )
                print(f"[Auto-Trade] ✅ {res['signal']} {res['pair']} executed")
            else:
                print(f"[Auto-Trade] ❌ {res['pair']} failed: {trade_res.get('error')}")

def auto_trade_loop():
    while True:
        if auto_trade_enabled:
            auto_trade_iteration()
        time.sleep(30)

def start_auto_trade_thread():
    global auto_trade_thread
    if auto_trade_thread is None or not auto_trade_thread.is_alive():
        auto_trade_thread = threading.Thread(target=auto_trade_loop, daemon=True)
        auto_trade_thread.start()
        print("🚀 Auto-trade thread started (runs every 30s).")

# ---------- Flask Routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/signals', methods=['POST'])
def get_signals():
    data = request.get_json()
    DEFAULT_PAIRS = ','.join([p + '=X' for p in ALL_PAIRS])
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
    DEFAULT_PAIRS = ','.join([p + '=X' for p in ALL_PAIRS])
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
            if res['signal'] in ('BUY', 'SELL') and res.get('can_trade', False) and res.get('trade_ready', False):
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
                        source='svm_hmm',
                        ml_conf=res.get('ml_conf', 0.0),
                        rule_conf=0.0
                    )
            else:
                res['trade'] = {"success": False, "error": "No valid signal, invalid SL/TP, or timing not ready"}
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
        allowed = ALL_PAIRS
        auto_trade_pairs = [p for p in pair_list if p in allowed]
        if not auto_trade_pairs:
            auto_trade_pairs = ALL_PAIRS
    else:
        auto_trade_pairs = ALL_PAIRS
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
    try:
        success = retrain_model()
        return jsonify({'success': success, 'message': 'Retraining completed' if success else 'Retraining failed'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ---------- Startup ----------
if __name__ == '__main__':
    init_db()
    print(f"✅ Database ready. Min confidence: {get_min_confidence()}")
    print("🔧 Signal mode: SVM+HMM hybrid (lightweight)")
    print(f"📊 Fixed volume: {FIXED_VOLUME} lot, target ${TARGET_PROFIT_DOLLARS} profit, 1:1.5 ratio (SL:TP)")
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