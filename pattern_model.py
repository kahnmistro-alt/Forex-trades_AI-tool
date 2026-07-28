# pattern_model.py – Lightweight SVM+HMM hybrid
import pickle
import base64
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import warnings
import io
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hmmlearn import hmm

warnings.filterwarnings('ignore')

class PatternModel:
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.svm = None
        self.hmm = None
        self.scaler = None
        self.feature_columns = [
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
        self.load_latest_model()

    def load_latest_model(self):
        try:
            resp = self.supabase.table('pattern_models') \
                .select('*') \
                .order('created_at', desc=True) \
                .limit(1) \
                .execute()
            if not resp.data:
                print("ℹ️ No existing SVM+HMM model found. Will train on first data.")
                return False

            blob_b64 = resp.data[0]['model_blob']
            blob = base64.b64decode(blob_b64)
            data = pickle.loads(blob)

            if 'svm' in data and data['svm'] is not None:
                self.svm = joblib.load(io.BytesIO(data['svm']))
                self.hmm = joblib.load(io.BytesIO(data['hmm']))
                self.scaler = joblib.load(io.BytesIO(data['scaler']))
                print("✅ SVM+HMM model loaded from Supabase.")
                return True
            else:
                print("ℹ️ No SVM+HMM model found in saved data.")
                self.svm = None
                self.hmm = None
                self.scaler = None
                return False
        except Exception as e:
            print(f"⚠️ Failed to load SVM+HMM model: {e}. Will retrain.")
            self.svm = None
            self.hmm = None
            self.scaler = None
            return False

    def save_model(self, svm_bytes, hmm_bytes, scaler_bytes):
        try:
            resp = self.supabase.table('pattern_models') \
                .select('id') \
                .order('created_at', desc=True) \
                .limit(1) \
                .execute()

            save_data = {
                'model_blob': base64.b64encode(pickle.dumps({
                    'svm': svm_bytes,
                    'hmm': hmm_bytes,
                    'scaler': scaler_bytes
                })).decode('utf-8'),
                'created_at': datetime.now().isoformat(),
                'version': 'svm_hmm_1.0'
            }

            if resp.data:
                row_id = resp.data[0]['id']
                self.supabase.table('pattern_models') \
                    .eq('id', row_id) \
                    .update(save_data)
                print("✅ SVM+HMM model updated in Supabase.")
            else:
                self.supabase.table('pattern_models').insert(save_data)
                print("✅ SVM+HMM model saved to Supabase.")
            return True
        except Exception as e:
            print(f"❌ Failed to save SVM+HMM model: {e}")
            return False

    def prepare_features(self, df):
        if len(df) < 20:
            return None, None, None, None
        df = df.copy()

        # Technical indicators (same as before)
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd'] = macd_df['MACD_12_26_9'] if macd_df is not None else 0
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

        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None and not stoch.empty:
            df['stoch_k'] = stoch.get('STOCHk_14_3_3', np.nan)
            df['stoch_d'] = stoch.get('STOCHd_14_3_3', np.nan)
        else:
            df['stoch_k'] = np.nan
            df['stoch_d'] = np.nan

        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        df['cci_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx_14'] = adx['ADX_14'] if adx is not None else np.nan

        df['volatility_20'] = ret.rolling(20).std()
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_20'].rolling(10).mean()
        df['volatility_ratio'] = df['volatility_ratio'].replace([np.inf, -np.inf], np.nan)

        for lag in [1, 2]:
            df[f'open_prev_{lag}'] = df['open'].shift(lag)
            df[f'high_prev_{lag}'] = df['high'].shift(lag)
            df[f'low_prev_{lag}'] = df['low'].shift(lag)
            df[f'close_prev_{lag}'] = df['close'].shift(lag)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].ffill()
        df[self.feature_columns] = df[self.feature_columns].fillna(0)

        usable_features = self.feature_columns
        df_clean = df.dropna(subset=usable_features)
        if df_clean.empty:
            return None, None, None, None

        X = df_clean[usable_features].values
        idx = df_clean.index
        return X, df_clean, idx, usable_features

    def train(self, df):
        X, df_clean, idx, usable_features = self.prepare_features(df)
        if X is None:
            return False

        # Create labels: 1 for up (>0.1%), -1 for down (<-0.1%), 0 for neutral
        future_ret = df_clean['close'].shift(-1) / df_clean['close'] - 1
        y = np.where(future_ret > 0.001, 1, np.where(future_ret < -0.001, -1, 0))
        # Use only non-neutral for binary classification
        mask = y != 0
        X_bin = X[mask]
        y_bin = (y[mask] > 0).astype(int)

        if len(X_bin) < 50:
            print(f"Not enough non‑neutral samples for SVM: {len(X_bin)}")
            return False

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_bin)

        # Train SVM
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        svm.fit(X_scaled, y_bin)

        # Train HMM on all data (including neutral) for regime detection
        X_all_scaled = scaler.transform(X)
        hmm_model = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=100, random_state=42)
        hmm_model.fit(X_all_scaled)

        # Serialize
        svm_bytes = io.BytesIO()
        joblib.dump(svm, svm_bytes)
        svm_bytes = svm_bytes.getvalue()

        hmm_bytes = io.BytesIO()
        joblib.dump(hmm_model, hmm_bytes)
        hmm_bytes = hmm_bytes.getvalue()

        scaler_bytes = io.BytesIO()
        joblib.dump(scaler, scaler_bytes)
        scaler_bytes = scaler_bytes.getvalue()

        self.svm = svm
        self.hmm = hmm_model
        self.scaler = scaler
        self.save_model(svm_bytes, hmm_bytes, scaler_bytes)
        return True

    def predict_pattern(self, df):
        if self.svm is None or self.hmm is None or self.scaler is None:
            return 'HOLD', 0.0

        X, df_clean, idx, usable_features = self.prepare_features(df)
        if X is None or len(df_clean) < 20:
            return 'HOLD', 0.0

        # Use the latest row's features
        last_row = X[-1:].reshape(1, -1)
        X_scaled = self.scaler.transform(last_row)

        # SVM prediction
        prob = self.svm.predict_proba(X_scaled)[0]
        pred = self.svm.predict(X_scaled)[0]
        # Confidence from SVM probability of the predicted class
        svm_conf = prob[1] if pred == 1 else prob[0]

        # HMM regime
        state_probs = self.hmm.predict_proba(X_scaled)[0]
        dominant_state = np.argmax(state_probs)

        # Adjust confidence based on regime (as in backtest.py)
        # Assuming state 0 = bearish, 1 = neutral, 2 = bullish (can be inferred)
        if pred == 1:  # BUY
            if dominant_state == 2:   # bullish regime
                conf = min(1.0, svm_conf + 0.15)
            elif dominant_state == 0: # bearish regime
                conf = max(0.0, svm_conf - 0.15)
            else:
                conf = svm_conf
        else:  # SELL
            if dominant_state == 0:   # bearish regime
                conf = min(1.0, svm_conf + 0.15)
            elif dominant_state == 2: # bullish regime
                conf = max(0.0, svm_conf - 0.15)
            else:
                conf = svm_conf

        signal = 'BUY' if pred == 1 else 'SELL'
        return signal, float(conf)