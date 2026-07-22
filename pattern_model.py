# pattern_model.py – CNN+LSTM only (fixed joblib serialization)
import pickle
import base64
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import warnings
import io
import joblib

warnings.filterwarnings('ignore')

from cnn_lstm_model import CNNLSTMClassifier

class PatternModel:
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.cnn_lstm = None
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
                print("ℹ️ No existing CNN model found. Will train on first data.")
                return False

            blob_b64 = resp.data[0]['model_blob']
            blob = base64.b64decode(blob_b64)
            data = pickle.loads(blob)

            if 'cnn_lstm' in data and data['cnn_lstm'] is not None:
                cnn_data = data['cnn_lstm']
                self.cnn_lstm = CNNLSTMClassifier.from_bytes(
                    model_bytes=cnn_data['model_bytes'],
                    scaler_bytes=cnn_data.get('scaler_bytes'),
                    cal_bytes=cnn_data.get('cal_bytes')
                )
                print("✅ CNN+LSTM model loaded from Supabase.")
                return True
            else:
                print("ℹ️ No CNN+LSTM model found in saved data.")
                self.cnn_lstm = None
                return False
        except Exception as e:
            print(f"⚠️ Failed to load CNN model: {e}. Will retrain.")
            self.cnn_lstm = None
            return False

    def save_model(self, cnn_model_bytes, cnn_scaler_bytes, cnn_cal_bytes=None):
        try:
            save_data = {
                'cnn_lstm': {
                    'model_bytes': cnn_model_bytes,
                    'scaler_bytes': cnn_scaler_bytes,
                    'cal_bytes': cnn_cal_bytes
                }
            }
            blob = pickle.dumps(save_data)
            blob_b64 = base64.b64encode(blob).decode('utf-8')
            data = {
                'model_blob': blob_b64,
                'created_at': datetime.now().isoformat(),
                'version': '12.0'
            }
            self.supabase.table('pattern_models').insert(data)
            print("✅ CNN+LSTM model saved to Supabase.")
            return True
        except Exception as e:
            print(f"❌ Failed to save CNN model: {e}")
            return False

    def prepare_features(self, df):
        if len(df) < 20:
            return None, None, None, None
        df = df.copy()

        # Technical indicators
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

        # Macro/commodity columns (assumed already merged)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df[col] = df[col].ffill()

        # Drop columns with >95% NaN
        nan_frac = df[self.feature_columns].isna().mean()
        bad_cols = nan_frac[nan_frac > 0.95]
        if not bad_cols.empty:
            print(f"⚠️ prepare_features: dropping unusable columns: {list(bad_cols.index)}")
            df = df.drop(columns=list(bad_cols.index))

        usable_features = [c for c in self.feature_columns if c in df.columns]
        df_clean = df.dropna(subset=usable_features)
        if df_clean.empty:
            print("⚠️ prepare_features: no rows left after dropping NaNs.")
            return None, None, None, None

        X = df_clean[usable_features].values
        idx = df_clean.index
        return X, df_clean, idx, usable_features

    def create_sequences(self, df, feature_cols, window_size=60):
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

    def train(self, df, labels=None):
        X, df_clean, idx, usable_features = self.prepare_features(df)
        if X is None:
            return False

        X_seq, y_seq = self.create_sequences(df_clean, usable_features, window_size=60)
        if X_seq is None or len(X_seq) < 100:
            print(f"Not enough sequences for CNN training: {len(X_seq) if X_seq is not None else 0}")
            return False

        print(f"Training CNN+LSTM with {len(X_seq)} sequences...")
        model = CNNLSTMClassifier(window_size=60, n_features=len(usable_features),
                                  epochs=30, batch_size=64, calibration=None)
        model.fit(X_seq, y_seq)

        # Serialize using joblib with BytesIO
        model_bytes = model.get_model_bytes()
        scaler_bytes = io.BytesIO()
        joblib.dump(model.scaler, scaler_bytes)
        scaler_bytes = scaler_bytes.getvalue()

        cal_bytes = None
        if model.calibrated_model is not None:
            cal_buffer = io.BytesIO()
            joblib.dump(model.calibrated_model, cal_buffer)
            cal_bytes = cal_buffer.getvalue()

        self.cnn_lstm = model
        self.save_model(model_bytes, scaler_bytes, cal_bytes)
        return True

    def predict_pattern(self, df):
        if self.cnn_lstm is None:
            return 'HOLD', 0.0

        X, df_clean, idx, usable_features = self.prepare_features(df)
        if X is None or len(df_clean) < 60:
            return 'HOLD', 0.0

        last_window = self._get_last_window(df_clean, usable_features, window_size=60)
        if last_window is None:
            return 'HOLD', 0.0

        try:
            prob = self.cnn_lstm.predict_proba(last_window)[0]
            up_prob = prob[1]
            if up_prob > 0.5:
                signal = 'BUY'
                conf = up_prob
            else:
                signal = 'SELL'
                conf = 1 - up_prob
            return signal, float(conf)
        except Exception as e:
            print(f"⚠️ CNN prediction error: {e}")
            return 'HOLD', 0.0

    def _get_last_window(self, df, feature_cols, window_size=60):
        if len(df) < window_size:
            return None
        for col in feature_cols:
            if col not in df.columns:
                return None
        window_df = df[feature_cols].iloc[-window_size:].values
        if np.isnan(window_df).any():
            return None
        return window_df.reshape(1, window_size, len(feature_cols))