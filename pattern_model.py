import pickle
import base64
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from hmmlearn import hmm
import pandas_ta as ta
from datetime import datetime
import json

class PairModelManager:
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.models = {}  # pair -> {'type': type, 'model': model, 'scaler': scaler}
        # 25 features: OHLCV + lags + indicators + BB + ROC + returns stats
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'open_prev_1', 'high_prev_1', 'low_prev_1', 'close_prev_1',
            'open_prev_2', 'high_prev_2', 'low_prev_2', 'close_prev_2',
            'rsi_14', 'macd', 'atr_14',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'roc_10', 'roc_20',
            'returns_std_10', 'returns_skew_10', 'returns_kurt_10'
        ]

    def prepare_features(self, df):
        """
        Compute all features and return X matrix and clean index.
        """
        if len(df) < 20:
            return None, None
        df = df.copy()
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
        for lag in [1, 2]:
            df[f'open_prev_{lag}'] = df['open'].shift(lag)
            df[f'high_prev_{lag}'] = df['high'].shift(lag)
            df[f'low_prev_{lag}'] = df['low'].shift(lag)
            df[f'close_prev_{lag}'] = df['close'].shift(lag)
        df_clean = df.dropna()
        X = df_clean[self.feature_columns].values
        return X, df_clean.index

    def load_model(self, pair, model_type):
        """
        Load the latest model for a pair and model_type from Supabase.
        Returns True on success.
        """
        if model_type == 'rule':
            self.models[pair] = {'type': 'rule', 'model': None, 'scaler': None}
            return True
        try:
            resp = self.supabase.table('pair_models') \
                .select('*') \
                .eq('pair', pair) \
                .eq('model_type', model_type) \
                .order('created_at', desc=True) \
                .limit(1) \
                .execute()
            if resp.data:
                blob_b64 = resp.data[0]['model_blob']
                blob = base64.b64decode(blob_b64)
                data = pickle.loads(blob)
                self.models[pair] = {
                    'type': model_type,
                    'model': data['model'],
                    'scaler': data['scaler']
                }
                print(f"✅ Loaded {model_type} model for {pair}")
                return True
            else:
                print(f"ℹ️ No existing model for {pair} ({model_type}).")
                return False
        except Exception as e:
            print(f"⚠️ Failed to load model for {pair}: {e}")
            return False

    def save_model(self, pair, model_type, model, scaler, metrics=None):
        """
        Save a trained model to Supabase.
        """
        try:
            save_data = {'model': model, 'scaler': scaler}
            blob = pickle.dumps(save_data)
            blob_b64 = base64.b64encode(blob).decode('utf-8')
            data = {
                'pair': pair,
                'model_type': model_type,
                'model_blob': blob_b64,
                'created_at': datetime.now().isoformat(),
                'version': '1.0',
                'metrics': json.dumps(metrics) if metrics else None
            }
            self.supabase.table('pair_models').insert(data)
            print(f"✅ Saved {model_type} model for {pair}")
            self.models[pair] = {'type': model_type, 'model': model, 'scaler': scaler}
            return True
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            return False

    def get_model(self, pair):
        """Return (model, scaler) for a pair, or (None, None) if rule."""
        entry = self.models.get(pair)
        if entry and entry['type'] != 'rule':
            return entry['model'], entry['scaler']
        return None, None

    def get_model_type(self, pair):
        return self.models.get(pair, {}).get('type', 'rule')

    def train_model(self, df, model_type):
        """
        Train a model of the specified type on the given DataFrame.
        Returns (model, scaler, accuracy) or (None, None, 0.0) on failure.
        """
        X, idx = self.prepare_features(df)
        if X is None:
            return None, None, 0.0
        df_clean = df.loc[idx]
        future_ret = df_clean['close'].shift(-1) / df_clean['close'] - 1
        y = np.where(future_ret > 0.001, 1, np.where(future_ret < -0.001, -1, 0))
        mask = y != 0
        X_bin = X[mask]
        y_bin = (y[mask] > 0).astype(int)
        if len(X_bin) < 50 or len(np.unique(y_bin)) < 2:
            print(f"Not enough samples for {model_type} training.")
            return None, None, 0.0

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_bin)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_bin, test_size=0.2, random_state=42)

        if model_type == 'xgboost':
            model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                      use_label_encoder=False, eval_metric='logloss',
                                      random_state=42)
            model.fit(X_train, y_train)
            acc = model.score(X_val, y_val)
            print(f"✅ XGBoost validation accuracy: {acc:.2f}")
            return model, scaler, acc

        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            acc = model.score(X_val, y_val)
            print(f"✅ Random Forest validation accuracy: {acc:.2f}")
            return model, scaler, acc

        elif model_type == 'svm_hmm':
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
            svm.fit(X_train, y_train)
            acc = svm.score(X_val, y_val)
            print(f"✅ SVM validation accuracy: {acc:.2f}")
            X_all_scaled = scaler.transform(X)
            hmm_model = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=100, random_state=42)
            hmm_model.fit(X_all_scaled)
            print("✅ HMM trained.")
            model = {'svm': svm, 'hmm': hmm_model}
            return model, scaler, acc
        else:
            return None, None, 0.0

    def predict(self, pair, df):
        """
        Predict signal and confidence using the pair's model.
        Returns (signal, confidence, model_type) or (None, None, reason).
        """
        model_type = self.get_model_type(pair)
        if model_type == 'rule':
            return None, None, 'rule'
        model, scaler = self.get_model(pair)
        if model is None:
            return None, None, 'no_model'
        X, idx = self.prepare_features(df)
        if X is None or len(X) == 0:
            return None, None, 'no_data'
        last_X = X[-1].reshape(1, -1)
        last_X_scaled = scaler.transform(last_X)

        if model_type in ('xgboost', 'rf'):
            prob = model.predict_proba(last_X_scaled)[0]
            pred = model.predict(last_X_scaled)[0]
            conf = prob[1] if pred == 1 else prob[0]
            signal = 'BUY' if pred == 1 else 'SELL'
            return signal, conf, model_type

        elif model_type == 'svm_hmm':
            svm = model['svm']
            hmm_model = model['hmm']
            prob = svm.predict_proba(last_X_scaled)[0]
            pred = svm.predict(last_X_scaled)[0]
            conf = prob[1] if pred == 1 else prob[0]
            state_probs = hmm_model.predict_proba(last_X_scaled)[0]
            dominant_state = np.argmax(state_probs)
            if pred == 1:
                if dominant_state == 2:      # uptrend
                    confidence = min(1.0, conf + 0.15)
                elif dominant_state == 0:    # downtrend
                    confidence = max(0.0, conf - 0.15)
                else:
                    confidence = conf
            else:
                if dominant_state == 0:
                    confidence = min(1.0, conf + 0.15)
                elif dominant_state == 2:
                    confidence = max(0.0, conf - 0.15)
                else:
                    confidence = conf
            signal = 'BUY' if pred == 1 else 'SELL'
            return signal, confidence, model_type
        else:
            return None, None, 'unknown'