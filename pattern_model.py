import os
import pickle
import base64
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

class PatternModel:
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'open_prev_1', 'high_prev_1', 'low_prev_1', 'close_prev_1',
            'open_prev_2', 'high_prev_2', 'low_prev_2', 'close_prev_2',
            'rsi_14', 'macd', 'atr_14'
        ]
        self.load_latest_model()

    def load_latest_model(self):
        try:
            resp = self.supabase.table('pattern_models') \
                .select('*') \
                .order('created_at', desc=True) \
                .limit(1) \
                .execute()
            if resp.data:
                blob_b64 = resp.data[0]['model_blob']
                blob = base64.b64decode(blob_b64)
                self.model = pickle.loads(blob)
                print("✅ ML model loaded from Supabase.")
                return True
            else:
                print("ℹ️ No existing ML model found. Will train on first data.")
                return False
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            return False

    def save_model(self):
        if self.model is None:
            print("No model to save.")
            return False
        try:
            blob = pickle.dumps(self.model)
            blob_b64 = base64.b64encode(blob).decode('utf-8')
            data = {
                'model_blob': blob_b64,
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            # Insert without .execute() – the wrapper handles the request
            self.supabase.table('pattern_models').insert(data)
            print("✅ Model saved to Supabase.")
            return True
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            return False

    def prepare_features(self, df):
        import pandas_ta as ta
        if len(df) < 10:
            return None, None
        df = df.copy()
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd'] = macd_df['MACD_12_26_9'] if macd_df is not None else 0
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        for lag in [1, 2]:
            df[f'open_prev_{lag}'] = df['open'].shift(lag)
            df[f'high_prev_{lag}'] = df['high'].shift(lag)
            df[f'low_prev_{lag}'] = df['low'].shift(lag)
            df[f'close_prev_{lag}'] = df['close'].shift(lag)
        df_clean = df.dropna()
        if len(df_clean) < 5:
            return None, None
        X = df_clean[self.feature_columns].values
        return X, df_clean.index

    def train(self, df, labels=None):
        X, idx = self.prepare_features(df)
        if X is None:
            print("Not enough data to prepare features.")
            return False

        positions = df.index.get_indexer(idx)

        if labels is None:
            df_clean = df.iloc[positions]
            future_ret = df_clean['close'].shift(-1) / df_clean['close'] - 1
            y = np.where(future_ret > 0.001, 1, np.where(future_ret < -0.001, -1, 0))
        else:
            if isinstance(labels, pd.Series):
                y = labels.iloc[positions].values
            else:
                y = np.array(labels)[positions]

        mask = y != 0
        X_bin = X[mask]
        y_bin = (y[mask] > 0).astype(int)

        if len(X_bin) < 20:
            print("Not enough samples to train.")
            return False

        X_train, X_test, y_train, y_test = train_test_split(
            X_bin, y_bin, test_size=0.2, random_state=42
        )

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)
        score = clf.score(X_test_scaled, y_test)
        print(f"✅ Model trained with accuracy: {score:.2f}")

        self.model = clf
        self.save_model()
        return True

    def predict_pattern(self, df):
        if self.model is None:
            return 'HOLD', 0.0
        X, idx = self.prepare_features(df)
        if X is None or len(X) == 0:
            return 'HOLD', 0.0
        last_X = X[-1].reshape(1, -1)
        last_X_scaled = self.scaler.transform(last_X)
        prob = self.model.predict_proba(last_X_scaled)[0]
        pred = self.model.predict(last_X_scaled)[0]
        confidence = max(prob) if pred == 1 else 1 - max(prob)
        if confidence > 0.6:
            signal = 'BUY' if pred == 1 else 'SELL'
        else:
            signal = 'HOLD'
        return signal, confidence