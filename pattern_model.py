import os
import pickle
import base64
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from hmmlearn import hmm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PatternModel:
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.scaler = StandardScaler()
        self.svm = None
        self.hmm_model = None
        self.n_states = 3
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
                # resp.data is a list of rows
                row = resp.data[0]
                blob_b64 = row['model_blob']
                blob = base64.b64decode(blob_b64)
                data = pickle.loads(blob)
                self.scaler = data['scaler']
                self.svm = data['svm']
                self.hmm_model = data['hmm']
                print("✅ HMM+SVM model loaded from Supabase.")
                return True
            else:
                print("ℹ️ No existing ML model found. Will train on first data.")
                return False
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            return False

    def save_model(self):
        if self.svm is None or self.hmm_model is None:
            print("No model to save.")
            return False
        try:
            save_data = {
                'scaler': self.scaler,
                'svm': self.svm,
                'hmm': self.hmm_model
            }
            blob = pickle.dumps(save_data)
            blob_b64 = base64.b64encode(blob).decode('utf-8')
            data = {
                'model_blob': blob_b64,
                'created_at': datetime.now().isoformat(),
                'version': '3.0'
            }
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

        # Binary classification for SVM
        mask = y != 0
        X_bin = X[mask]
        y_bin = (y[mask] > 0).astype(int)

        if len(X_bin) < 20:
            print("Not enough samples to train SVM.")
            return False

        self.scaler.fit(X_bin)
        X_scaled = self.scaler.transform(X_bin)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_bin, test_size=0.2, random_state=42
        )

        svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        svm.fit(X_train, y_train)
        acc = svm.score(X_test, y_test)
        print(f"✅ SVM trained with accuracy: {acc:.2f}")

        # HMM on full feature set (including neutral)
        X_all_scaled = self.scaler.transform(X)
        hmm_model = hmm.GaussianHMM(n_components=self.n_states, covariance_type='full',
                                    n_iter=100, random_state=42)
        hmm_model.fit(X_all_scaled)
        print("✅ HMM trained.")

        self.svm = svm
        self.hmm_model = hmm_model
        self.save_model()
        return True

    def predict_pattern(self, df):
        if self.svm is None or self.hmm_model is None:
            return 'HOLD', 0.0

        X, idx = self.prepare_features(df)
        if X is None or len(X) == 0:
            return 'HOLD', 0.0

        last_X = X[-1].reshape(1, -1)
        last_X_scaled = self.scaler.transform(last_X)

        svm_prob = self.svm.predict_proba(last_X_scaled)[0]
        svm_pred = self.svm.predict(last_X_scaled)[0]
        svm_conf = svm_prob[1] if svm_pred == 1 else svm_prob[0]

        state_probs = self.hmm_model.predict_proba(last_X_scaled)[0]
        dominant_state = np.argmax(state_probs)

        # State 0 = downtrend, 1 = sideways, 2 = uptrend
        if svm_pred == 1:  # BUY
            if dominant_state == 2:
                confidence = min(1.0, svm_conf + 0.15)
            elif dominant_state == 0:
                confidence = max(0.0, svm_conf - 0.15)
            else:
                confidence = svm_conf
        else:  # SELL
            if dominant_state == 0:
                confidence = min(1.0, svm_conf + 0.15)
            elif dominant_state == 2:
                confidence = max(0.0, svm_conf - 0.15)
            else:
                confidence = svm_conf

        if confidence > 0.6:
            signal = 'BUY' if svm_pred == 1 else 'SELL'
        else:
            signal = 'HOLD'

        return signal, float(confidence)