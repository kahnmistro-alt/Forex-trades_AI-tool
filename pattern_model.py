import pickle
import base64
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from hmmlearn import hmm
import pandas_ta as ta
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("ℹ️ imbalanced-learn not installed. Using class_weight='balanced' only (no SMOTE).")

class PatternModel:
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.scaler = StandardScaler()
        self.svm = None
        self.hmm_model = None
        self.selected_features = None
        self.current_val_acc = 0.0
        self.feature_columns = [
            # OHLC + lags
            'open', 'high', 'low', 'close',
            'open_prev_1', 'high_prev_1', 'low_prev_1', 'close_prev_1',
            'open_prev_2', 'high_prev_2', 'low_prev_2', 'close_prev_2',

            # Standard indicators
            'rsi_14', 'macd', 'atr_14',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'roc_10', 'roc_20',
            'returns_std_10', 'returns_skew_10', 'returns_kurt_10',

            # Momentum
            'stoch_k', 'stoch_d', 'williams_r', 'cci_20', 'adx_14',

            # Volatility
            'volatility_20', 'volatility_ratio',

            # Commodities
            'crude_oil', 'gold', 'agri',

            # ---- Macro: yield spreads ----
            'spread_us_de', 'spread_us_uk', 'spread_us_au', 'spread_us_ca',
            'spread_us_ch', 'spread_de_uk', 'spread_de_jp', 'spread_us_nz',
            'spread_uk_jp', 'spread_us_jp',

            # ---- Macro: VIX ----
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
                print("ℹ️ No existing model found. Will train on first data.")
                return False

            blob_b64 = resp.data[0]['model_blob']
            blob = base64.b64decode(blob_b64)
            data = pickle.loads(blob)

            required_keys = ['scaler', 'svm', 'hmm', 'selected_features', 'val_acc']
            for key in required_keys:
                if key not in data:
                    raise KeyError(f"Missing key '{key}' in model data. Old model format?")
            
            self.scaler = data['scaler']
            self.svm = data['svm']
            self.hmm_model = data['hmm']
            self.selected_features = data['selected_features']
            self.current_val_acc = data['val_acc']
            print(f"✅ SVM+HMM model loaded from Supabase (val acc: {self.current_val_acc:.3f})")
            return True

        except KeyError as e:
            print(f"⚠️ Model loading failed due to missing key: {e}. Will retrain a new model.")
            self.svm = None
            self.hmm_model = None
            self.selected_features = None
            self.current_val_acc = 0.0
            return False
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}. Will retrain a new model.")
            self.svm = None
            self.hmm_model = None
            self.selected_features = None
            self.current_val_acc = 0.0
            return False

    def save_model(self, validation_acc):
        if self.svm is None or self.hmm_model is None:
            print("No model to save.")
            return False
        if validation_acc <= self.current_val_acc:
            print(f"Validation acc {validation_acc:.3f} not better than current {self.current_val_acc:.3f}. Skipping save.")
            return False
        try:
            save_data = {
                'scaler': self.scaler,
                'svm': self.svm,
                'hmm': self.hmm_model,
                'selected_features': self.selected_features,
                'val_acc': validation_acc
            }
            blob = pickle.dumps(save_data)
            blob_b64 = base64.b64encode(blob).decode('utf-8')
            data = {
                'model_blob': blob_b64,
                'created_at': datetime.now().isoformat(),
                'version': '10.0'
            }
            self.supabase.table('pattern_models').insert(data)
            self.current_val_acc = validation_acc
            print(f"✅ SVM+HMM model saved to Supabase (val acc: {validation_acc:.3f}).")
            return True
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            return False

    def prepare_features(self, df):
        if len(df) < 20:
            return None, None
        df = df.copy()

        # ---- Existing indicators ----
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

        # ---- Momentum ----
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

        # ---- Volatility ----
        df['volatility_20'] = ret.rolling(20).std()
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_20'].rolling(10).mean()
        df['volatility_ratio'] = df['volatility_ratio'].replace([np.inf, -np.inf], np.nan)

        # ---- Lags ----
        for lag in [1, 2]:
            df[f'open_prev_{lag}'] = df['open'].shift(lag)
            df[f'high_prev_{lag}'] = df['high'].shift(lag)
            df[f'low_prev_{lag}'] = df['low'].shift(lag)
            df[f'close_prev_{lag}'] = df['close'].shift(lag)

        # ---- Commodity & Macro features ----
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df[col] = df[col].ffill()

        # Diagnostic: dropna() below drops on ANY NaN across the whole frame,
        # so a single all-NaN feature column (e.g. a failed macro/VIX fetch)
        # would silently wipe every row and make training look like "not
        # enough data" with no explanation. Surface that here instead.
        nan_frac = df[self.feature_columns].isna().mean()
        bad_cols = nan_frac[nan_frac > 0.95]
        if not bad_cols.empty:
            print(f"⚠️ prepare_features: dropping unusable (mostly-NaN) columns: {list(bad_cols.index)}")
            df = df.drop(columns=list(bad_cols.index))

        usable_features = [c for c in self.feature_columns if c in df.columns]
        df_clean = df.dropna(subset=usable_features)
        if df_clean.empty:
            print("⚠️ prepare_features: no rows left after dropping NaNs — check data feeds.")
            return None, None
        X = df_clean[usable_features].values
        if self.selected_features is not None:
            # selected_features are positional indices into the ORIGINAL
            # self.feature_columns ordering — only valid if no columns were
            # dropped above. If the feature set shrank, force reselection
            # rather than silently indexing into the wrong columns.
            if len(usable_features) == len(self.feature_columns):
                X = X[:, self.selected_features]
            else:
                print("⚠️ prepare_features: feature set changed since last selection; "
                      "ignoring cached selected_features for this call.")
        return X, df_clean.index

    def feature_selection(self, X, y, n_features=20):
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[-n_features:]
        print("Selected feature indices:", indices)
        return indices

    def train(self, df, labels=None):
        X, idx = self.prepare_features(df)
        if X is None:
            return False
        df_clean = df.iloc[idx]
        if labels is None:
            future_ret = df_clean['close'].shift(-1) / df_clean['close'] - 1
            y = np.where(future_ret > 0.001, 1, np.where(future_ret < -0.001, -1, 0))
        else:
            if isinstance(labels, pd.Series):
                y = labels.iloc[idx].values
            else:
                y = np.array(labels)[idx]

        mask = y != 0
        X_bin = X[mask]
        y_bin = (y[mask] > 0).astype(int)

        if len(X_bin) < 30:
            print("Not enough trading samples for training.")
            return False

        if self.selected_features is None:
            self.selected_features = self.feature_selection(X_bin, y_bin, n_features=20)
            X_bin = X_bin[:, self.selected_features]
            X = X[:, self.selected_features]

        X_train, X_val, y_train, y_val = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

        if SMOTE_AVAILABLE:
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            print("✅ SMOTE applied to balance classes.")
        else:
            X_train_res, y_train_res = X_train, y_train
            print("ℹ️ Using class_weight='balanced' (SMOTE unavailable).")

        self.scaler.fit(X_train_res)
        X_train_scaled = self.scaler.transform(X_train_res)
        X_val_scaled = self.scaler.transform(X_val)
        X_all_scaled = self.scaler.transform(X)

        param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.01, 0.1]}
        svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
        grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_scaled, y_train_res)
        best_svm = grid.best_estimator_
        val_acc = accuracy_score(y_val, best_svm.predict(X_val_scaled))
        print(f"Best SVM params: {grid.best_params_} (val acc: {val_acc:.3f})")

        best_hmm = None
        best_score = -np.inf
        for n_comp in [2, 3, 4]:
            hmm_model = hmm.GaussianHMM(n_components=n_comp, covariance_type='full', n_iter=100, random_state=42)
            hmm_model.fit(X_all_scaled)
            score = hmm_model.score(X_all_scaled)
            if score > best_score:
                best_score = score
                best_hmm = hmm_model
        print(f"HMM chosen with {best_hmm.n_components} states (log-lik: {best_score:.2f})")

        self.svm = best_svm
        self.hmm_model = best_hmm
        self.save_model(val_acc)
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
        ml_conf = svm_prob[1] if svm_pred == 1 else svm_prob[0]

        state_probs = self.hmm_model.predict_proba(last_X_scaled)[0]
        dominant_state = np.argmax(state_probs)

        if svm_pred == 1:
            if dominant_state == 2:
                ml_conf = min(1.0, ml_conf + 0.15)
            elif dominant_state == 0:
                ml_conf = max(0.0, ml_conf - 0.15)
        else:
            if dominant_state == 0:
                ml_conf = min(1.0, ml_conf + 0.15)
            elif dominant_state == 2:
                ml_conf = max(0.0, ml_conf - 0.15)

        if ml_conf > 0.7:
            signal = 'BUY' if svm_pred == 1 else 'SELL'
        else:
            signal = 'HOLD'
        return signal, float(ml_conf)

    def fuse_signals(self, ml_signal, ml_conf, rule_signal, rule_conf, rule_weight):
        if ml_signal == 'HOLD' and rule_signal == 'HOLD':
            return 'HOLD', 0.0
        if ml_signal == 'HOLD':
            return rule_signal, rule_conf * rule_weight
        if rule_signal == 'HOLD':
            return ml_signal, ml_conf * (1 - rule_weight)
        if ml_signal == rule_signal:
            conf = (1 - rule_weight) * ml_conf + rule_weight * rule_conf
            return ml_signal, min(1.0, conf)
        else:
            ml_weighted = (1 - rule_weight) * ml_conf
            rule_weighted = rule_weight * rule_conf
            if ml_weighted > rule_weighted:
                return ml_signal, ml_weighted
            else:
                return rule_signal, rule_weighted