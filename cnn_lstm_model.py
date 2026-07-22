# cnn_lstm_model.py – Keras 3 compatible (no save_format)
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import joblib
import os
import tempfile

class CNNLSTMClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid CNN-LSTM model for Forex pattern detection.
    """
    def __init__(self, window_size=60, n_features=20, lstm_units=64,
                 cnn_filters=32, kernel_size=3, dropout=0.2,
                 learning_rate=0.001, epochs=50, batch_size=64,
                 calibration=None, early_stopping_patience=5):
        self.window_size = window_size
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.calibration = calibration
        self.early_stopping_patience = early_stopping_patience
        self.scaler = StandardScaler()
        self.model = None
        self.calibrated_model = None

    def _build_model(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv1D(filters=self.cnn_filters, kernel_size=self.kernel_size,
                          activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(self.dropout)(x)

        x = layers.LSTM(self.lstm_units, return_sequences=False)(x)
        x = layers.Dropout(self.dropout)(x)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.scaler.fit(X.reshape(-1, X.shape[-1]))
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        self.model = self._build_model((self.window_size, X.shape[-1]))

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )

        self.model.fit(X_scaled, y,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       validation_split=0.1,
                       verbose=1,
                       callbacks=[early_stop])

        if self.calibration == 'platt':
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.base import clone
            class SklearnWrapper(BaseEstimator, ClassifierMixin):
                def __init__(self, model):
                    self.model = model
                def fit(self, X, y):
                    self.model.fit(X, y)
                    return self
                def predict_proba(self, X):
                    return np.hstack([1 - self.model.predict(X), self.model.predict(X)])
                def get_params(self, deep=True):
                    return {'model': self.model}
                def set_params(self, **params):
                    self.model = params.get('model', self.model)
                    return self
            base = SklearnWrapper(self.model)
            self.calibrated_model = CalibratedClassifierCV(base, method='sigmoid', cv=3)
            self.calibrated_model.fit(X_scaled, y)
        else:
            self.calibrated_model = None
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X_scaled)
        else:
            raw = self.model.predict(X_scaled)
            return np.hstack([1 - raw, raw])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def save(self, path):
        # Keras 3: use .keras extension (no save_format)
        self.model.save(os.path.join(path, 'cnn_lstm.keras'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        if self.calibrated_model is not None:
            joblib.dump(self.calibrated_model, os.path.join(path, 'calibrated.pkl'))

    def load(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'cnn_lstm.keras'))
        self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        cal_path = os.path.join(path, 'calibrated.pkl')
        if os.path.exists(cal_path):
            self.calibrated_model = joblib.load(cal_path)

    def get_model_bytes(self):
        """
        Serialize model to bytes using a temporary .keras file.
        Keras 3 no longer supports save_format in BytesIO.
        """
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
            temp_path = tmp_file.name
        try:
            self.model.save(temp_path)  # uses .keras extension
            with open(temp_path, 'rb') as f:
                model_bytes = f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        return model_bytes

    @classmethod
    def from_bytes(cls, model_bytes, scaler_bytes=None, cal_bytes=None):
        """
        Load model from bytes saved from a .keras file.
        """
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
            tmp_file.write(model_bytes)
            temp_path = tmp_file.name
        try:
            model = tf.keras.models.load_model(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        instance = cls()
        instance.model = model
        if scaler_bytes:
            import io
            instance.scaler = joblib.load(io.BytesIO(scaler_bytes))
        if cal_bytes:
            instance.calibrated_model = joblib.load(io.BytesIO(cal_bytes))
        return instance