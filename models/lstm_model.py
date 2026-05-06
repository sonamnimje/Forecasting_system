"""
models/lstm_model.py
--------------------
LSTM (Long Short-Term Memory) forecaster built with TensorFlow/Keras.
Uses a sliding-window approach for sequence input.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Tuple

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Input, BatchNormalization
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    _TF_OK = True
except ImportError:
    _TF_OK = False


class LSTMForecaster:
    """
    Bidirectional / stacked LSTM for weekly sales.
    Input  : sliding windows of length `lookback`
    Output : single next-step prediction (recursive for multi-step)
    """

    NAME = "LSTM"

    def __init__(
        self,
        lookback: int = 52,          # weeks of history per input window
        units: int = 64,             # LSTM units per layer
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        patience: int = 15,
        random_state: int = 42,
    ):
        if not _TF_OK:
            raise ImportError(
                "TensorFlow is not installed. Run: pip install tensorflow"
            )
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        self.lookback      = lookback
        self.units         = units
        self.dropout       = dropout
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.patience      = patience

        self.model_   = None
        self.scale_   = None   # (mean, std) for z-score normalisation
        self.history_ = None   # raw training series (for recursive pred)

    # ── normalisation ──────────────────────────────────────────────────────

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        mean, std = self.scale_
        return (x - mean) / (std + 1e-8)

    def _denormalise(self, x: np.ndarray) -> np.ndarray:
        mean, std = self.scale_
        return x * (std + 1e-8) + mean

    # ── sequence builder ───────────────────────────────────────────────────

    def _make_sequences(
        self, series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.lookback, len(series)):
            X.append(series[i - self.lookback: i])
            y.append(series[i])
        return np.array(X)[..., np.newaxis], np.array(y)

    # ── model architecture ─────────────────────────────────────────────────

    def _build_model(self) -> "tf.keras.Model":
        model = Sequential([
            Input(shape=(self.lookback, 1)),
            LSTM(self.units, return_sequences=True),
            BatchNormalization(),
            Dropout(self.dropout),
            LSTM(self.units // 2),
            BatchNormalization(),
            Dropout(self.dropout),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        model.compile(
            optimizer=Adam(self.learning_rate),
            loss="huber",
            metrics=["mae"],
        )
        return model

    # ── fit ────────────────────────────────────────────────────────────────

    def fit(self, train: pd.DataFrame) -> "LSTMForecaster":
        series = train.sort_values("date")["sales"].values.astype(float)
        self.scale_ = (series.mean(), series.std())
        norm_series = self._normalise(series)

        X, y = self._make_sequences(norm_series)

        val_split = max(1, int(len(X) * 0.15))
        X_tr, X_val = X[:-val_split], X[-val_split:]
        y_tr, y_val = y[:-val_split], y[-val_split:]

        self.model_ = self._build_model()

        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=self.patience,
                restore_best_weights=True, verbose=0
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=7, verbose=0
            ),
        ]

        self.model_.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        self.history_ = series.copy()
        return self

    # ── predict (recursive) ─────────────────────────────────────────────────

    def predict(self, horizon: int = 8) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")

        norm_hist = list(self._normalise(self.history_))
        preds_norm = []

        for _ in range(horizon):
            window = np.array(norm_hist[-self.lookback:])[np.newaxis, :, np.newaxis]
            yhat_norm = float(self.model_.predict(window, verbose=0)[0, 0])
            preds_norm.append(yhat_norm)
            norm_hist.append(yhat_norm)

        preds = self._denormalise(np.array(preds_norm))
        return np.clip(preds, 0, None)

    # ── val predict ─────────────────────────────────────────────────────────

    def predict_val(self, val: pd.DataFrame) -> np.ndarray:
        """
        Recursive prediction over the validation window
        using training history as seed.
        """
        return self.predict(horizon=len(val))
