"""
models/arima_model.py
---------------------
ARIMA / SARIMA forecaster with auto-order selection via AIC grid search.
Falls back to (1,1,1) if pmdarima is unavailable.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import pmdarima as pm
    _AUTO_ARIMA = True
except ImportError:
    from statsmodels.tsa.arima.model import ARIMA
    _AUTO_ARIMA = False


class ARIMAForecaster:
    """
    Wraps pmdarima.auto_arima (preferred) or statsmodels ARIMA.
    Automatically handles seasonal and non-seasonal data.
    """

    NAME = "ARIMA/SARIMA"

    def __init__(self, seasonal: bool = True, m: int = 52):
        """
        Parameters
        ----------
        seasonal : bool  – fit SARIMA if True
        m        : int   – seasonal period (52 = weekly annual)
        """
        self.seasonal = seasonal
        self.m = m
        self.model_ = None
        self.fitted_ = None

    # ── fit ────────────────────────────────────────────────────────────────

    def fit(self, train: pd.DataFrame) -> "ARIMAForecaster":
        y = train.set_index("date")["sales"].asfreq("W-MON")

        if _AUTO_ARIMA:
            self.model_ = pm.auto_arima(
                y,
                seasonal=self.seasonal,
                m=self.m,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                information_criterion="aic",
                max_p=3, max_q=3, max_P=2, max_Q=2, max_d=2,
                n_jobs=1,
            )
        else:
            # Fallback: simple ARIMA(1,1,1)
            self.model_ = ARIMA(y, order=(1, 1, 1)).fit()

        return self

    # ── predict ────────────────────────────────────────────────────────────

    def predict(self, horizon: int = 8) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")

        if _AUTO_ARIMA:
            preds = self.model_.predict(n_periods=horizon)
        else:
            preds = self.model_.forecast(steps=horizon)

        return np.clip(np.array(preds), 0, None)

    # ── val predict (in-sample) ─────────────────────────────────────────────

    def predict_val(self, val: pd.DataFrame) -> np.ndarray:
        horizon = len(val)
        return self.predict(horizon=horizon)

    # ── summary ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        if self.model_ is None:
            return "Model not fitted."
        if _AUTO_ARIMA:
            return str(self.model_.summary())
        return str(self.model_.summary())
