"""
models/prophet_model.py
-----------------------
Facebook Prophet forecaster with US holiday regressors and
automatic changepoint / seasonality tuning.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    _PROPHET_OK = True
except ImportError:
    try:
        from fbprophet import Prophet
        _PROPHET_OK = True
    except ImportError:
        _PROPHET_OK = False


class ProphetForecaster:
    """
    Wraps Facebook Prophet for weekly sales forecasting.
    Adds yearly + monthly seasonality and a US-holidays component.
    """

    NAME = "Prophet"

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        add_monthly: bool = True,
        changepoint_prior_scale: float = 0.1,
        seasonality_prior_scale: float = 10.0,
    ):
        if not _PROPHET_OK:
            raise ImportError(
                "prophet is not installed. Run: pip install prophet"
            )
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.add_monthly = add_monthly
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.model_ = None
        self.last_train_date_ = None

    # ── fit ────────────────────────────────────────────────────────────────

    def fit(self, train: pd.DataFrame) -> "ProphetForecaster":
        df_p = train[["date", "sales"]].rename(
            columns={"date": "ds", "sales": "y"}
        )

        self.model_ = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
        )

        if self.add_monthly:
            self.model_.add_seasonality(
                name="monthly", period=30.5, fourier_order=5
            )

        # Built-in US holidays
        self.model_.add_country_holidays(country_name="US")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(df_p)

        self.last_train_date_ = train["date"].max()
        return self

    # ── predict ────────────────────────────────────────────────────────────

    def predict(self, horizon: int = 8) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")

        future = self.model_.make_future_dataframe(
            periods=horizon, freq="W-MON", include_history=False
        )
        forecast = self.model_.predict(future)
        preds = forecast["yhat"].values
        return np.clip(preds, 0, None)

    # ── val predict ─────────────────────────────────────────────────────────

    def predict_val(self, val: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")

        future = val[["date"]].rename(columns={"date": "ds"})
        forecast = self.model_.predict(future)
        preds = forecast["yhat"].values
        return np.clip(preds, 0, None)
