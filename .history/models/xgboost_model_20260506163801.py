"""
models/xgboost_model.py
-----------------------
XGBoost forecaster using lag + rolling + calendar features.
Recursive multi-step forecasting for the prediction horizon.
"""

import warnings
import numpy as np
import pandas as pd
from typing import List

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    _XGB_OK = True
except ImportError:
    _XGB_OK = False

# Feature columns produced by utils.preprocessing.add_features
LAG_COLS    = [f"lag_{k}"        for k in [1, 2, 4, 8, 13, 26, 52]]
ROLL_COLS   = [f"roll_mean_{w}"  for w in [4, 8, 13, 26]] + \
              [f"roll_std_{w}"   for w in [4, 8, 13, 26]] + \
              [f"roll_max_{w}"   for w in [4, 8, 13, 26]] + \
              [f"roll_min_{w}"   for w in [4, 8, 13, 26]]
CAL_COLS    = ["day_of_week", "week_of_year", "month", "quarter",
               "year", "is_holiday"]
EXPAND_COLS = ["expanding_mean"]

ALL_FEATURE_COLS: List[str] = LAG_COLS + ROLL_COLS + CAL_COLS + EXPAND_COLS


class XGBoostForecaster:
    """
    XGBoost with rich feature engineering.
    Recursive strategy: each future step is predicted one-at-a-time,
    updating the lag/rolling windows with the latest prediction.
    """

    NAME = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        if not _XGB_OK:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")

        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective="reg:squarederror",
            tree_method="hist",
            verbosity=0,
        )
        self.model_    = None
        self.history_  = None   # stores training sales series for recursive pred

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _get_feature_cols(df: pd.DataFrame) -> List[str]:
        return [c for c in ALL_FEATURE_COLS if c in df.columns]

    @staticmethod
    def _build_row_features(
        date: pd.Timestamp,
        history: np.ndarray,
    ) -> dict:
        """Build a single feature row from a rolling history array."""
        lag_map = {1: 1, 2: 2, 4: 4, 8: 8, 13: 13, 26: 26, 52: 52}
        row = {}
        for lag, k in lag_map.items():
            row[f"lag_{k}"] = history[-lag] if len(history) >= lag else np.nan

        for w in [4, 8, 13, 26]:
            window = history[-w:] if len(history) >= w else history
            row[f"roll_mean_{w}"] = np.nanmean(window)
            row[f"roll_std_{w}"]  = np.nanstd(window)
            row[f"roll_max_{w}"]  = np.nanmax(window)
            row[f"roll_min_{w}"]  = np.nanmin(window)

        row["expanding_mean"] = np.nanmean(history)
        row["day_of_week"]    = date.dayofweek
        row["week_of_year"]   = date.isocalendar()[1]
        row["month"]          = date.month
        row["quarter"]        = date.quarter
        row["year"]           = date.year
        row["is_holiday"]     = int((date.month, date.day) in {
            (1,1),(7,4),(11,25),(11,26),(12,25),(12,26),
            (5,27),(9,2),(1,20),(2,17),
        })
        return row

    # ── fit ────────────────────────────────────────────────────────────────

    def fit(self, train: pd.DataFrame) -> "XGBoostForecaster":
        feat_cols = self._get_feature_cols(train)
        target_col = "sales"

        df_clean = train.dropna(subset=feat_cols + [target_col])
        X = df_clean[feat_cols].values
        y = df_clean[target_col].values

        self.model_ = xgb.XGBRegressor(**self.params)
        self.model_.fit(X, y)

        # Keep the full training sales series for recursive forecasting
        self.history_     = train.sort_values("date")["sales"].values.copy()
        self.feat_cols_   = feat_cols
        self.last_date_   = train["date"].max()
        return self

    # ── predict (recursive) ─────────────────────────────────────────────────

    def predict(self, horizon: int = 8) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")

        history  = list(self.history_)
        preds    = []
        cur_date = self.last_date_

        for _ in range(horizon):
            cur_date = cur_date + pd.DateOffset(weeks=1)
            row      = self._build_row_features(cur_date, np.array(history))
            X_row    = np.array([[row.get(c, np.nan) for c in self.feat_cols_]])
            yhat     = float(self.model_.predict(X_row)[0])
            yhat     = max(yhat, 0.0)
            preds.append(yhat)
            history.append(yhat)

        return np.array(preds)

    # ── val predict ─────────────────────────────────────────────────────────

    def predict_val(self, val: pd.DataFrame) -> np.ndarray:
        """
        For validation we use val features directly (teacher-forcing style)
        to get a fair comparison with other models on actual lag history.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")

        feat_cols = [c for c in self.feat_cols_ if c in val.columns]
        df_clean  = val.copy()
        # Fill any missing features with median
        for c in feat_cols:
            df_clean[c] = df_clean[c].fillna(df_clean[c].median())

        X    = df_clean[feat_cols].values
        preds = np.clip(self.model_.predict(X).astype(float), 0, None)
        return preds

    # ── feature importance ──────────────────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        if self.model_ is None:
            return pd.DataFrame()
        imp = self.model_.feature_importances_
        return (
            pd.DataFrame({"feature": self.feat_cols_, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
