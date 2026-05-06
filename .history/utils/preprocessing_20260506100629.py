"""
utils/preprocessing.py
-----------------------
Shared preprocessing, feature engineering, and metric utilities.
"""

import warnings
import pandas as pd
import numpy as np
from typing import Tuple

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ──────────────────────────────────────────────

def load_and_clean(path: str = "data/sales_data.csv") -> pd.DataFrame:
    """
    Load raw CSV, parse dates, enforce schema, handle missing values.
    Expected columns: date, state, sales
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower()

    required = {"date", "state", "sales"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required}. Found: {set(df.columns)}")

    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    # Fill missing dates per state with full weekly calendar
    df = _fill_missing_dates(df)

    # Interpolate missing sales values
    df["sales"] = (
        df.groupby("state")["sales"]
        .transform(lambda s: s.interpolate(method="linear").ffill().bfill())
    )

    df["sales"] = df["sales"].clip(lower=0)
    return df


def _fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure each state has a complete weekly date range."""
    frames = []
    global_min = df["date"].min()
    global_max = df["date"].max()
    full_range = pd.date_range(global_min, global_max, freq="W-MON")

    for state, grp in df.groupby("state"):
        grp = grp.set_index("date").reindex(full_range)
        grp.index.name = "date"
        grp["state"] = state
        grp = grp.reset_index()
        frames.append(grp)

    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────

US_HOLIDAYS = {
    # A simplified set of approximate US holiday weeks (month, day)
    (1, 1), (7, 4), (11, 25), (11, 26), (12, 25), (12, 26),
    (5, 27), (9, 2), (1, 20), (2, 17),
}


def _is_holiday(dt: pd.Timestamp) -> int:
    return int((dt.month, dt.day) in US_HOLIDAYS)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-series feature engineering columns.
    Requires columns: date, sales (sorted per state).
    """
    df = df.copy()
    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    def _featurize(grp):
        g = grp.copy().sort_values("date")
        s = g["sales"]

        # Calendar features
        g["day_of_week"]  = g["date"].dt.dayofweek
        g["week_of_year"] = g["date"].dt.isocalendar().week.astype(int)
        g["month"]        = g["date"].dt.month
        g["quarter"]      = g["date"].dt.quarter
        g["year"]         = g["date"].dt.year
        g["is_holiday"]   = g["date"].apply(_is_holiday)

        # Lag features
        for lag in [1, 2, 4, 8, 13, 26, 52]:
            g[f"lag_{lag}"] = s.shift(lag)

        # Rolling statistics (trailing, no leakage)
        for window in [4, 8, 13, 26]:
            g[f"roll_mean_{window}"] = s.shift(1).rolling(window).mean()
            g[f"roll_std_{window}"]  = s.shift(1).rolling(window).std()
            g[f"roll_max_{window}"]  = s.shift(1).rolling(window).max()
            g[f"roll_min_{window}"]  = s.shift(1).rolling(window).min()

        # Expanding mean
        g["expanding_mean"] = s.shift(1).expanding().mean()

        return g

    return df.groupby("state", group_keys=False).apply(_featurize)


# ──────────────────────────────────────────────
# 3. TRAIN / VALIDATION SPLIT  (no leakage)
# ──────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    state: str,
    val_weeks: int = 16
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/val split for a single state.
    val_weeks controls how many weeks are held out for validation.
    """
    grp = df[df["state"] == state].sort_values("date").copy()
    cutoff = grp["date"].iloc[-val_weeks]
    train = grp[grp["date"] < cutoff].copy()
    val   = grp[grp["date"] >= cutoff].copy()
    return train, val


# ──────────────────────────────────────────────
# 4. METRICS
# ──────────────────────────────────────────────

def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%)"""
    actual, predicted = np.array(actual), np.array(predicted)
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denom > 0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.array(actual) - np.array(predicted))))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def compute_metrics(actual, predicted) -> dict:
    return {
        "RMSE":  round(rmse(actual, predicted), 2),
        "MAE":   round(mae(actual, predicted), 2),
        "MAPE":  round(mape(actual, predicted), 2),
        "SMAPE": round(smape(actual, predicted), 2),
    }
