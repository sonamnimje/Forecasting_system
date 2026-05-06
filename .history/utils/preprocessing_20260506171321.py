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

def load_and_clean(path: str = "data/Sales_data.csv") -> pd.DataFrame:
    """
    Load raw CSV, parse dates, enforce schema, handle missing values.

    Expected dataset columns:
    - state
    - date
    - total (sales column in your dataset)
    - category (optional)
    """

    df = pd.read_csv(path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    print("Original columns:", df.columns.tolist())

    # Rename total → sales
    if "total" in df.columns:
        df.rename(columns={"total": "sales"}, inplace=True)

    required = {"date", "state", "sales"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns: {required}. Found: {set(df.columns)}"
        )

    # FIX DATE FORMAT (DD-MM-YYYY)
    df["date"] = pd.to_datetime(
        df["date"],
        dayfirst=True,
        errors="coerce"
    )

    # Remove invalid dates if any
    df = df.dropna(subset=["date"])

    # Drop category column (not needed for current forecasting logic)
    if "category" in df.columns:
        print("Dropping category column (not used in forecasting)")
        df.drop(columns=["category"], inplace=True)

    # Sort data
    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    # Fill missing dates
    df = _fill_missing_dates(df)

    # Convert sales column to numeric
df["sales"] = (
    df["sales"]
    .astype(str)
    .str.replace(",", "", regex=False)   # remove commas
    .str.replace("$", "", regex=False)   # remove dollar sign if present
    .str.strip()
)

df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

print("Sales dtype after conversion:", df["sales"].dtype)

# Handle missing values after numeric conversion
df["sales"] = (
    df.groupby("state")["sales"]
    .transform(
        lambda s: s.interpolate(method="linear").ffill().bfill()
    )
)

# Prevent negative sales
df["sales"] = df["sales"].clip(lower=0)

    print(f"Cleaned dataset shape: {df.shape}")
    print(df.head())

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
