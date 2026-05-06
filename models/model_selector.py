"""
models/model_selector.py
-------------------------
Trains all four models per state, evaluates on a held-out validation set,
selects the best model by SMAPE, and generates 8-week forecasts.

Saves:
  - results/model_scores.csv      – per-state metric comparison
  - results/best_models.csv       – winning model per state
  - results/forecasts.csv         – 8-week forecast per state
"""

import json
import warnings
import traceback
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from utils.preprocessing import (
    load_and_clean, add_features, time_split, compute_metrics
)
from models.arima_model   import ARIMAForecaster
from models.prophet_model import ProphetForecaster
from models.xgboost_model import XGBoostForecaster
from models.lstm_model    import LSTMForecaster

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _forecast_dates(last_date: pd.Timestamp, horizon: int = 8) -> list:
    return [
        (last_date + pd.DateOffset(weeks=i + 1)).strftime("%Y-%m-%d")
        for i in range(horizon)
    ]


def _safe_train_predict(model, train, val):
    """Fit model, return val predictions and errors (if any)."""
    try:
        model.fit(train)
        preds = model.predict_val(val)
        return preds, None
    except Exception as e:
        return None, traceback.format_exc()


# ──────────────────────────────────────────────────────────────────────────────
# Per-state pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_state(
    df_feat: pd.DataFrame,
    state: str,
    horizon: int = 8,
    val_weeks: int = 16,
) -> Dict[str, Any]:
    """
    Full pipeline for a single state.
    Returns a dict with scores, best_model, and forecast.
    """
    print(f"\n{'='*60}")
    print(f"  Processing: {state}")
    print(f"{'='*60}")

    train, val = time_split(df_feat, state, val_weeks=val_weeks)

    models = {
        "ARIMA":   ARIMAForecaster(seasonal=True, m=52),
        "Prophet": ProphetForecaster(),
        "XGBoost": XGBoostForecaster(),
        "LSTM":    LSTMForecaster(lookback=min(52, len(train) - 1)),
    }

    scores   = {}
    val_preds_map = {}

    for name, model in models.items():
        print(f"  ► Training {name}...", end=" ", flush=True)
        preds, err = _safe_train_predict(model, train, val)

        if err:
            print(f"FAILED\n    {err.splitlines()[-1]}")
            scores[name] = {"RMSE": np.nan, "MAE": np.nan,
                            "MAPE": np.nan, "SMAPE": np.nan}
        else:
            actual  = val["sales"].values
            metrics = compute_metrics(actual, preds)
            scores[name] = metrics
            val_preds_map[name] = preds
            print(f"done  |  SMAPE={metrics['SMAPE']:.2f}%  RMSE={metrics['RMSE']:,.0f}")

    # Pick best model by SMAPE (lower = better)
    valid_models = {k: v for k, v in scores.items() if not np.isnan(v["SMAPE"])}
    if not valid_models:
        raise RuntimeError(f"All models failed for state: {state}")

    best_name = min(valid_models, key=lambda k: valid_models[k]["SMAPE"])
    print(f"  ★ Best model: {best_name}  (SMAPE={valid_models[best_name]['SMAPE']:.2f}%)")

    # Re-fit best model on FULL data to generate future forecast
    full_data = df_feat[df_feat["state"] == state].sort_values("date")
    best_model = models[best_name]
    best_model.fit(full_data)
    future_preds = best_model.predict(horizon=horizon)

    last_date   = full_data["date"].max()
    future_dates = _forecast_dates(last_date, horizon)

    return {
        "state":      state,
        "scores":     scores,
        "best_model": best_name,
        "forecast": {
            "dates":      future_dates,
            "sales":      [round(float(p), 2) for p in future_preds],
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline (all states)
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    data_path: str = "data/sales_data.csv",
    horizon: int = 8,
    val_weeks: int = 16,
    states: list = None,
) -> dict:
    """
    End-to-end pipeline.

    Parameters
    ----------
    data_path : path to CSV with columns: date, state, sales
    horizon   : forecast horizon in weeks
    val_weeks : validation window size
    states    : subset of states to process (None = all)

    Returns
    -------
    dict keyed by state name, each value is the run_state() output dict
    """

    print("Loading and cleaning data...")
    df = load_and_clean(data_path)
    print(f"  {len(df):,} rows  |  {df['state'].nunique()} states")

    print("Engineering features...")
    df_feat = add_features(df)

    # ---------------- FIX START ----------------
    # Sometimes pandas groupby/apply moves state into index
    # Restore it if missing
    if "state" not in df_feat.columns:
        print("State column missing after feature engineering. Restoring it...")
        
        # reset index first
        df_feat = df_feat.reset_index()

        # if still missing, pull from original dataframe
        if "state" not in df_feat.columns:
            df_feat["state"] = df["state"].values

    print("Final feature columns:")
    print(df_feat.columns.tolist())
    print(df_feat.head())

    # Get all unique states safely
    all_states = df_feat["state"].unique().tolist()
    # ---------------- FIX END ----------------

    if states:
        all_states = [s for s in all_states if s in states]

    results = {}

    for state in all_states:
        results[state] = run_state(
            df_feat=df_feat,
            state=state,
            horizon=horizon,
            val_weeks=val_weeks
        )

    _save_results(results, horizon)
    return results


def _save_results(results: dict, horizon: int):
    """Persist scores, best models, and forecasts to CSV / JSON."""
    score_rows, best_rows, forecast_rows = [], [], []

    for state, res in results.items():
        for model_name, metrics in res["scores"].items():
            score_rows.append({
                "state": state,
                "model": model_name,
                **metrics,
                "is_best": model_name == res["best_model"],
            })

        best_rows.append({
            "state":      state,
            "best_model": res["best_model"],
            **res["scores"][res["best_model"]],
        })

        for i, (dt, sale) in enumerate(
            zip(res["forecast"]["dates"], res["forecast"]["sales"])
        ):
            forecast_rows.append({
                "state":     state,
                "week":      i + 1,
                "date":      dt,
                "sales_forecast": sale,
            })

    pd.DataFrame(score_rows).to_csv(RESULTS_DIR / "model_scores.csv", index=False)
    pd.DataFrame(best_rows).to_csv(RESULTS_DIR / "best_models.csv",   index=False)
    pd.DataFrame(forecast_rows).to_csv(RESULTS_DIR / "forecasts.csv", index=False)

    # Also save full JSON
    (RESULTS_DIR / "full_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )

    print(f"\n✓ Results saved to {RESULTS_DIR}/")
    print(f"  model_scores.csv  |  best_models.csv  |  forecasts.csv  |  full_results.json")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
