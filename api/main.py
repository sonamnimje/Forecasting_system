"""
api/main.py
-----------
FastAPI REST API exposing the forecasting system.

Start the server:
    uvicorn api.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_PATH    = BASE_DIR / "data" / "sales_data.csv"
RESULTS_DIR  = BASE_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "full_results.json"

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sales Forecasting API",
    description=(
        "Production-ready time-series forecasting system.\n\n"
        "Trains ARIMA/SARIMA, Prophet, XGBoost, and LSTM per state. "
        "Auto-selects best model by SMAPE and serves 8-week forecasts."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ForecastPoint(BaseModel):
    week: int
    date: str
    sales_forecast: float


class StateForecast(BaseModel):
    state: str
    best_model: str
    horizon_weeks: int
    forecast: List[ForecastPoint]
    model_scores: dict


class ModelComparisonRow(BaseModel):
    state: str
    model: str
    RMSE: Optional[float]
    MAE: Optional[float]
    MAPE: Optional[float]
    SMAPE: Optional[float]
    is_best: bool


class TrainRequest(BaseModel):
    data_path: Optional[str] = Field(
        default=None,
        description="Path to CSV file. Leave blank to use default data.",
    )
    horizon: int = Field(default=8, ge=1, le=52, description="Forecast horizon (weeks)")
    val_weeks: int = Field(default=16, ge=4, le=52, description="Validation window size")
    states: Optional[List[str]] = Field(
        default=None, description="Subset of states. None = all."
    )


class TrainResponse(BaseModel):
    status: str
    states_trained: List[str]
    best_models: dict


# ── helpers ───────────────────────────────────────────────────────────────────

_results_cache: dict = {}


def _load_results() -> dict:
    global _results_cache
    if _results_cache:
        return _results_cache
    if RESULTS_JSON.exists():
        _results_cache = json.loads(RESULTS_JSON.read_text())
    return _results_cache


def _run_pipeline_task(req: TrainRequest):
    """Background task that runs the full ML pipeline."""
    from models.model_selector import run_pipeline
    global _results_cache

    path   = req.data_path or str(DATA_PATH)
    result = run_pipeline(
        data_path=path,
        horizon=req.horizon,
        val_weeks=req.val_weeks,
        states=req.states,
    )
    _results_cache = result


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Sales Forecasting API",
        "version": "1.0.0",
        "docs":    "/docs",
        "status":  "running",
    }


@app.get("/health", tags=["Health"])
def health():
    results_ready = RESULTS_JSON.exists() or bool(_results_cache)
    return {
        "status":        "ok",
        "results_ready": results_ready,
        "data_file":     str(DATA_PATH),
        "data_exists":   DATA_PATH.exists(),
    }


@app.post("/train", response_model=TrainResponse, tags=["Training"])
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger model training (async background job).
    Poll /health or /states to know when results are ready.
    """
    background_tasks.add_task(_run_pipeline_task, req)
    return TrainResponse(
        status="training_started",
        states_trained=[],
        best_models={},
    )


@app.post("/train/sync", response_model=TrainResponse, tags=["Training"])
def train_sync(req: TrainRequest):
    """
    Synchronous training (blocks until complete).
    Use for small datasets or testing.
    """
    global _results_cache
    from models.model_selector import run_pipeline

    path   = req.data_path or str(DATA_PATH)
    result = run_pipeline(
        data_path=path,
        horizon=req.horizon,
        val_weeks=req.val_weeks,
        states=req.states,
    )
    _results_cache = result

    best_models = {
        state: res["best_model"] for state, res in result.items()
    }
    return TrainResponse(
        status="training_complete",
        states_trained=list(result.keys()),
        best_models=best_models,
    )


@app.get("/states", tags=["Forecast"])
def list_states():
    """List all states with trained forecasts."""
    results = _load_results()
    if not results:
        raise HTTPException(
            status_code=404,
            detail="No results found. Run POST /train first.",
        )
    return {"states": list(results.keys()), "count": len(results)}


@app.get("/forecast/{state}", response_model=StateForecast, tags=["Forecast"])
def get_forecast(state: str):
    """
    Retrieve the 8-week sales forecast for a given state.

    Example:
        GET /forecast/California
    """
    results = _load_results()
    if not results:
        raise HTTPException(
            status_code=404,
            detail="No results found. Run POST /train first.",
        )

    # Case-insensitive match
    match = next(
        (k for k in results if k.lower() == state.lower()), None
    )
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"State '{state}' not found. Available: {list(results.keys())}",
        )

    res = results[match]
    fcast = res["forecast"]

    points = [
        ForecastPoint(
            week=i + 1,
            date=fcast["dates"][i],
            sales_forecast=fcast["sales"][i],
        )
        for i in range(len(fcast["dates"]))
    ]

    return StateForecast(
        state=match,
        best_model=res["best_model"],
        horizon_weeks=len(points),
        forecast=points,
        model_scores=res["scores"],
    )


@app.get("/forecast", tags=["Forecast"])
def get_all_forecasts(
    states: Optional[str] = Query(
        default=None,
        description="Comma-separated list of states. Empty = all."
    )
):
    """
    Retrieve forecasts for multiple states at once.

    Example:
        GET /forecast?states=California,Texas,Florida
    """
    results = _load_results()
    if not results:
        raise HTTPException(
            status_code=404,
            detail="No results found. Run POST /train first.",
        )

    filter_states = (
        [s.strip() for s in states.split(",")] if states else list(results.keys())
    )

    output = {}
    for state in filter_states:
        match = next(
            (k for k in results if k.lower() == state.lower()), None
        )
        if match:
            output[match] = results[match]["forecast"]

    return {"count": len(output), "forecasts": output}


@app.get("/compare", response_model=List[ModelComparisonRow], tags=["Analysis"])
def compare_models(
    state: Optional[str] = Query(default=None, description="Filter by state")
):
    """
    Compare all models' validation metrics across states.
    Useful for understanding which model performs best where.
    """
    score_file = RESULTS_DIR / "model_scores.csv"
    if not score_file.exists():
        raise HTTPException(
            status_code=404,
            detail="No scores found. Run POST /train first.",
        )

    df = pd.read_csv(score_file)
    if state:
        df = df[df["state"].str.lower() == state.lower()]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"State '{state}' not found.")

    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")


@app.get("/best-models", tags=["Analysis"])
def best_models_summary():
    """Summary of the winning model selected per state."""
    best_file = RESULTS_DIR / "best_models.csv"
    if not best_file.exists():
        raise HTTPException(
            status_code=404,
            detail="No results found. Run POST /train first.",
        )

    df = pd.read_csv(best_file)
    model_counts = df["best_model"].value_counts().to_dict()
    return {
        "per_state":    df.set_index("state")["best_model"].to_dict(),
        "model_counts": model_counts,
        "total_states": len(df),
    }
