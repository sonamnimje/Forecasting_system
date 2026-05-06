"""
run.py
------
Convenience entry point for the forecasting system.

Usage
-----
# 1. Train all models and produce forecasts
python run.py --train

# 2. Start REST API
python run.py --serve

# 3. Train + Serve
python run.py --all
"""

import argparse
import subprocess
import sys
from pathlib import Path


DATA_PATH = Path("data/sales_data.csv")


# ---------------------------------------------------
# Validate dataset
# ---------------------------------------------------
def validate_dataset():
    """Ensure dataset exists before training."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"""
Dataset not found: {DATA_PATH}

Please place your assignment dataset here:
data/sales_data.csv

Required columns:
- date
- state
- sales
"""
        )


# ---------------------------------------------------
# Train models
# ---------------------------------------------------
def train():
    print("=" * 60)
    print(" Step 1: Training models & selecting best per state")
    print("=" * 60)

    validate_dataset()

    from models.model_selector import run_pipeline
    run_pipeline(data_path=str(DATA_PATH))


# ---------------------------------------------------
# Start API
# ---------------------------------------------------
def serve():
    print("=" * 60)
    print(" Step 2: Starting FastAPI server")
    print(" API URL: http://localhost:8000")
    print(" Docs: http://localhost:8000/docs")
    print("=" * 60)

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.main:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "8000"
            ],
            cwd=str(Path(__file__).parent),
            check=True
        )

    except FileNotFoundError:
        print("\nERROR: uvicorn not installed")
        print("Run: pip install uvicorn")


# ---------------------------------------------------
# Train + Serve
# ---------------------------------------------------
def run_all():
    train()
    serve()


# ---------------------------------------------------
# Main CLI
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sales Forecasting System"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train forecasting models"
    )

    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start FastAPI server"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Train models + start API"
    )

    args = parser.parse_args()

    if args.all:
        run_all()

    elif args.train:
        train()

    elif args.serve:
        serve()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()