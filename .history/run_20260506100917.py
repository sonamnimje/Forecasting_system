"""
run.py
------
Convenience entry point.

Usage
-----
# 1. Generate dummy data (or skip if you have real data at data/sales_data.csv)
python run.py --generate-data

# 2. Train all models and produce forecasts
python run.py --train

# 3. Start the REST API server
python run.py --serve

# 4. All steps in sequence
python run.py --all
"""

import argparse
import subprocess
import sys
from pathlib import Path


def generate_data():
    print("=" * 60)
    print("  Step 1: Generating synthetic sales data")
    print("=" * 60)
    from data.generate_data import build_dataset
    build_dataset()


def train():
    print("=" * 60)
    print("  Step 2: Training models & selecting best per state")
    print("=" * 60)
    from models.model_selector import run_pipeline
    run_pipeline()


def serve():
    print("=" * 60)
    print("  Step 3: Starting FastAPI server on http://localhost:8000")
    print("  Interactive docs → http://localhost:8000/docs")
    print("=" * 60)
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--reload", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(Path(__file__).parent),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sales Forecasting System – end-to-end pipeline"
    )
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate synthetic sales data")
    parser.add_argument("--train",         action="store_true",
                        help="Run model training pipeline")
    parser.add_argument("--serve",         action="store_true",
                        help="Start the FastAPI REST API")
    parser.add_argument("--all",           action="store_true",
                        help="Run all steps: generate → train → serve")
    args = parser.parse_args()

    if args.all:
        generate_data()
        train()
        serve()
    else:
        if args.generate_data:
            generate_data()
        if args.train:
            train()
        if args.serve:
            serve()

    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()
