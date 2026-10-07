"""
Run TabPFN on the same 85/15 single split used by py_neural_eval.py,
then merge results into neural_metrics.json.
"""
import numpy as np
import pandas as pd
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(__file__))
from py_metrics import compute_all_metrics, print_metrics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    print("=" * 70)
    print("  TabPFN SINGLE-SPLIT EVALUATION")
    print("=" * 70)

    # ── 1. Load Data (same as py_neural_eval.py) ─────────────────────────
    data_dir = "data"
    training_data = pd.read_csv(os.path.join(data_dir, "training_eligible.csv"))

    with open(os.path.join(data_dir, "predictor_columns.txt"), "r") as f:
        predictors = [line.strip() for line in f if line.strip()]

    X = training_data[predictors].values
    y = training_data['InvRedshift'].values

    # Same split parameters as py_neural_eval.py
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    print(f"  Train: {X_train_sc.shape[0]} samples, Val: {X_val_sc.shape[0]} samples")
    print(f"  Features: {X_train_sc.shape[1]}")

    # ── 2. Run TabPFN ────────────────────────────────────────────────────
    print("\nTraining TabPFN...")
    t0 = time.time()

    try:
        os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'
        from tabpfn import TabPFNRegressor

        model = TabPFNRegressor(
            random_state=42,
            ignore_pretraining_limits=True
        )
        model.fit(X_train_sc, y_train)
        preds_inv = model.predict(X_val_sc)

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        metrics = compute_all_metrics(preds_inv, y_val)
        print_metrics(metrics, "TabPFN")

        # ── 3. Merge into neural_metrics.json ────────────────────────────
        json_path = os.path.join("output", "python_results", "neural_metrics.json")

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        all_metrics['tabpfn'] = {
            'R_z': float(metrics['z']['R']),
            'RMSE_z': float(metrics['z']['RMSE']),
            'outlier_pct_fixed': float(metrics['outlier_pct']),
            'outlier_pct_2sigma': float(metrics['outlier_pct_2sigma'])
        }

        with open(json_path, "w") as f:
            json.dump(all_metrics, f, indent=4)

        print(f"\n  [OK] TabPFN results saved to {json_path}")
        print(f"  [OK] Runtime: {elapsed:.1f}s")

    except ImportError:
        print("  [ERROR] tabpfn package is not installed.")
        print("  Install with: pip install tabpfn")
        sys.exit(1)
    except Exception as e:
        print(f"  [ERROR] TabPFN failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
