"""
AutoML — Array Script
==========================================================================
(Target = Binary Classification)

Run on the cluster with the accompanying .sh (array job)
"""

import os
import sys
import time
import pickle
import joblib
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="AutoML Array Job Script")
    parser.add_argument("--run_id", type=str, default="1", help="SGE Task ID / Run ID")
    parser.add_argument("--base_dir", type=str, required=True, help="Base working directory")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--run_name", type=str, default=int(time.time()), help="Name of run")
    parser.add_argument("--train_csv_path", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--test_csv_paths", type=str, nargs="+", required=False, help="Path(s) to external test CSV(s)")
    parser.add_argument("--target_col", type=str, default="MMSE", help="Target column name")
    parser.add_argument("--exploitation", type=float, default=0.75, help="Exploitation parameter")
    parser.add_argument("--time_exp", type=int, default=600, help="Seconds of BO per run")
    parser.add_argument("--test_size", type=float, default=0.25, help="Test set split fraction")
    parser.add_argument("--val_size", type=float, default=0.20, help="Validation set split fraction")
    parser.add_argument('--save_plots', type=lambda x: (str(x).lower() == 'true'))
    return parser.parse_args()


def save_run_results(run_id, ensemble, model_accuracies, final_acc, base_output_dir):
    """Dynamic saving: each run writes its own CSV (models + accuracies)."""
    runs_dir = os.path.join(base_output_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    row = {
        "run_id": run_id,
        "n_models": len(ensemble),
        "stacked_bal_acc": final_acc,
    }
    for i, model in enumerate(ensemble, 1):
        try:
            clf_name = type(model.steps[-1][1]).__name__
        except Exception:
            clf_name = type(model).__name__
        row[f"model_{i}"] = clf_name
        row[f"model_{i}_acc"] = model_accuracies.get(f"model_{i}_acc", np.nan)
    pd.DataFrame([row]).to_csv(
        os.path.join(runs_dir, f"run_{run_id}.csv"), index=False
    )


def save_bo_history(run_id, bo_history, base_output_dir, run_name):

    bo_dir = os.path.join(base_output_dir, "bo_history")
    os.makedirs(bo_dir, exist_ok=True)

    df = pd.DataFrame(bo_history)
    df.insert(0, "run_id", run_id)

    df.to_csv(
        os.path.join(bo_dir, f"bo_run_{run_id}_{run_name}.csv"),
        index=False
    )


def load_and_prepare_data(csv_path, target_col):
    """Load dataset, extract binary target variable, return feature matrix X and target y."""
    df = pd.read_csv(csv_path)
    y = df[target_col].values
    X_raw = df.drop(columns=[target_col])
    return X_raw, y


def main():
    args = parse_args()

    # Dynamic Directory Setup
    os.makedirs(args.results_dir, exist_ok=True)
    BASE_DIR    = os.path.expanduser(args.base_dir)
    MZ_DIR      = os.path.join(BASE_DIR, "Codebase")
    DATA_DIR    = os.path.join(MZ_DIR, "data/space")
    CODE_DIR    = os.path.join(MZ_DIR, "code")
    RESULTS_DIR = os.path.expanduser(args.results_dir)

    sys.path.append(DATA_DIR)
    sys.path.append(CODE_DIR)

    from sample_space_script_16_8 import ModelZoom
    from ensemble_models_script_17_8_new_hamming import do_ensemble, train_ensemble

    # ── Load data (raw) ──────────────────────────────────────────────
    X_raw, y = load_and_prepare_data(args.train_csv_path, args.target_col)
    print(f"[Run {args.run_id}] Data shape: {X_raw.shape}, Target balance (positive class ratio): {y.mean():.3f}")

    # ── Load latent space + pipelines ────────────────────────────────
    kernel = Matern(length_scale=50, length_scale_bounds=(1e-2, 1e3), nu=0.5) + WhiteKernel(noise_level=0.5)
    bad_models = np.load(os.path.join(DATA_DIR, "bad_models_23_7.npy"), allow_pickle=True)
    
    with open(os.path.join(MZ_DIR, "pipelines.pkl"), "rb") as f:
        pipelines = pickle.load(f)

    bad_models.sort()
    for i in range(bad_models.shape[0]):
        del pipelines[bad_models[-i - 1]]

    test_drive = ModelZoom(kernel=kernel, uf="ucb", exploitation=args.exploitation)
    space = test_drive.load_space(os.path.join(DATA_DIR, "reduced_space_23_7.npy"))

    # ── Split FIRST (raw), then transform — stratified for binary target
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, stratify=y, test_size=args.test_size
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, stratify=y_train, test_size=args.val_size
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    # Fit transformations ONLY on train set to prevent leakage
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_train     = scaler.fit_transform(X_train_imp)

    X_val_imp   = imputer.transform(X_val_raw)
    X_val       = scaler.transform(X_val_imp)

    X_test_imp  = imputer.transform(X_test_raw)
    X_test      = scaler.transform(X_test_imp)

    # ── Output folder ────────────────────────────────────────────────
    base_output_dir = RESULTS_DIR
    os.makedirs(base_output_dir, exist_ok=True)
    time_output_dir = os.path.join(base_output_dir, f"run_{args.run_id}_{args.run_name}")
    os.makedirs(time_output_dir, exist_ok=True)

    # ── BO sampling ──────────────────────────────────────────────────
    sample_data_ls = test_drive.sampling_algo(
        space=space,
        time_exp=args.time_exp,
        pipelines=pipelines,
        X=X_train,
        y=y_train,
        path=os.path.join(time_output_dir, "modelzoom"),
        saving=args.save_plots
    )
    joblib.dump(sample_data_ls, os.path.join(time_output_dir, "modelzoom"))

    #Save BO CSV
    save_bo_history(
        args.run_id,
        sample_data_ls[0]["bo_history"],
        base_output_dir,
        args.run_name,
    )


    # ── Ensemble (on train+val) ──────────────────────────────────────
    X_sel = np.vstack([X_train, X_val])
    y_sel = np.hstack([y_train, y_val])
    
    ensemble, acc, idx_to_keep = do_ensemble(
        sample_data=sample_data_ls[0],
        pipelines=pipelines,
        X=X_sel,
        y=y_sel,
        path=time_output_dir
    )

    final_acc = np.nan
    model_accuracies = {}

    if acc != 0:
        for i, model in enumerate(ensemble, 1):
            acc_model = balanced_accuracy_score(y_test, model.predict(X_test))
            model_accuracies[f"model_{i}_acc"] = acc_model
            print(f"Model {i} test bal acc: {acc_model:.4f}")

        stacked_clf = train_ensemble(ensemble, X_sel, y_sel)
        final_acc = balanced_accuracy_score(y_test, stacked_clf.predict(X_test))
        print(f"Stacked ensemble test bal acc: {final_acc:.4f}")
        np.save(os.path.join(time_output_dir, "stacked_ensemble_acc"), final_acc)

    # ── Dynamic saving ───────────────────────────────────────────────
    save_run_results(
        args.run_id,
        ensemble if acc != 0 else [],
        model_accuracies,
        final_acc,
        base_output_dir
    )

    return sample_data_ls


if __name__ == "__main__":
    main()
