"""
AutoML — Multi-Dataset Transferability Array Job
==========================================================================
(Target = Binary Classification)

Trains on a primary dataset and evaluates models (during BO iterations 
and final ensembling) on one or more external test datasets.
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


def get_dataset_name(csv_path):
    """Extracts clean name from path for column header labeling."""
    return os.path.splitext(os.path.basename(csv_path))[0]


def save_run_results(run_id, ensemble, model_accuracies, stacked_accuracies, ext_names, base_output_dir):
    """Saves final run results including internal and external evaluation metrics."""
    runs_dir = os.path.join(base_output_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    
    row = {
        "run_id": run_id,
        "n_models": len(ensemble),
        "stacked_internal_bal_acc": stacked_accuracies.get("internal", np.nan),
    }
    
    # Add stacked ensemble scores on external datasets
    for name in ext_names:
        row[f"stacked_ext_bal_acc_{name}"] = stacked_accuracies.get(f"ext_{name}", np.nan)

    # Add individual model scores
    for i, model in enumerate(ensemble, 1):
        try:
            clf_name = type(model.steps[-1][1]).__name__
        except Exception:
            clf_name = type(model).__name__
        
        row[f"model_{i}"] = clf_name
        row[f"model_{i}_internal_acc"] = model_accuracies.get(f"model_{i}_internal_acc", np.nan)
        for name in ext_names:
            row[f"model_{i}_ext_acc_{name}"] = model_accuracies.get(f"model_{i}_ext_acc_{name}", np.nan)

    pd.DataFrame([row]).to_csv(
        os.path.join(runs_dir, f"run_{run_id}.csv"), index=False
    )


def save_bo_history(run_id, bo_history, base_output_dir, run_name):
    """Saves Bayesian Optimization history to CSV with external metrics included."""
    bo_dir = os.path.join(base_output_dir, "bo_history")
    os.makedirs(bo_dir, exist_ok=True)

    print("\n================ DEBUG CSV SAVE PROCESS ================")
    print(f"Target Directory: {os.path.abspath(bo_dir)}")
    print(f"Directory exists? {os.path.exists(bo_dir)}")

    df = pd.DataFrame(bo_history)
    print(f"Raw DataFrame Shape: {df.shape}")
    print(f"Raw DataFrame Columns: {list(df.columns)}")

    df.insert(0, "run_id", run_id)
    out_file = os.path.join(bo_dir, f"bo_run_{run_id}_{run_name}.csv")
    
    print(f"Full output path: {out_file}")

    # Force write & flush
    df.to_csv(out_file, index=False)
    
    # Confirm file on disk
    if os.path.exists(out_file):
        file_size = os.path.getsize(out_file)
        print(f"SUCCESS: File created! Size = {file_size} bytes.")
        print("First 3 lines of actual CSV file content on disk:")
        with open(out_file, "r") as f:
            for _ in range(3):
                print("  |", f.readline().strip())
    else:
        print("CRITICAL: to_csv executed but file WAS NOT FOUND on disk!")
    print("========================================================\n")


def load_dataset(csv_path, target_col):
    """Load dataset, split target variable, and return feature DataFrame and target array."""
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}")
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    return X, y


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

    # ── Define ModelZoom Subclass for On-The-Fly External Evaluation ───
    # ── Define ModelZoom Subclass for On-The-Fly External Evaluation ───
    class ExternalEvaluatedModelZoom(ModelZoom):
        def __init__(self, external_datasets, ext_names, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.external_datasets = external_datasets
            self.ext_names = ext_names
            self.external_history_records = []

        def train_model(self, index, pipelines, X_train, y_train):
            # 1. Run standard ModelZoom cross-validation
            res = super().train_model(index, pipelines, X_train, y_train)
            [mean_metric], estimator = res
            
            ext_scores = {}

            # 2. Extract fitted model from returned tuple/list
            fitted_model = None

            # Unpack estimator if ModelZoom returned it inside a tuple or list
            if isinstance(estimator, (tuple, list)):
                for item in estimator:
                    # Check for scikit-learn Pipeline or Classifier object
                    if hasattr(item, "predict") or hasattr(item, "steps"):
                        fitted_model = item
                        break
                    # If folds were returned in a sub-list, take the last fold
                    elif isinstance(item, (list, tuple)) and len(item) > 0 and hasattr(item[-1], "predict"):
                        fitted_model = item[-1]
                        break
            elif hasattr(estimator, "predict"):
                fitted_model = estimator

            # 3. Predict on external test set
            if fitted_model is not None:
                for name in self.ext_names:
                    try:
                        X_ext = self.external_datasets[name]["X_raw"]
                        preds = fitted_model.predict(X_ext)
                        score = balanced_accuracy_score(self.external_datasets[name]["y"], preds)
                        ext_scores[f"ext_bal_acc_{name}"] = score
                    except Exception as e:
                        print(f"  [Warning] External evaluation error for '{name}': {e}")
                        ext_scores[f"ext_bal_acc_{name}"] = np.nan
            else:
                for name in self.ext_names:
                    ext_scores[f"ext_bal_acc_{name}"] = np.nan

            self.external_history_records.append(ext_scores)
            return res

    # ── 1. Load Primary & External Data ───────────────────────────────
    X_primary_raw, y_primary = load_dataset(args.train_csv_path, args.target_col)
    
    external_datasets = {}
    ext_names = []
    for ext_path in args.test_csv_paths:
        name = get_dataset_name(ext_path)
        X_ext_raw, y_ext = load_dataset(ext_path, args.target_col)
        
        # Align columns to match primary dataset features
        X_ext_raw = X_ext_raw.reindex(columns=X_primary_raw.columns, fill_value=0)
        
        external_datasets[name] = {"X_raw": X_ext_raw, "y": y_ext}
        ext_names.append(name)

    print(f"[Run {args.run_id}] Primary dataset shape: {X_primary_raw.shape}")
    print(f"[Run {args.run_id}] Loaded {len(external_datasets)} external test dataset(s): {', '.join(ext_names)}")

    # ── 2. Load latent space + pipelines ──────────────────────────────
    kernel = Matern(length_scale=50, length_scale_bounds=(1e-2, 1e3), nu=0.5) + WhiteKernel(noise_level=0.5)
    bad_models = np.load(os.path.join(DATA_DIR, "bad_models_23_7.npy"), allow_pickle=True)
    
    with open(os.path.join(MZ_DIR, "pipelines.pkl"), "rb") as f:
        pipelines = pickle.load(f)

    bad_models.sort()
    for i in range(bad_models.shape[0]):
        del pipelines[bad_models[-i - 1]]

    # ── 3. Train/Val/Test Split (Preserving DataFrame Format) ────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_primary_raw, y_primary, stratify=y_primary, test_size=args.test_size
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, stratify=y_train, test_size=args.val_size
    )

    # ── Output folder ────────────────────────────────────────────────
    base_output_dir = RESULTS_DIR
    os.makedirs(base_output_dir, exist_ok=True)
    time_output_dir = os.path.join(base_output_dir, f"run_{args.run_id}_{args.run_name}")
    os.makedirs(time_output_dir, exist_ok=True)

    # ── 5. BO Sampling & On-The-Fly External Evaluation ───────────────
    test_drive = ExternalEvaluatedModelZoom(
        external_datasets=external_datasets,
        ext_names=ext_names,
        kernel=kernel, 
        uf="ucb", 
        exploitation=args.exploitation
    )
    space = test_drive.load_space(os.path.join(DATA_DIR, "reduced_space_23_7.npy"))

    print(f"\n[Run {args.run_id}] Starting BO Sampling & On-The-Fly External Evaluation...")

    # Pass raw DataFrame X_train_raw so pipelines execute imputation and transformers properly
    sample_data_ls = test_drive.sampling_algo(
        space=space,
        time_exp=args.time_exp,
        pipelines=pipelines,
        X=X_train_raw,
        y=y_train,
        path=os.path.join(time_output_dir, "modelzoom"),
        saving=args.save_plots
    )
    joblib.dump(sample_data_ls, os.path.join(time_output_dir, "modelzoom"))

    # Zip and attach recorded external accuracies directly into bo_history
    # Inside main(), right before calling save_bo_history:
    bo_data = sample_data_ls[0]
    bo_history = bo_data["bo_history"]

    print("\n================ DEBUG ZIP & DICT MERGE ================")
    print(f"Type of bo_history: {type(bo_history)}")
    print(f"Length of bo_history: {len(bo_history)}")
    print(f"Length of external_history_records: {len(test_drive.external_history_records)}")

    if len(bo_history) > 0:
        print(f"Sample raw bo_history[0] entry (type={type(bo_history[0])}):")
        print("  ", bo_history[0])

    for i, (row, ext_record) in enumerate(zip(bo_history, test_drive.external_history_records)):
        if isinstance(row, dict):
            row.update(ext_record)
        else:
            print(f"  [WARNING] Row {i} is NOT a dict! It is: {type(row)} -> {row}")

    if len(bo_history) > 0:
        print("Sample merged bo_history[0] entry:")
        print("  ", bo_history[0])
    print("========================================================\n")

    # Save BO history populated with external generalisation accuracies
    save_bo_history(args.run_id, bo_history, time_output_dir, args.run_name)

    # ── 6. Ensemble Evaluation ────────────────────────────────────────
    X_sel = pd.concat([X_train_raw, X_val_raw], axis=0)
    y_sel = np.hstack([y_train, y_val])
    
    ensemble, acc, idx_to_keep = do_ensemble(
        sample_data=sample_data_ls[0],
        pipelines=pipelines,
        X=X_sel,
        y=y_sel,
        path=time_output_dir
    )

    stacked_accuracies = {}
    model_accuracies = {}

    if acc != 0:
        # Evaluate individual ensemble models
        for i, model in enumerate(ensemble, 1):
            acc_internal = balanced_accuracy_score(y_test, model.predict(X_test_raw))
            model_accuracies[f"model_{i}_internal_acc"] = acc_internal
            print(f"Model {i} Internal Test Acc: {acc_internal:.4f}")

            for name in ext_names:
                acc_ext = balanced_accuracy_score(
                    external_datasets[name]["y"], 
                    model.predict(external_datasets[name]["X_raw"])
                )
                model_accuracies[f"model_{i}_ext_acc_{name}"] = acc_ext
                print(f"Model {i} External [{name}] Acc: {acc_ext:.4f}")

        # Train & evaluate stacked ensemble
        stacked_clf = train_ensemble(ensemble, X_sel, y_sel)
        
        stacked_accuracies["internal"] = balanced_accuracy_score(y_test, stacked_clf.predict(X_test_raw))
        print(f"Stacked Ensemble Internal Test Acc: {stacked_accuracies['internal']:.4f}")

        for name in ext_names:
            acc_ext_stacked = balanced_accuracy_score(
                external_datasets[name]["y"], 
                stacked_clf.predict(external_datasets[name]["X_raw"])
            )
            stacked_accuracies[f"ext_{name}"] = acc_ext_stacked
            print(f"Stacked Ensemble External [{name}] Acc: {acc_ext_stacked:.4f}")

        np.save(os.path.join(time_output_dir, "stacked_ensemble_acc"), stacked_accuracies["internal"])

    # ── 7. Save Comprehensive Run Results ─────────────────────────────
    save_run_results(
        args.run_id,
        ensemble if acc != 0 else [],
        model_accuracies,
        stacked_accuracies,
        ext_names,
        time_output_dir
    )

    return sample_data_ls


if __name__ == "__main__":
    main()