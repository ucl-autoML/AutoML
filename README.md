# AutoML for Tabular Classification

Automated machine learning for classification problems using tabular CSV data.

The project searches a precomputed space of machine-learning pipelines using Bayesian Optimisation, evaluates candidate models, and builds a diverse stacking ensemble. It supports:

- **Single-dataset evaluation** using an internal train/validation/test split.
- **Multi-dataset evaluation** using one training dataset and one or more external test datasets.

The primary entry points are:

- `SingleDataset/Codebase/Main.py`
- `MultiDataset/Codebase/Main.py`

The repository also includes Sun Grid Engine (SGE) submission scripts for running experiments as cluster array jobs.

## Repository structure

```text
.
├── README.md
├── SingleDataSubmissionFile.sh
├── MultiDataSubmissionFile.sh
├── SingleDataset/
│   └── Codebase/
│       ├── Main.py
│       ├── requirements.txt
│       ├── pipelines.pkl
│       ├── code/
│       │   ├── bayesian_optimization.py
│       │   ├── ensemble_models_script_17_8_new_hamming.py
│       │   ├── sample_space_script_16_8.py
│       │   └── model_list_draft.py
│       └── data/
│           └── space/
│               ├── reduced_space_23_7.npy
│               └── bad_models_23_7.npy
└── MultiDataset/
    └── Codebase/
        └── Same core structure, with external-dataset evaluation
```

The `SingleDataset` and `MultiDataset` directories contain separate copies of the core search-space and pipeline assets. Their main difference is the orchestration performed by `Main.py`.

## How the workflow works

1. Load the training CSV.
2. Separate the target column from the features.
3. Split the training data into training, validation, and test subsets.
4. Load the precomputed pipeline library from `pipelines.pkl`.
5. Load the reduced model-search space from `data/space/`.
6. Run Bayesian Optimisation for the configured search duration.
7. Select high-performing models.
8. Train a stacking ensemble utilising diverse high-performing models.
9. Evaluate individual models and the final ensemble.
10. Save search history, models, metrics, and run summaries.

The multi-dataset workflow additionally evaluates models against external CSV files during model search and final ensemble evaluation.

## Requirements

The current code targets an older Python scientific-computing environment:

- Python 3.8
- NumPy 1.17.4
- pandas 0.25.3
- scikit-learn 0.23.2
- SciPy 1.3.2
- matplotlib 3.1.0
- umap-learn 0.4.1
- XGBoost 1.2.1

The code also imports `joblib`, so install it explicitly if it is not installed by your environment.

### Create an environment

```bash
conda create -n automl-python38 python=3.8.5
conda activate automl-python38

pip install -r requirements.txt
pip install joblib
```


> The pinned dependencies are from an older Python ecosystem and may require an older Conda or pip environment. Installing on modern Python versions may fail because of deprecated package versions.

## Input data format

Input data must be supplied as CSV files.

Each CSV must:

- contain a target column specified by `--target_col` or `TARGET_COL`;
- contain a classification target;
- contain numeric or pipeline-compatible feature columns;
- use compatible feature names and data types across datasets.

Example:

```csv
feature_1,feature_2,feature_3,target
0.52,12,1.4,0
0.81,15,2.1,1
0.33,10,1.1,0
```

The target values should be encoded as integer values such as `0`, `1`, `2` etc.

For the multi-dataset workflow:

- every external CSV must contain the target column;
- external feature columns are aligned to the training columns;

## Single-dataset workflow

The single-dataset workflow trains and evaluates models using one CSV file.

### Command-line usage

From the repository root:

```bash
python SingleDataset/Codebase/Main.py \
  --run_id 1 \
  --base_dir "$PWD/SingleDataset" \
  --results_dir "$PWD/results" \
  --run_name example_run \
  --train_csv_path /absolute/path/to/train.csv \
  --target_col target \
  --exploitation 1 \
  --time_exp 600 \
  --test_size 0.25 \
  --val_size 0.20 \
  --save_plots "False"
```

### Single-dataset parameters

| Parameter | Description | Example |
|---|---|---:|
| `--run_id` | Identifier for the run or SGE task | "$SGE_TASK_ID" |
| `--base_dir` | Path to the dataset workflow directory | "/path/to/SingleDataset" |
| `--results_dir` | Directory where results are written | "/path/to/results" |
| `--run_name` | Human-readable run name | "example_run" |
| `--train_csv_path` | Training CSV path | "/data/train.csv" |
| `--target_col` | Name of the binary target column | "target" |
| `--exploitation` | Bayesian-optimization utility parameter | 1 |
| `--time_exp` | Search duration in seconds | 600 |
| `--test_size` | Fraction reserved for the test set | 0.25 |
| `--val_size` | Fraction of the remaining training data reserved for validation | 0.20 |
| `--save_plots` | Whether to save search-space plots | "True" or "False" |

`TEST_SIZE` and `VAL_SIZE` are fractions. Use 0.25, not 25.

## Multi-dataset workflow

The multi-dataset workflow trains on one primary dataset and evaluates against one or more external datasets.

### Command-line usage

```bash
python MultiDataset/Codebase/Main.py \
  --run_id 1 \
  --base_dir "$PWD/MultiDataset" \
  --results_dir "$PWD/results" \
  --run_name external_example \
  --train_csv_path /absolute/path/to/train.csv \
  --test_csv_paths \
    /absolute/path/to/external_1.csv \
    /absolute/path/to/external_2.csv \
  --target_col target \
  --exploitation 1 \
  --time_exp 600 \
  --test_size 0.25 \
  --val_size 0.20 \
  --save_plots False
```

The `--test_csv_paths` argument accepts one or more space-separated CSV paths.

## Running on an SGE cluster

The repository includes two SGE submission scripts:

- `SingleDataSubmissionFile.sh`
- `MultiDataSubmissionFile.sh`

These scripts request cluster resources, load a Python environment, and execute `Main.py`.

### Configure a single-dataset job

Edit `SingleDataSubmissionFile.sh`:

```bash
BASE_DIR="/absolute/path/to/AutoML/SingleDataset"
RESULTS_DIR="/absolute/path/to/results"
RUN_NAME="example_run"

TRAIN_CSV="/absolute/path/to/train.csv"
TARGET_COL="target"

EXPLOITATION=1
TIME_EXP=600
TEST_SIZE=0.25
VAL_SIZE=0.20
SAVE_PLOTS="False"
```

Submit the job:

```bash
qsub SingleDataSubmissionFile.sh
```

### Configure a multi-dataset job

Edit `MultiDataSubmissionFile.sh`:

```bash
BASE_DIR="/absolute/path/to/AutoML/MultiDataset"
RESULTS_DIR="/absolute/path/to/results"
RUN_NAME="external_example"

TRAIN_CSV="/absolute/path/to/train.csv"
TEST_CSVS="/absolute/path/to/external_1.csv /absolute/path/to/external_2.csv"

TARGET_COL="target"

EXPLOITATION=1
TIME_EXP=600
TEST_SIZE=0.25
VAL_SIZE=0.20
SAVE_PLOTS="False"
```

Submit the job:

```bash
qsub MultiDataSubmissionFile.sh
```

The submission scripts currently use:

```bash
source /share/apps/source_files/python/python-3.8.3.source
```

This is cluster-specific. On another system, replace it with the command used to activate the appropriate Python environment.

The scripts also use the SGE variable `SGE_TASK_ID` as the run identifier. To change the number of array jobs, edit the relevant line:

```bash
#$ -t 1-2
```

For example:

```bash
#$ -t 1-5
```

runs five array tasks.

## Configuration parameters

### `BASE_DIR`

The parent directory of the workflow being executed:

```text
/path/to/AutoML/SingleDataset
/path/to/AutoML/MultiDataset
```

The code appends `Codebase` internally, so do not set this directly to the `Codebase` directory.

### `RESULTS_DIR`

Directory where run outputs are stored.

### `RUN_NAME`

Name used to distinguish experiments and Bayesian-optimization history files.

### `EXPLOITATION`

Controls the exploration/exploitation behavior of the Bayesian-optimization utility function.

The current entry points use the UCB utility function. Values around `0.5` to `1` are reasonable starting points, but the appropriate value depends on the dataset and experiment.

### `TIME_EXP`

Maximum Bayesian-optimization search duration in seconds.

Short values can be useful for smoke tests:

```bash
TIME_EXP=60
```

Longer searches generally evaluate more candidate pipelines:

```bash
TIME_EXP=3600
```

### `SAVE_PLOTS`

When enabled, the search process periodically saves visualizations of the model-search space.

Plots can consume substantial storage during many runs.

## Output files

For each run, the code creates a directory similar to:

```text
results/
└── run_1_example_run/
    ├── modelzoom
    ├── ensemble_0.XXX
    ├── stacked_ensemble_acc.npy
    ├── bo_history/
    │   └── bo_run_1_example_run.csv
    └── runs/
        └── run_1.csv
```

Depending on the workflow and whether an ensemble is successfully created, outputs may include:

- `modelzoom` — serialized Bayesian-optimization results.
- `ensemble_*` — serialized selected ensemble models.
- `stacked_ensemble_acc.npy` — final internal balanced-accuracy score.
- `bo_history/bo_run_*.csv` — pipeline-search history.
- `runs/run_*.csv` — summary of models and evaluation metrics.
- plot files — generated when `SAVE_PLOTS=True`.

The multi-dataset workflow additionally records:

- internal test balanced accuracy;
- per-model external balanced accuracy;
- stacked-ensemble external balanced accuracy;
- external dataset names derived from their filenames.

## Evaluation metric

The main workflow uses **balanced accuracy**:

```text
balanced accuracy = average recall across classes
```

Balanced accuracy is useful for binary classification when the classes are imbalanced.

The Bayesian-optimization search code contains older references to ordinary accuracy, but the current `Main.py` workflows report balanced-accuracy metrics for final evaluation.

## Reproducibility

The search implementation sets a NumPy random seed internally, but the data-splitting commands do not currently expose a configurable random seed through the command line.

For reproducible experiments:

- preserve the exact code revision;
- record all command-line parameters;
- record the Python and dependency versions;
- use the same input files;
- retain the generated result directory;
- record the SGE array task ID.

Multiple SGE array tasks can be used to generate multiple experiment runs.

## Troubleshooting

### `Target column not found`

Check that the target column exists in the CSV and that `--target_col` exactly matches its name:

```bash
head -n 1 /path/to/train.csv
```

### Missing pipeline or search-space files

The `--base_dir` value must point to the workflow directory containing `Codebase`:

```text
/path/to/AutoML/SingleDataset
```

The following files must then exist:

```text
/path/to/AutoML/SingleDataset/Codebase/pipelines.pkl
/path/to/AutoML/SingleDataset/Codebase/data/space/reduced_space_23_7.npy
/path/to/AutoML/SingleDataset/Codebase/data/space/bad_models_23_7.npy
```

### External evaluation errors

For the multi-dataset workflow, verify that:

- every external CSV contains the target column;
- external features are compatible with the training features;
- the target encoding is consistent across datasets;
- all input paths are separated by spaces;
- the shell variable is named `TEST_CSVS`.

### Dependency installation failures

The project uses older pinned dependencies. Try Python 3.8 with Conda, or create an environment matching the versions listed in `requirements.txt`.

## License

No license is currently provided. Add a license before distributing or reusing this project.
