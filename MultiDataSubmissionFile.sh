#$ -S /bin/bash
#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -j y
#$ -pe smp 4
#$ -R y
#$ -l h_rt=2:30:0
#$ -cwd
#$ -t 1-2

# ─────────────────────────────────────────────
# CONFIGURABLE PATHS & HYPERPARAMETERS
# ─────────────────────────────────────────────
BASE_DIR="/Path/To/Code/MultiDataset"
RESULTS_DIR="/Path/To/Multiverse/Results"
RUN_NAME="NameOfYourRun"


# Primary training dataset
TRAIN_CSV="/Path/To/Your/Data.csv"


# Space-separated list of external test datasets
TEST_CSVS="/Path/To/Your/DataExt1.csv /Path/To/Your/DataExt2.csv /Path/To/Your/DataExt3.csv /Path/To/Your/DataExt4.csv"

TARGET_COL="Target"
EXPLOITATION=1
TIME_EXP=600
TEST_SIZE=0.25
VAL_SIZE=0.20
SAVE_PLOTS="False"

# Load Python environment
source /share/apps/source_files/python/python-3.8.3.source

# Execute python array task
python3 "$BASE_DIR"/Codebase/Main.py \
    --run_id "$SGE_TASK_ID" \
    --base_dir "$BASE_DIR" \
    --results_dir "$RESULTS_DIR" \
    --run_name "$RUN_NAME" \
    --train_csv_path "$TRAIN_CSV" \
    --test_csv_paths $TEST_CSVS  \
    --target_col "$TARGET_COL" \
    --exploitation "$EXPLOITATION" \
    --time_exp "$TIME_EXP" \
    --test_size "$TEST_SIZE" \
    --val_size "$VAL_SIZE" \
    --save_plots "$SAVE_PLOTS"
