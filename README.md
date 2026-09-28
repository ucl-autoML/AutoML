Placeholder README (to update)

Automated machine learning for tabular data classification tasks. Code allowing external testing within the model search will be uploaded soon.

Submission file holds all the changeable parameters.

BASE_DIR="/Path/To/Code/SingleDataset" - Path to the code, with the path ending at /SingleDataset

RESULTS_DIR="/Path/To/Multiverse/Results" - Path you want the results folder to be located

RUN_NAME="NameOfYourRun" - Name of the run i.e. ADNIMRI

TRAIN_CSV="/Path/To/Your/Data.csv" - Path to CSV file you want to train (and test for Single Dataset) on

TEST_CSV="/Path/To/Your/DataExt1.csv /Path/To/Your/DataExt1.csv" - Space separated list of paths to CSV files you want to externally test on using the MultiData code

TARGET_COL="Target" - Name of the Target column predicting on in numerical format (i.e. binary 0-1)

EXPLOITATION=1 - Search parameter, lower means more localised searching, higher is wider searching (range of between 0.1 and 5, range of 0.5-1 is the default values)

TIME_EXP=600 - Time spent searching the multiverse in seconds (for testing purposes 60-180s is sufficient)

TEST_SIZE=0.25 - Percentage size of test set

VAL_SIZE=0.20 - Percentage size of validation set

SAVE_PLOTS="False" - True/False of saving plots of the multiverse in the results folder. Can start to take up a lot of space if set to True across many runs
