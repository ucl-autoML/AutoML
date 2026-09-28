conda version -> 4.8.3
python version -> 3.8.5


### Create virtual environment
conda create -n venvname python=3.8.5 anaconda
conda activate venvname

### Install packages from requirements file
conda install --file requirements.txt 

### Running the code
Go to folder ./code
main_script_16_8.py has the main code to run ModelZoom + ensemble in the space. Check this code
To run MZ on 10 predefined datasets with 6 different running periods run
python main_script_16_8.py --ds 0

The sampled pipelines that create the space are in ./pipelines.pkl
The space is organised by the file ./data/space/reduced_space_23_7.npy
The file code/model_list_draft.py has all the different combinations available to be sampled for pipelines in the space.

