import  numpy as np
from sample_space_script_16_8 import ModelZoom
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import train_test_split
import joblib
import import_ds_draft
from ensemble_models_script_17_8 import do_ensemble
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import os
import argparse
from scipy import stats

parser = argparse.ArgumentParser()

parser.add_argument('-DS', '--ds',
                    dest='ds',
                    help='Dataset we are running')

args = parser.parse_args()

def main(ds_idx):
    
    # Choose kernel
    # This kernel has been found to work well, but any kernel will do
    kernel = Matern(length_scale=50, length_scale_bounds=(1e-2, 1e3), nu=0.5) + WhiteKernel(noise_level=0.5)


    # Run sampling with real data
    pipelines = joblib.load('../pipelines.pkl')
    bad_models = np.load('../data/space/bad_models_23_7.npy', allow_pickle=True)
    bad_models.sort()
    for i in range(bad_models.shape[0]):
        del pipelines[bad_models[-i - 1]]

    test_datasets = [39, 51, 184, 41, 871, 49, 34, 35, 40910, 172, 137, 40978, 336, 885, 171, 1558, 867, 313, 163, 875] # datasets from openml being analysed
    register_periods = [60, 300,900, 1800, 3600, 7200] # 
    print("Start of program")
    print("Building on ModelZoom")


    for count_time, time in enumerate(register_periods):
        for count_ds in range(len(test_datasets)):
            # TODO: TEMP ###############
            # if count_time in [0,1,2,3,4]:
            #     continue
            if count_ds in [2]:
                continue
            #############################
            ds = test_datasets[count_ds]

            print(f"Dataset {ds}")
            print("_____________")
            test_drive = ModelZoom(kernel=kernel)
            # Load space
            space = test_drive.load_space("../data/space/reduced_space_23_7.npy")  # Can also choose TSNE, SVD and PCA space
            # Choose appropriate dimensions - 30 is too vast (curse of dimensionality)
            # space = space[:, :4]
            # space = np.delete(space, bad_models, axis=0)
            # Fetch data
            X, y = import_ds_draft.load_dataset(ds)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            output_dir = f'../results/16_8/{count_ds}/'
            os.makedirs(output_dir, exist_ok=True)


            time_output_dir = f'{output_dir}/{time}'
            os.makedirs(time_output_dir, exist_ok=True)
            test_drive = ModelZoom(kernel=kernel)
            print(f"Model time: {time}")
            sample_data_ls = test_drive.sampling_algo(space=space, time_exp=time, pipelines=pipelines, X=X_train, y=y_train, path=f'{time_output_dir}/modelzoom_{count_ds}_{count_time}')
            joblib.dump(sample_data_ls, f'{time_output_dir}/modelzoom_{count_ds}_{count_time}')
            # prior_space = np.load('PLACEHOLDER') # both memory and time
            # sample_data_ls_w_prior = test_drive.sampling_algo(space=space, pipelines=pipelines, X=X_train, y=y_train,
            #                                           prior_space=prior_space)
            # Ill receive 5 different time points
            # need to run all of them for the ensemble
            # I ll use the ensemble_script that generates 3 types of ensemble tuples

            ensemble, acc = do_ensemble(
                                        sample_data=sample_data_ls[0],
                                        pipelines=pipelines,
                                        X=X_val,
                                        y=y_val,
                                        path=time_output_dir
                                        )
            
            if acc != 0:
                y_pred, _ = stats.mode([model.predict(X_test) for model in ensemble])
                final_acc = accuracy_score(y_test, y_pred.flatten())
                print(f"Final accuracy is {final_acc}")
                np.save(f'{time_output_dir}/final_acc',final_acc)


if __name__ == '__main__':
    main(args.ds)

