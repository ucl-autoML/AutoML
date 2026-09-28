"""
Code to generate the predictions for each dataset. It is optimized
to not drain the RAMs memory and to process the code in a child
process, allowing for a time limit and containing code freezes

Solves one dataset per run of script (-DS). no loop
"""

import import_ds
# import choose_pipelines
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import numpy as np
import time
from pathlib import Path
import joblib
import pickle  # is it worth having the two? they do the same
import sys
import argparse
import concurrent
import signal
from tqdm import tqdm

PATH = Path.cwd().parent

parser = argparse.ArgumentParser()

parser.add_argument('-DS', '--ds_nr',
                    dest='ds_nr',
                    help='Ordinal that defines where to start on ds_id_list')
parser.add_argument('-B', '--batch',
                    dest='batch',
                    default=None,
                    help='Batch we re running')

args = parser.parse_args()


# ---------- New script---------- #


class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


def run_model(dict_data):
    i = dict_data['i']

    X_train = dict_data['X_train']
    X_test = dict_data['X_test']
    y_train = dict_data['y_train']
    y_test = dict_data['y_test']
    model = clone(dict_data['pipeline_log'][i])  # we clone to avoid memory expenditure

    # try:
    tic_train = time.time()
    model.fit(X_train, y_train)
    time_train = time.time() - tic_train
    # print('Time to train: {}'.format(time_train))

    tic_test = time.time()
    y_hat = model.predict(X_test)
    time_test = time.time() - tic_test

    acc = accuracy_score(y_test, y_hat)

    resources0 = time_train
    resources1 = time_test

    predictions = y_hat
    # no error
    error = 0

    p = pickle.dumps(model)
    memory = sys.getsizeof(p)
    return (resources0, resources1, predictions, acc, memory, error)


def main(ds_nr, batch=None):
    '''

    Parameters
    ----------
    ds_nr: int - Ordinal number of ds to train on (from import_ds list)
    batch: int - Batch of pipelines training on (we are breaking the 20k pipelines into batches of 2k)

    Returns
    -------

    '''
    time_limit = 300
    nr_pipelines = 20000
    if batch is not None:
        batch = int(batch)
        n_batch = 2000
        file_suffix = str(batch)
    else:
        n_batch = 20000
        batch = 0
        file_suffix = ''

    # Sets a signal to break the training if it's taking too long
    signal.signal(signal.SIGALRM, timeout_handler)

    pipeline_log = joblib.load(PATH / 'pipelines.pkl')

    # Make sure models take advantage of parallel processing (n_jobs = -1)
    for i in range(nr_pipelines):
        params = pipeline_log[i][-1].get_params()
        params['n_jobs'] = -1
        try:
            pipeline_log[i][-1].set_params(**params)
        except ValueError:
            pass

    print('> Models loaded to explore every node available.')
    ds_nr = int(ds_nr)
    ds_id_list = import_ds.ds_list()
    ds = ds_id_list[ds_nr]

    dict_data = {
        'ds_id_list': ds_id_list,
        'pipeline_log': pipeline_log
    }

    # X, y = import_ds.get_dataset(ds)
    X, y = import_ds.load_dataset(ds)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    print(f'> Dataset {ds_nr} loaded.')

    resources = np.zeros((nr_pipelines, 2))
    memory = np.zeros(nr_pipelines)
    acc = np.zeros((nr_pipelines, 1))
    error_message = []
    predictions = []

    dict_data['X_train'] = X_train
    dict_data['X_test'] = X_test
    dict_data['y_train'] = y_train
    dict_data['y_test'] = y_test

    for i, batch_i in enumerate(tqdm(range(batch*n_batch, (batch+1) * n_batch))):

        dict_data['i'] = batch_i
        print(f'> Training model {batch_i}.')
        # Set time limit for training (sec)
        signal.alarm(time_limit)

        try:
            (resources0, resources1, temp_predictions, temp_acc, temp_memory, error) = \
                run_model(dict_data)
        except TimeoutException as e:
            print(f'> Model {batch_i} timed out.')
            resources0 = -1
            resources1 = -1
            temp_acc = 0
            temp_predictions = np.ones_like(y_test) * -1
            temp_memory = -1
            error = ((i, e))
            # continue

        except Exception as e:
            print(f'> Model {i} threw an error. {e}')
            resources0 = -1
            resources1 = -1
            temp_acc = 0
            temp_predictions = np.ones_like(y_test) * -1
            temp_memory = -1
            error = ((i, e))
            signal.alarm(0)
        else:
            # Reset the alarm
            signal.alarm(0)


        resources[i, 0] = resources0
        resources[i, 1] = resources1
        predictions.append(temp_predictions)
        acc[i] = temp_acc
        memory[i] = temp_memory
        if error != 0:
            error_message.append(error)

    # create folder
    ds_dir = PATH / 'data' / str(ds_nr)
    ds_dir.mkdir(exist_ok=True)
    np.save(ds_dir / f'acc{file_suffix}', acc)
    np.save(ds_dir / f'resources{file_suffix}', resources)
    np.save(ds_dir / f'predictions{file_suffix}', np.array(predictions))
    np.save(ds_dir / f'memory{file_suffix}', memory)
    with open(ds_dir / f'errors{file_suffix}.txt', 'w') as fp:
        fp.write('\n'.join('%i - %s' % x for x in error_message))


if __name__ == '__main__':
    main(args.ds_nr, args.batch)
