import runpy
from pathlib import Path
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_validate
from bayesian_optimization import Bayesian_Optimization, UtilityFunction
# import create_space
from ensemble_models_script_1 import do_ensemble
import time
import signal


def set_random_seed(seed):
    np.random.seed(seed)
    print(f'Random seed {seed} has been set.')


class TimeoutException(Exception):   # Custom exception class
    pass


def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


# -----------------------------------------------------------------------------
# DEFAULT WARM START INDICES
# Pre-computed warm-start pipeline indices (obtained in analyse_meta_data.py).
# -----------------------------------------------------------------------------
DEFAULT_WARM_START = [18435, 2521, 13139, 15719, 4674,
                      10376, 10041, 13008, 16132, 16817]
# Previous version (kept for reference):
# DEFAULT_WARM_START = [15263, 14310, 13771, 10299, 17872,
#                       16153, 18633, 5449, 6607, 5422]


class ModelZoom():
    """ Description of optimization techniques
    Params etc"""

    def __init__(self, kernel, random_state=42, scoring='accuracy', cv=5, uf='ei',
                 exploitation=1e-1, utility_warmup=1000, utility_iter=10,
                 warm_start=None, model_time_limit=300, n_neighbors=4,
                 top_n=100, distance_threshold=0.1):
        self.kernel = kernel
        self.random_state = random_state
        self.scoring = scoring
        self.cv = cv
        self.uf = uf
        self.utility_warmup = utility_warmup
        self.utility_iter = utility_iter
        self.exploitation = exploitation

        self.warm_start = warm_start if warm_start is not None else list(DEFAULT_WARM_START)
        self.model_time_limit = model_time_limit     
        self.n_neighbors = n_neighbors               
        self.top_n = top_n                           
        self.distance_threshold = distance_threshold  

    # Load the space
    def load_space(self, path):
        space = np.load(path)
        return space

    def train_model(self, index, pipelines, X_train, y_train):
        # If no model was identified, skip
        time_limit = self.model_time_limit
        if index[0][0] == -1:
            return [0], 0

        try:
            model = pipelines[index[0][0]]
        except TypeError:
            model = pipelines[index[0][0][0]]
        try:
            signal.alarm(time_limit)
            metric = cross_validate(model, X_train, y_train,
                                   cv=self.cv,
                                   scoring=self.scoring,
                                   return_estimator=True)

            estimator = metric['estimator']
            mean_metric = np.mean(metric['test_score'])
        except TimeoutException as e:
            mean_metric = 0
            estimator = [0]
        except:  # If model fails to run
            mean_metric = 0
            estimator = [0]
            signal.alarm(0)

        if mean_metric != mean_metric:  # Checking if the value is nan
            mean_metric = 0
            estimator = [0]

        return [mean_metric], estimator

    @staticmethod
    def f(index, metric):
        # If no model was identified, skip
        if index[0][0] == -1:
            return [0]
        mean_metric = metric[index].ravel()
        return mean_metric

    # Initialize BO
    def initialize_BO(self, space):
        dimensions = space.shape[1]
        # Create the space boundaries based on the min/max per dimension on `space`
        pbounds = {f"b{i}":(min(space[:, i]), max(space[:, i])) for i in range(dimensions)}
        # Measure max distance based on space min/max
        sqr_dist = [(space[:, i].max()-space[:, i].min())**2 for i in range(dimensions)]
        max_dist = np.sqrt(np.sum(sqr_dist))
        # Define the kernel. These values depend on the bounds of the search space, noise etc.
        kernel = self.kernel
        # Initialize KNN to pick closest model to point sampled
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(space)  # Space: Models x Dimensions
        # Define the optimizer
        optimizer = Bayesian_Optimization(pbounds=pbounds,
                                          random_state=self.random_state)

        if self.uf == 'ei':
            # Utility function used to sample next (Expectation Improvement)
            utility = UtilityFunction(function='ei', hyperparam=self.exploitation)
        elif self.uf == 'ucb':
            utility = UtilityFunction(function='ucb', hyperparam=self.exploitation)
        elif self.uf == 'poi':
            utility = UtilityFunction(function='poi', hyperparam=self.exploitation)
        else:
            raise KeyError(f'uf value {self.uf} is not accepted. Please choose from [ei, ucb, poi]')
        # TODO: Utility function to measure Stopping Criteria
        optimizer._gp.set_params(kernel=kernel, normalize_y=True)

        return optimizer, utility, nbrs, pbounds, max_dist

    def choose_sample(self, nbrs, sample, max_dist, prior_space=None):
        """

        :param nbrs:
        :param priority_space:
        :return:
        """

        distance, index = nbrs.kneighbors(sample)

        args_to_delete = np.where(distance > max_dist * self.distance_threshold)[1]
        if args_to_delete.size > 0:
            distance = np.delete(distance, args_to_delete)
            distance = distance[np.newaxis, :]
            index = np.delete(index, args_to_delete)
            index = index[np.newaxis, :]
        # If point to sample further than 10% of space -> 0
        if distance.size == 0:  # All samples were deleted
            # target = [0]
            index = [[-1]]
            distance = [[-1]]
        else:
            if prior_space is not None:
                # Choose point that minimizes prior
                best_prior = prior_space[np.array(index).ravel()]
                slot = np.argmin(best_prior, axis = 0)
            else:
                slot = 0
            index = [[np.array(index[0])[slot].ravel()]]
            distance = [[distance[:, slot]]]

        return distance, index

    def sampling_algo(self, space, burn_in=10, max_sample=100000, metric_path=None, time_exp=1000000,
                      plot=True, pipelines=None, X=None, y=None, prior_space=None, path='None', saving=True):
        """
        :param space:
        :param burn_in:
        :param max_sample:
        :param metric_path:
        :param plot:
        :param pipelines:
        :param X:
        :param y:
        :return:
        """

        set_random_seed(self.random_state)
        signal.signal(signal.SIGALRM, timeout_handler)
        # Warm start indices are now taken from self.warm_start (set in __init__),
        # instead of being hard-coded here. Default value is unchanged.
        # (Obtained in analyse_meta_data.py script.)
        warm_start = self.warm_start
        sample_data_ls = []

        dimensions = space.shape[1]
        optimizer, utility, nbrs, pbounds, max_dist = self.initialize_BO(space)

        sample_data = {}
        sample_data['space'] = space
        sample_data['utility'] = utility
        sample_data['time'] = []
        sample_data['ensemble'] = []
        sample_data['ensemble_list'] = []
        sample_data['weights'] = []
        sample_data['tt_estimators'] = list(np.zeros(self.top_n))
        sample_data['top_targets'] = np.zeros(self.top_n)
        sample_data['top_targets_idx'] = np.zeros(self.top_n)
        sample_data['bo_history'] = []

        if pipelines is None:  # Comparing to predated metric instead
            sample_data['metric'] = np.load(metric_path)

        indices = []
        initial_time = time.time()
        # print(f"Model time: {time_exp}")
        for it in range(burn_in + max_sample):

            timeit = time.time()
            total_time = timeit - initial_time
            sample_data['time'].append(total_time)
            if total_time > time_exp:
                sample_data['optimizer'] = optimizer
                sample_data['indices'] = indices
                ensm, ensemble_top_dist, weight_list = do_ensemble(sample_data, pipelines, X, y, path)
                sample_data['ensemble'].append(ensm)
                sample_data['ensemble_list'].append(ensemble_top_dist)
                sample_data['weights'].append(weight_list)
                sample_data_ls.append(sample_data)
                # sample_data['time'].append(time.time() - initial_time)
                print(f"Max is {optimizer.max}")
                print("_________________________________")
                return sample_data_ls  # Finish run

            # If burn-in (5 first its)
            if it < burn_in:
                # Choose point in space to probe next in search space randomly
                next_point_to_probe = {f'b{dim}': space[warm_start[it], dim]
                                       for dim in range(dimensions)}
                # print("Next point to probe is:", next_point_to_probe)

            else:
                # Choose point in space to probe next in search space using optimizer
                next_point_to_probe = optimizer.suggest(utility,
                                                        n_warmup=self.utility_warmup,
                                                        n_iter=self.utility_iter)
                # print("Next point to probe is:", next_point_to_probe)
            # Convert dict to array
            sample = np.array(list(next_point_to_probe.values()))
            # Convert to (model x dimensions) shape
            sample_ = sample[np.newaxis, :]
            if it < burn_in:
                distance, index = self.choose_sample(nbrs, sample_, max_dist)
            else:
                # Get point from space to sample (discrete space)
                distance, index = self.choose_sample(nbrs, sample_, max_dist, prior_space=prior_space)

            # Measure error from coordinates
            if pipelines is None:
                target, estimator = self.f(index, sample_data['metric'])
            else:
                target, estimator = self.train_model(index, pipelines, X, y)

            indices.append(index[0][0])

            # Save obtained values
            register_sample = {f'b{j}': sample[j] for j in range(dimensions)}
            # TODO: Build stopping criteria

            optimizer.register(params=register_sample, target=target[0])

            # Record Bayesian optimisation history

            # Convert pipeline index to a plain integer
            try:
                pipeline_idx = int(np.asarray(index[0][0]).item())
            except Exception:
                try:
                    pipeline_idx = int(index[0][0][0])
                except Exception:
                    pipeline_idx = -1


            # Try to determine the model name
            if pipeline_idx == -1:
                model_name = "InvalidPipeline"
            else:
                try:
                    model_name = pipelines[pipeline_idx].named_steps["model"].__class__.__name__
                except Exception:
                    try:
                        model_name = pipelines[pipeline_idx].steps[-1][1].__class__.__name__
                    except Exception:
                        model_name = pipelines[pipeline_idx].__class__.__name__

            sample_data["bo_history"].append({
                "iteration": it + 1,
                "pipeline_index": pipeline_idx,
                "model_type": model_name,
                "balanced_accuracy": float(target[0])
            })


            if np.any(target[0] > sample_data['top_targets']):
                # Save target, index and estimator because its on the top performers
                pos = np.where(target[0] > sample_data['top_targets'])[0][0]

                sample_data['top_targets'] = np.concatenate((sample_data['top_targets'][:pos], target, sample_data['top_targets'][pos:-1]))
                sample_data['top_targets_idx'] = np.concatenate((sample_data['top_targets_idx'][:pos], index[0][0], sample_data['top_targets_idx'][pos:-1]))
                sample_data['tt_estimators'].insert(pos, estimator)
                sample_data['tt_estimators'] = sample_data['tt_estimators'][:self.top_n]

            # plot results every 50 trials, for bookeeping
            if plot and it % 50 == 49:
                print(f'Iteration: {it}')
                print(f'Result: {target[0]}')
                figure_path = f'{path}_iteration_{it}.png'
                sample_data['optimizer'] = optimizer
                if metric_path is None:
                    ground_truth = False
                else:
                    ground_truth = True
                if saving:
                    self.plot_and_save(sample_data, figure_path, ground_truth)

        sample_data['optimizer'] = optimizer
        sample_data['indices'] = indices
        sample_data_ls.append(sample_data)
        return sample_data_ls

    def plot_and_save(self, sample_data, path, ground_truth=True, figsize=(20, 10)):

        space = sample_data['space']
        optimizer = sample_data['optimizer']
        utility = sample_data['utility']
        dimensions = space.shape[1]
        if ground_truth:
            metric = sample_data['metric']

        plt.figure(figsize=figsize)
        for lat in range(dimensions-1):
            # If  we have access to the metrics, plot real space
            if ground_truth:
                plt.subplot(dimensions-1,3,(lat*3)+1)
                plt.scatter(space[:, lat], space[:, lat+1], c=metric.ravel(), s=1, marker=',',)
                plt.xticks([])
                plt.yticks([])
                plt.title(f"Original Space ({lat},{lat+1})", fontdict={'fontsize': 8})

            plt.subplot(dimensions-1,3,(lat*3)+2)
            y_pred = optimizer._gp.predict(space)
            plt.scatter(space[:, lat], space[:, lat+1], c=y_pred, s=1, marker=',',)
            plt.xticks([])
            plt.yticks([])
            if ground_truth:
                r_score = optimizer._gp.score(space, metric)
            else:
                r_score = None
            plt.title(f"GP Score {r_score}", fontdict={'fontsize': 8})

            plt.subplot(dimensions-1,3,(lat*3)+3)
            z_obs = optimizer.res
            z_obs = np.array([np.array(list(z_obs[i]['params'].values())) for i in range(len(z_obs))])
            plt.scatter(space[:, lat], space[:, lat+1],
                        cmap='Greys',
                        c=utility.utility(space, optimizer._gp,optimizer._target.max()),
                        s=1, marker=',',)
            plt.scatter(z_obs[:,lat], z_obs[:,lat+1], marker='X',
                        c=np.array(range(z_obs.shape[0])),
                        cmap='Reds')
            plt.xticks([])
            plt.yticks([])
            plt.title("Utility function across space", fontdict={'fontsize': 8})

        plt.savefig(path)
       # plt.savefig(f'../results/{i}.png')
        plt.close()
