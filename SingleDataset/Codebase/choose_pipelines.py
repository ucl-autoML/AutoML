import numpy as np
from sklearn import impute, preprocessing, decomposition, cluster, \
                    feature_selection, neighbors, gaussian_process, naive_bayes, tree, \
                    ensemble, linear_model, svm, neural_network
import xgboost
#import sklearn_rvm
from sklearn.pipeline import Pipeline
import model_list_draft
import random
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
import pandas as pd


# create class that transforms X array into X dataframe - utils(?)
class DataframeTransformer:
    def __init__(self):
        pass

    def transform(self, input_df, **transform_params):
        return pd.DataFrame(input_df)

    def fit(self, X, y=None, **fit_params):
        return self


def sample_pipelines(nr_pipelines, random_seed=42):

    np.random.seed(random_seed)

    step_list = ['impute', '', 'preproc', 'discrete', 'dim_red', 'model']
    list_config, prob_config = model_list_draft.config_list()
    pipeline_log = []
    pipe_list_dict = []

    for nr in range(nr_pipelines):
        pipeline_steps = []
        pipeline_dict = {}# dictionary
        # Choose a method per value in list
        for step in range(len(list_config)):
            # For each pipeline step get a method
            method_str = np.random.choice(list(list_config[step]), p=prob_config[step])
            method = eval(method_str)

            # If method is None, then ignore step
            if method is None:
                continue
            # For each method randomly sample the hyperparameters
            try:
                arg_dict = list_config[step][method_str].copy()
                for key in arg_dict.keys():
                    arg_dict[key] = random.sample(arg_dict[key], 1)[0]

                algo = method(**arg_dict)
            except AttributeError:
                arg_dict = None
                algo = method()

            # The following step is taken because cat and num columns receive different preprocessing
            # Standardization of numerical columns
            if step == 1:
                categorical_transformer = algo
                pipeline_dict[method_str] = arg_dict
                continue
            # Cat encoding of categorical columns
            elif step == 2:
                numerical_transformer = algo
                # ColumnTransformer chooses the correct columns to apply preproc
                algo = ColumnTransformer(transformers=[
                    ('cat', categorical_transformer, selector(dtype_include="category")),
                    ('num', numerical_transformer, selector(dtype_exclude="category"))
                ])

            # Add the hyperparameterized method to the Pipeline
            pipeline_steps.append((step_list[step], algo))
            pipeline_dict[method_str] = arg_dict

            # Before the preprocessing steps, I need to convert to dataframe to find categorical columns
            if step == 0:
                pipeline_steps.append(('to_df', DataframeTransformer()))

        # Append pipeline
        pipe = Pipeline(pipeline_steps)
        # Log sample choices - lista de dicionarios
        pipeline_log.append(pipe) # pipe?
        pipe_list_dict.append(pipeline_dict)

    return pipeline_log, pipe_list_dict

