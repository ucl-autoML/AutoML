
import numpy as np
from sklearn import *


# Models that we are choosing from
# We are creating a dictionnary with all the options available.
# From the dict, when sampling, it chooses randomly to build and save a pipeline
# If the problem is regression or classification will result in different model choices
# 2 different spaces?

def config_list():
    '''Return list of dictionaries that compose the pipelines.'''
    # TODO: Add probabilities for each dict
    prob_list = []

    imputation_config_dict = {
        'impute.SimpleImputer': {
            'strategy': ['mean', 'median', 'most_frequent'],
            'fill_value': [0]
        },
        'impute.KNNImputer': {
            'n_neighbors': range(2,10),
            'weights': ['uniform', 'distance']
        }
    }
    prob_list.append([0.5, 0.5])

    # 1- IMPUTATION (https://scikit-learn.org/stable/modules/impute.html#impute)
    # 1.1 - UNIVARITE
    # Take the mean
    # Take the median
    # Take the mode
    # (sklearn.impute import SimpleImputer)

    # 1.2 - MULTIVARIATE
    # Nearest neighbour imputation (KNNImputer)
    # Multivariate feature imputation - experimental - not included

    cat_feat_config_dict = {
        'preprocessing.OrdinalEncoder': None,
        'preprocessing.OneHotEncoder': {
            'sparse': [False],
            'handle_unknown': ['ignore']
        }
    }
    prob_list.append([0.5, 0.5])

    preproc_config_dict = {
        'preprocessing.StandardScaler': None,
        'preprocessing.MinMaxScaler': None,
        'preprocessing.MaxAbsScaler': None,
        'preprocessing.RobustScaler': None,
        'preprocessing.PowerTransformer': {
            'method': ['yeo-johnson', 'box-cox']
        },
        'preprocessing.QuantileTransformer': {
            'output_distribution': ['uniform', 'normal']
        },
        'preprocessing.Normalizer': {
            'norm': ['l1', 'l2', 'max']
        },
        'None': None
    }
    prob_list.append([0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1])

    discret_config_dict = {
        'preprocessing.KBinsDiscretizer':{
            'n_bins': range(2,10),
            'encode': ['onehot-dense', 'ordinal'],
            'strategy': ['uniform', 'quantile', 'kmeans']
        },
        'None': None
    }
    prob_list.append([0.5, 0.5])

    dim_red_config_dict = {
        'decomposition.FastICA': {
            'tol': list(np.arange(0.0, 1.01, 0.05))
        },
        'cluster.FeatureAgglomeration': {
            'linkage': ['ward', 'complete', 'average'],
            'affinity': ['euclidean']#, 'l1', 'l2', 'manhattan', 'cosine']
        },
        'decomposition.PCA': {
            'svd_solver': ['randomized'],
            'iterated_power': range(1, 11)
        },
        'feature_selection.VarianceThreshold': {
            'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
        },
        'None': None
    }
    prob_list.append([0.1, 0.1, 0.2, 0.1, 0.5])

        # PREPROCESSING
        # Standardization:
        # StandardScaler
        # MinMaxScaler
        # MaxAbsScaler
        # RobustScaler (for ds w outliers)

        # PowerTransformer
        # QuantileTransformer
        # Normalization

        # Encoding Categorical Features - mandatory?
        # OrdinalEncoder()
        # OneHotEncoder()

        # Discretization - parse continuous features into partitions (bins)
        # KBinsDiscretizer
        # Binarizer()

    models_config_dict = {
        'neighbors.KNeighborsClassifier': {
            'n_neighbors': range(1, 101),
            'weights': ["uniform", "distance"],
            'p': [1, 2, 3],
            'n_jobs': [-1]
        },
        # Kernels can be revised. We can explore more the hyperparams
        'gaussian_process.GaussianProcessClassifier': {
            'kernel': [1.0 * gaussian_process.kernels.RBF(1.0),
                       1.0 * gaussian_process.kernels.Matern(length_scale=1.0, nu=1.5),
                       gaussian_process.kernels.RationalQuadratic(length_scale=1.0, alpha=1.5),
                       gaussian_process.kernels.ExpSineSquared(length_scale=1, periodicity=1),
                       gaussian_process.kernels.DotProduct()],
            'n_restarts_optimizer': range(0,10),
            'max_iter_predict': [20,60,100,150,200],
            'warm_start': [True, False],
            'n_jobs': [-1]
        },
        # Naive Bayes - Need to control for proportionality
        'naive_bayes.GaussianNB': None,
        'naive_bayes.MultinomialNB': {
            'alpha': [0, 0.5, 1],
            'fit_prior': [True, False]
        },
        'naive_bayes.ComplementNB': {
            'alpha': [0, 0.5, 1],
            'fit_prior': [True, False],
            'norm': [True, False]
        },
        'naive_bayes.BernoulliNB': {
            'alpha': [0, 0.5, 1],
            'fit_prior': [True, False],
        },
        'naive_bayes.CategoricalNB': {
            'alpha': [0, 0.5, 1],
            'fit_prior': [True, False],
        },
        'tree.DecisionTreeClassifier': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': range(2,25),
            'max_features': list(np.arange(0.05, 1.01, 0.05)),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'class_weight': ['balanced', None]
        },
        'ensemble.RandomForestClassifier': {
            'n_estimators': [50,100,150,200],
            'criterion': ['gini', 'entropy'],
            'max_depth': range(2,25),
            'max_features': list(np.arange(0.05, 1.01, 0.05)),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'bootstrap': [True, False],
            'n_jobs': [-1]
        },
        'ensemble.AdaBoostClassifier': {
            'n_estimators': [50,100,150,200],
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'algorithm': ['SAMME', 'SAMME.R']
        },
        'ensemble.GradientBoostingClassifier': {
            'loss': ['deviance'],#, 'exponential'],
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'n_estimators': [50,100,150,200],
            'criterion': ['friedman_mse', 'mse', 'mae'],
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'warm_start': [True, False]
        },
        'xgboost.XGBClassifier': {
            'max_depth': range(1, 11),
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'subsample': list(np.arange(0.05, 1.01, 0.05)),
            'min_child_weight': range(1, 21),
            'objective': ['binary:logistic'],
            'n_jobs': [-1]
        },
        'sklearn_rvm.em_rvm.EMRVC':{
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        },
        'linear_model.RidgeClassifier': {
            'alpha': list(np.arange(0.20, 2.01, 0.05))
        },
        'linear_model.LogisticRegression': {
            'penalty': ['l2', 'none'],
            'dual': [True, False],
            'C': list(np.arange(0.20, 2.01, 0.05)),
            'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
            'n_jobs': [-1]

        },
        'svm.LinearSVC':{
            'penalty': ['l1', 'l2'],
            'loss':['hinge', 'squared_hinge'],
            'dual': [True, False],
            'C': list(np.arange(0.20, 2.01, 0.05)),
        },
        'svm.SVC':{
            'C': list(np.arange(0.20, 2.01, 0.05)),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        },
        'neural_network.MLPClassifier':{
            'hidden_layer_sizes': (range(1,100, 2), range(1,100, 2)),
            'activation': ['logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 10],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init':[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
        },


    }

    prob_list.append([0.1176, 0.1176, 0.0235, 0.0235, 0.0235, 0.0235, 0.0235, 0.0588, 0.0588,
                      0.0588, 0.0588, 0.0588, 0.0392, 0.0392, 0.0392, 0.0588, 0.0588, 0.1181])

    # MODELS
    # Linear models - Log Regression, Ridge
    # SVM
    # RVM (???)
    # SGD
    # Nearest Neighbour
    # GPR
    # Naive Bayes
    # Decision Trees
    # Ensemble Methods:
    #   Adaboost
    #   Random forest
    #   XGBoost? - xgboost.XGBRegressor
    # Ridge
    # Neural Nets - MLP

    list_config = [imputation_config_dict, cat_feat_config_dict, preproc_config_dict,
                   discret_config_dict, dim_red_config_dict, models_config_dict]

    return list_config, prob_list

