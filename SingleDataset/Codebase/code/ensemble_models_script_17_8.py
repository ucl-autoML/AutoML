import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from scipy import stats
import joblib

import warnings
warnings.filterwarnings("ignore")

class TimeoutException(Exception):   # Custom exception class
    pass

def do_ensemble(sample_data, pipelines, X, y, path, n_models=100):

    space = sample_data['space']
    dimensions = space.shape[1]
    optimizer = sample_data['optimizer']
    # X_train, X_test, y_train, y_test = train_test_split(
    # X, y, test_size=0.25, random_state=42)

    sqr_dist = [(space[:, i].max() - space[:, i].min()) ** 2 for i in range(dimensions)]
    max_dist = np.sqrt(np.sum(sqr_dist))

    space_prediction = optimizer._gp.predict(space)
    # pred_std = np.std(space_prediction)

    top_models_idx = sample_data['top_targets_idx'][np.where(sample_data['top_targets_idx']!=0)]
    top_pipelines = [pipeline for pipeline in sample_data['tt_estimators'] if pipeline != 0]

    idx_to_keep = []
    new_top_models_idx = top_models_idx.copy()
    # Only stop finding pipelines  when all new pipelines have been eliminated
    end = False
    print("____________________________________________")
    print(f"Creating ensemble based on models_distances")
    print("____________________________________________")
    clf = []
    acc = 0
    while new_top_models_idx.shape[0] > 1:
        print(f"Number of models to analyse: {new_top_models_idx.shape}")
        print("_____________________________")
        i = 0

        if not idx_to_keep:
            # single model - no ensemble
            
            clf.append(top_pipelines[0][0])
            y_pred = clf[0].predict(X)
            print(f"Predicted labels (y_pred): {y_pred}")
            new_acc = accuracy_score(y, y_pred)
            # cross_validate(clf, X, y, cv=5, scoring="accuracy")# allow to change parameters
            # new_acc = np.mean(error['test_score'])
            print(f"Initial accuracy: {new_acc}")
            print("_____________________________")
            best = clf
            i += 1
        else:
            new_acc = 0

            while new_acc < acc:

                try:
                    y_pred, _ = stats.mode([model.predict(X) for model in clf]
                                        +[top_pipelines[0][0].predict(X)])
                    
                    y_pred = y_pred.flatten()
                    new_acc = accuracy_score(y, y_pred)
                except:
                    new_acc =  0

               # proto_ensemble = [(str(i), pipelines[idx_to_keep[i]]) for i in range(len(idx_to_keep))]
               # proto_ensemble.append((str(len(idx_to_keep)), pipelines[new_top_models_idx[i]]))
               # clf = VotingClassifier(estimators=proto_ensemble)
               # try:
                #    error = cross_validate(clf, X, y, cv=5, scoring="accuracy")# allow to change parameters
                #    new_acc = np.mean(error['test_score'])
               # except:
                #    new_acc = 0
               # if np.isnan(new_acc):
                #    new_acc = 0
               # print(f"{i}/{new_top_models_idx.shape[0]}: {new_acc} acc")
               # Go down the list of best models
                if i >= new_top_models_idx.shape[0]:
                    end = True
                    i += 1
                    break
                i += 1

        if end or i>=new_top_models_idx.shape[0]:
            break
        clf.append(top_pipelines[0][i-1])
        idx_to_keep.append(new_top_models_idx[i-1])
        # Update new best acc
        # best = clf
        acc = new_acc
        # Recalculate distances for new model added
        dists = np.sqrt(np.sum((space[new_top_models_idx[i].astype(int)] - 
                                np.take(space, new_top_models_idx.astype(int), axis=0)
                                )**2, axis=1))[i:]
        new_top_models_idx = new_top_models_idx[i:]
        top_pipelines = top_pipelines[i:]
        # Eliminate all models at a distance < 0.1 max distance to the chosen model to be added
        args_to_delete = np.where(dists < max_dist * 0.05)[0]
        new_top_models_idx = np.delete(new_top_models_idx, args_to_delete)
        top_pipelines = [top_pipelines[n] for n in range(len(top_pipelines))
                         if n not in args_to_delete]

    # Plotting the top models
    idx_to_keep = np.array(idx_to_keep)

   # ensemble_top_dist = [(str(i), pipelines[idx_to_keep[i]]) for i in range(idx_to_keep.shape[0])]
    joblib.dump(clf, f"{path}/ensemble_{acc:.3f}")
    
    return clf, acc, idx_to_keep

def train_ensemble(ensemble_tuple, X, y):
    clf = VotingClassifier(estimators=ensemble_tuple)
    clf.fit(X, y)
    return clf
