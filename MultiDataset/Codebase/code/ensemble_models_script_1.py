import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import balanced_accuracy_score


def do_ensemble(sample_data, pipelines, X, y, path, n_models=100,
                cv_splits=5, scoring='balanced_accuracy',
                distance_threshold=0.05, random_state=None):
    """
    Greedy, distance-based ensemble construction (used during BO sampling).

    Starts from the best single pipeline (by GP-predicted performance) and
    greedily adds pipelines that improve cross-validated accuracy, removing
    pipelines that sit too close in the latent space to keep the ensemble
    diverse.

    Parameters:
      n_models           : how many top pipelines to consider
      cv_splits          : StratifiedKFold splits
      scoring            : CV scoring metric
      distance_threshold : min latent-space distance, as a fraction of the
                           max distance, to keep a pipeline 
      random_state       : seed for StratifiedKFold
    """
    space = sample_data['space']
    dimensions = space.shape[1]
    optimizer = sample_data['optimizer']

    sqr_dist = [(space[:, i].max() - space[:, i].min()) ** 2 for i in range(dimensions)]
    max_dist = np.sqrt(np.sum(sqr_dist))

    space_prediction = optimizer._gp.predict(space)
    top_models_idx = np.argsort(space_prediction)[-n_models:]

    idx_to_keep = []
    new_top_models_idx = top_models_idx.copy()
    end = False
    print("____________________________________________")
    print(f"Creating ensemble based on models_distances")
    print("____________________________________________")

    acc = 0  # Initialize the accuracy

    while new_top_models_idx.shape[0] > 1:
        print(f"Number of models to analyse: {new_top_models_idx.shape}")
        print("_____________________________")
        i = 0  # Start indexing from 0 for clarity

        if not idx_to_keep:
            # Start with the best single model
            clf = pipelines[new_top_models_idx[-1]]
            skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
            error = cross_validate(clf, X, y, cv=skf, scoring=scoring, return_train_score=True, return_estimator=True)
            acc = np.mean(error['test_score'])
            print(f"Initial accuracy: {acc}")
            print("_____________________________")
            best = clf
            i += 1
        else:
            new_acc = acc  # Set new_acc to current acc initially

            while i < new_top_models_idx.shape[0]:
                proto_ensemble = [(str(j), pipelines[idx_to_keep[j]]) for j in range(len(idx_to_keep))]
                proto_ensemble.append((str(len(idx_to_keep)), pipelines[new_top_models_idx[i]]))
                clf = VotingClassifier(estimators=proto_ensemble)

                try:
                    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
                    error = cross_validate(clf, X, y, cv=skf, scoring=scoring, return_train_score=True, return_estimator=True)
                    new_acc = np.mean(error['test_score'])
                except Exception as e:
                    print(f"Error during cross-validation: {e}")
                    new_acc = 0

                print(f"Model {i+1}/{new_top_models_idx.shape[0]}: {new_acc} acc")

                if new_acc > acc:
                    idx_to_keep.append(new_top_models_idx[i])
                    acc = new_acc
                    best = clf
                    print(f"Model {i+1} improved accuracy to {new_acc}. Added to ensemble.")
                    break  # Exit the loop if we found a model that improves accuracy
                else:
                    print(f"Model {i+1} did not improve accuracy. Skipping.")

                i += 1

            if new_acc <= acc:
                end = True

        if end or i >= new_top_models_idx.shape[0]:
            break

        # Remove pipelines too close in latent space to keep the ensemble diverse.
        dists = np.sqrt(np.sum((space[new_top_models_idx[i-1]] - space[new_top_models_idx])**2, axis=1))[:i]
        new_top_models_idx = new_top_models_idx[:i]
        args_to_delete = np.where(dists < max_dist * distance_threshold)[0]
        new_top_models_idx = np.delete(new_top_models_idx, args_to_delete)

    idx_to_keep = np.array(idx_to_keep)
    ensemble_top_dist = [(str(i), pipelines[idx_to_keep[i]]) for i in range(idx_to_keep.shape[0])]

    return best, ensemble_top_dist, idx_to_keep


def train_ensemble(ensemble_tuple, X, y):
    clf = VotingClassifier(estimators=ensemble_tuple)
    clf.fit(X, y)
    return clf