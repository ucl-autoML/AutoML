import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings("ignore")

def do_ensemble(sample_data, pipelines, X, y, path, n_models=100):

    space = sample_data['space']
    dimensions = space.shape[1]
    optimizer = sample_data['optimizer']
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.25, random_state=42)

    sqr_dist = [(space[:, i].max() - space[:, i].min()) ** 2 for i in range(dimensions)]
    max_dist = np.sqrt(np.sum(sqr_dist))

    space_prediction = optimizer._gp.predict(space)
    # pred_std = np.std(space_prediction)

    top_models_idx = np.argsort(space_prediction)[-n_models:]

    idx_to_keep = []
    new_top_models_idx = top_models_idx.copy()
    # Only stop finding pipelines  when all new pipelines have been eliminated
    end = False
    print("____________________________________________")
    print(f"Creating ensemble based on models_distances")
    print("____________________________________________")

    while new_top_models_idx.shape[0] > 1:
        print(f"Number of models to analyse: {new_top_models_idx.shape}")
        print("_____________________________")
        i = -1
        if not idx_to_keep:
            # single model - no ensemble
            
            clf = pipelines[new_top_models_idx[-1]]
            error = cross_validate(clf, X, y, cv=5, scoring="accuracy")# allow to change parameters
            #print("Output of cross_validate")
            #print(error)
            new_acc = np.mean(error['test_score'])
            print(f"Initial accuracy: {new_acc}")
            print(f"Error: {error}")
            print("_____________________________")
            best = clf
            i -=1
        else:
            new_acc = 0

            while new_acc < acc:
                proto_ensemble = [(str(i), pipelines[idx_to_keep[i]]) for i in range(len(idx_to_keep))]
                proto_ensemble.append((str(len(idx_to_keep)), pipelines[new_top_models_idx[i]]))
                clf = VotingClassifier(estimators=proto_ensemble)
                try:
                    error = cross_validate(clf, X, y, cv=5, scoring="accuracy")# allow to change parameters
                    new_acc = np.mean(error['test_score'])
                except:
                    new_acc = 0
                if np.isnan(new_acc):
                    new_acc = 0
                print(f"{-i}/{new_top_models_idx.shape[0]}: {new_acc} acc")
                # Go down the list of best models
                if -i >= new_top_models_idx.shape[0]:
                    end = True
                    break
                i -= 1

        if end:
            break
        idx_to_keep.append(new_top_models_idx[-1])
        # Update new best acc
        best = clf
        acc = new_acc
        # Recalculate distances for new model added
        dists = np.sqrt(np.sum((space[new_top_models_idx[i+1]] - space[new_top_models_idx])**2, axis=1))[:i+1]
        new_top_models_idx = new_top_models_idx[:i+1]
        # Eliminate all models at a distance < 0.1 max distance to the chosen model to be added
        args_to_delete = np.where(dists < max_dist * 0.05)[0]
        new_top_models_idx = np.delete(new_top_models_idx, args_to_delete)

    # Plotting the top models
    idx_to_keep = np.array(idx_to_keep)

    ensemble_top_dist = [(str(i), pipelines[idx_to_keep[i]]) for i in range(idx_to_keep.shape[0])]

    return best, ensemble_top_dist, idx_to_keep

def train_ensemble(ensemble_tuple, X, y):
    clf = VotingClassifier(estimators=ensemble_tuple)
    clf.fit(X, y)
    return clf
