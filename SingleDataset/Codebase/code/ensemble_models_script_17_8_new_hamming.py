# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import balanced_accuracy_score, hamming_loss, make_scorer
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Custom exception for handling timeout scenarios
class TimeoutException(Exception):
    pass

# Function to compute Hamming distance as a percentage
def calculate_hamming_distance(y_pred1, y_pred2):
    """Calculate Hamming distance as a percentage between two prediction arrays."""
    return hamming_loss(y_pred1, y_pred2) * 100

# Function to create an ensemble of models using stacking with Gaussian Naive Bayes
def do_ensemble(
    sample_data, pipelines, X, y, path,
    n_models=100,
    acc_threshold=1.01,
    max_non_improving_models=20,
    max_hamming_distance=2,
    cv_splits=5,
    meta_learner=None,
    stacking_passthrough=True,
):
    """
    Build a stacking ensemble with Hamming-distance diversity filtering.

    Parameters:
      cv_splits            : StratifiedKFold splits used in cross_val_predict 
      meta_learner         : final estimator for StackingClassifier
      stacking_passthrough : whether to pass original features to meta-learner

    Already-configurable parameters (unchanged):
      n_models, acc_threshold, max_non_improving_models, max_hamming_distance
    """
    # Default meta-learner: GaussianNB (same as original)
    if meta_learner is None:
        meta_learner = GaussianNB()

    # Extract information from the sample data
    space = sample_data['space']
    optimizer = sample_data['optimizer']

    # Filter out non-zero top model indices and corresponding pipelines
    top_models_idx = sample_data['top_targets_idx'][np.where(sample_data['top_targets_idx'] != 0)]
    top_pipelines = [pipeline for pipeline in sample_data['tt_estimators'] if pipeline != 0]

    print(f"Number of pipelines: {len(top_pipelines)}")
    idx_to_keep = []
    print("____________________________________________")
    print(f"Creating ensemble based on models_distances with Stacking and GaussianNB")
    print("____________________________________________")

    clf = []  # This will hold the ensemble models
    acc = 0   # This will hold the ensemble accuracy
    skf = StratifiedKFold(n_splits=cv_splits)

    # Initialize a list to store model index and accuracy
    model_accuracies = []

    # Step 1: Identify the most accurate models
    for i, pipeline in enumerate(top_pipelines):
        try:
            y_pred = cross_val_predict(pipeline[0], X, y, cv=skf)
            model_acc = balanced_accuracy_score(y, y_pred)
            model_accuracies.append((i + 1, model_acc, y_pred))  # Store model index, accuracy, and predictions
        except Exception as e:
            print(f"Error encountered during cross_val_predict for model {i+1}: {e}")
            continue

    # Sort the model_accuracies list by accuracy in descending order
    model_accuracies.sort(key=lambda x: x[1], reverse=True)

    # Step 2: Begin ensemble creation
    if model_accuracies:
        best_model_idx, best_model_acc, best_model_preds = model_accuracies[0]
        best_model = top_pipelines[best_model_idx - 1][0]  # Retrieve the most accurate model
        clf.append(best_model)
        acc = best_model_acc
        last_model_preds = best_model_preds  # Store predictions of the most recently added model
        print(f"Most accurate model is Model {best_model_idx} with accuracy: {acc}")

        # Remove the best model from the list of remaining models
        del top_pipelines[best_model_idx - 1]
        top_models_idx = np.delete(top_models_idx, np.where(top_models_idx == best_model_idx))

    # Step 3: Sequentially add models to the ensemble
    consecutive_non_improving = 0
    for i, (model_idx, next_model_acc, next_model_preds) in enumerate(model_accuracies[1:], start=1):
        if consecutive_non_improving >= max_non_improving_models:
            print(f"Stopping after {consecutive_non_improving} consecutive non-improving models.")
            break

        # Skip invalid indices
        if model_idx > len(top_pipelines):
            continue

        # Calculate Hamming distance between the last added model's predictions and the current candidate
        hamming_distance = calculate_hamming_distance(last_model_preds, next_model_preds)
        print(f"Model {model_idx} Hamming Distance with last model: {hamming_distance:.2f}%")

        # Discard the model if Hamming distance is <= max_hamming_distance
        if hamming_distance <= max_hamming_distance:
            print(f"Model {model_idx} discarded due to low Hamming distance ({hamming_distance:.2f}%).")
            continue

        # Try adding the model to the ensemble
        next_model = top_pipelines[model_idx - 1][0]
        provisional_ensemble = clf + [next_model]

        # Train a stacking model with the meta-learner as the final estimator
        estimators = [(f'model_{j}', model) for j, model in enumerate(provisional_ensemble)]
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            passthrough=stacking_passthrough,
        )

        # Perform cross-validation to get ensemble accuracy
        try:
            y_pred_ensemble = cross_val_predict(stacking_clf, X, y, cv=skf)
            new_acc = balanced_accuracy_score(y, y_pred_ensemble)
            print(f"Model {model_idx} provisional ensemble accuracy: {new_acc}")

            # Keep the model if it improves ensemble accuracy by at least acc_threshold
            if new_acc > acc * acc_threshold:
                print(f"Model {model_idx} added to ensemble: Improved accuracy to {new_acc}")
                clf.append(next_model)
                idx_to_keep.append(model_idx)
                acc = new_acc  # Update ensemble accuracy
                last_model_preds = next_model_preds  # Update last model predictions
                consecutive_non_improving = 0  # Reset the counter
            else:
                print(f"Model {model_idx} discarded: No significant improvement (new_acc = {new_acc})")
                consecutive_non_improving += 1

        except Exception as e:
            print(f"Error encountered during stacking: {e}")
            consecutive_non_improving += 1
            continue

    # Save the final ensemble model using joblib
    joblib.dump(clf, f"{path}/ensemble_{acc:.3f}")
    print("Models in the final ensemble:")
    for i, model in enumerate(clf):
        print(f"Model {i+1}: Type = {type(model).__name__}, Model = {model}")

    return clf, acc, idx_to_keep

# Function to train a stacking ensemble classifier
def train_ensemble(ensemble_tuple, X, y, meta_learner=None, stacking_passthrough=True):
    """
    Train a stacking ensemble classifier.

    Parameters:
      meta_learner         : final estimator
      stacking_passthrough : whether to pass original features to meta-learner
    """
    # Default meta-learner: GaussianNB (same as original)
    if meta_learner is None:
        meta_learner = GaussianNB()

    # Impute any NaN that the internal ColumnTransformer pipelines may produce
    # before passing to the meta-learner (GaussianNB does not accept NaN).
    from sklearn.impute import SimpleImputer
    X = SimpleImputer(strategy="median").fit_transform(X)

    estimators = [(f'model_{i}', model) for i, model in enumerate(ensemble_tuple)]
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        passthrough=stacking_passthrough,
    )
    clf.fit(X, y)  # Fit the stacking classifier to the data
    return clf