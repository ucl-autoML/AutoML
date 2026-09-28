# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, make_scorer
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Custom exception for handling timeout scenarios
class TimeoutException(Exception):
    pass

# Function to create an ensemble of models using stacking with Gaussian Naive Bayes
def do_ensemble(sample_data, pipelines, X, y, path, n_models=100, acc_threshold=1.01, max_non_improving_models=10):
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
    clf = []
    acc = 0  # This will hold the ensemble accuracy
    skf = StratifiedKFold(n_splits=5)

    # Initialize a list to store model index and accuracy
    model_accuracies = []

    # Step 1: Identify the most accurate models (without printing individual accuracies)
    for i, pipeline in enumerate(top_pipelines):
        try:
            y_pred = cross_val_predict(pipeline[0], X, y, cv=skf)
            model_acc = balanced_accuracy_score(y, y_pred)
            model_accuracies.append((i + 1, model_acc))  # Append the model index and accuracy
        except Exception as e:
            print(f"Error encountered during cross_val_predict for model {i+1}: {e}")
            continue

    # Sort the model_accuracies list by accuracy in descending order
    model_accuracies.sort(key=lambda x: x[1], reverse=True)

    # Print the sorted models and their accuracies
    print("\nSorted models by accuracy (most to least accurate):")
    for model_idx, acc in model_accuracies:
        print(f"Model {model_idx}: Accuracy = {acc}")

    # Step 2: Begin ensemble creation with the first model in the sorted list
    if model_accuracies:
        best_model_idx, best_model_acc = model_accuracies[0]
        best_model = top_pipelines[best_model_idx - 1][0]  # Retrieve the most accurate model
        clf.append(best_model)
        acc = best_model_acc
        print(f"Most accurate model is Model {best_model_idx} with accuracy: {acc}")

        # Remove the best model from the list of remaining models
        top_pipelines.pop(best_model_idx - 1)
        new_top_models_idx = np.delete(top_models_idx, best_model_idx - 1)

        # Step 3: Add models sequentially until 10 consecutive models don't improve accuracy
        consecutive_non_improving = 0
        for i, (model_idx, next_model_acc) in enumerate(model_accuracies[1:], start=1):
            if consecutive_non_improving >= max_non_improving_models:
                print(f"Stopping after {consecutive_non_improving} consecutive non-improving models.")
                break

            # Adjust index to ensure it's within bounds
            try:
                next_model = top_pipelines[i][0]
            except IndexError:
                print(f"Index error encountered at model {model_idx}. Skipping this model.")
                consecutive_non_improving += 1
                continue

            try:
                # Get accuracy of the next model
                y_pred_next = cross_val_predict(next_model, X, y, cv=skf)
                next_model_acc = balanced_accuracy_score(y, y_pred_next)

                # Create provisional ensemble
                provisional_ensemble = clf + [next_model]

                # Train a stacking model with Gaussian Naive Bayes as the final estimator
                estimators = [(f'model_{j}', model) for j, model in enumerate(provisional_ensemble)]
                stacking_clf = StackingClassifier(estimators=estimators, final_estimator=GaussianNB(), passthrough=True)

                # Perform cross-validation to get ensemble accuracy
                y_pred_ensemble = cross_val_predict(stacking_clf, X, y, cv=skf)
                new_acc = balanced_accuracy_score(y, y_pred_ensemble)
                print(f"Model {model_idx} provisional ensemble accuracy: {new_acc}")

                # Keep the model if it improves ensemble accuracy by at least 1%
                if new_acc > acc * acc_threshold:
                    print(f"Model {model_idx} added to ensemble: Improved accuracy to {new_acc}")
                    clf.append(next_model)
                    idx_to_keep.append(model_idx)
                    acc = new_acc  # Update ensemble accuracy
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
def train_ensemble(ensemble_tuple, X, y):
    # Create a list of estimators with a naming convention for each model
    estimators = [(f'model_{i}', model) for i, model in enumerate(ensemble_tuple)]
    final_estimator = GaussianNB()  # Final estimator in the stack
    # Create a StackingClassifier with passthrough option to use raw features along with predictions
    clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator, passthrough=True)
    clf.fit(X, y)  # Fit the stacking classifier to the data
    return clf  # Return the trained stacking classifier

# Function to evaluate a model using cross-validation
def evaluate_model(clf, X, y):
    skf = StratifiedKFold(n_splits=5)  # Create a stratified k-fold cross-validator
    scorer = make_scorer(balanced_accuracy_score)  # Define a scorer using balanced accuracy
    scores = cross_validate(clf, X, y, cv=skf, scoring=scorer)  # Perform cross-validation
    return scores  # Return the cross-validation scores
