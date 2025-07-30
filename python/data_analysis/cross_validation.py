import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_cross_validation(
    dataset,
    feature_extractor,
    classifier=None,
    n_splits=5,
    random_state=42
):
    """
    Runs k-fold cross-validation on given data.

    Parameters
    ----------
    dataset : list of (data, label) tuples
    feature_extractor : function mapping data -> feature vector (1D numpy array)
    classifier : sklearn-like classifier (implements fit & predict). Defaults to LogisticRegression.
    n_splits : int, number of folds
    random_state : int, seed for shuffling

    Returns
    -------
    accuracies : list of float accuracy scores per fold
    """
    # Build feature matrix X and label vector y
    features = [feature_extractor(data) for data, _ in dataset]
    # Determine max feature length
    max_len = max(f.shape[0] for f in features)
    # Pad features to uniform length
    features_padded = [np.pad(f, (0, max_len - f.shape[0]), mode='constant') for f in features]
    X = np.vstack(features_padded)
    y = np.array([label for _, label in dataset])

    # Default classifier
    if classifier is None:
        classifier = LogisticRegression(random_state=42)

    # Set up cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracies = []

    # Fold loop
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize features
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0)
        X_train_std = (X_train - mu) / sigma
        X_test_std = (X_test - mu) / sigma

        # Train & evaluate
        clf = classifier
        clf.fit(X_train_std, y_train)
        preds = clf.predict(X_test_std)
        accuracies.append(accuracy_score(y_test, preds))

    return accuracies