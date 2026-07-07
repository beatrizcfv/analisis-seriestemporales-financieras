# Feature engineering, training and evaluation of the predictive model.

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """ Assemble the feature matrix and binary target.

    Target: 1 if next day's log return is positive, 0 otherwise.
    """
    features = pd.DataFrame(index=df.index)
    features["Volatility"] = df["Volatility"].squeeze()
    features["Trend_Fourier"] = df["Close_Clean"].squeeze()
    features["Deviation_Fourier"] = df["Close"].squeeze() - df["Close_Clean"].squeeze()
    features["Return_lag1"] = df["Returns"].squeeze().shift(1)
    features["Return_lag2"] = df["Returns"].squeeze().shift(2)
    features["Target"] = np.where(df["Returns"].squeeze().shift(-1) > 0, 1, 0)
    features.dropna(inplace=True)

    return features


def chronological_split(features: pd.DataFrame, train_fraction: float = 0.8):
    """ Split features/target chronologically to avoid data leakage.

    A random train/test split would leak future information into
    training, which is invalid for time series data.
    """
    feature_cols = [c for c in features.columns if c != 'Target']
    X, y = features[feature_cols], features['Target']

    split_index = int(len(X) * train_fraction)

    return (X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:])


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    return {'accuracy': accuracy, 'report': report}


def walk_forward_split(features: pd.DataFrame, n_splits: int = 5, min_train_fraction: float = 0.5):
    """ Generate expanding-window walk-forward train/test splits.

    A single chronological 80/20 split reports accuracy on one test window, which can be lucky or 
    unlucky. Walk-forward validation trains on an expanding history and evaluates on several 
    sequential test windows instead, giving a distribution of out-of-sample accuracies rather than
    a single point estimate.
    """
    features_cols = [c for c in features.columns if c != 'Target']
    X, y = features[features_cols], features['Target']

    n = len(X)
    min_train_size = int(n * min_train_fraction)
    test_size = (n - min_train_size) // n_splits

    for i in range(n_splits):
        train_end = min_train_size + i * test_size
        test_end = train_model + test_size if i < n_splits - 1 else n

        if train_end >= test_end:
            continue
        yield (
            X.iloc[:train_end],
            X.iloc[train_end:test_end],
            y.iloc[:train_end],
            y.iloc[train_end:test_end],
        )


def walk_forward_validate(features: pd.DataFrame, n_splits: int = 5, min_train_fraction: float = 0.5, verbose: bool = True) -> dict:
    """ Run walk-forward validation and return per-fold and aggregate accuracy. """
    fold_accuracies = []

    for fold, (X_train, X_test, y_train, y_test) in enumerate(
        walk_forward_split(features, n_splits=n_splits, min_train_fraction=min_train_fraction), start=1):
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        if verbose:
            print(f"    Fold {fold}: train={len(X_train)}, test={len(X_test)}, accuracy={acc:.4f}")

    return {
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies))
    }
