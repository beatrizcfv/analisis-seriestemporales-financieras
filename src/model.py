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