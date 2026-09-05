"""Baseline estimators."""
from sklearn.dummy import DummyClassifier, DummyRegressor


def classification_baseline() -> DummyClassifier:
    return DummyClassifier(strategy="prior")


def regression_baseline() -> DummyRegressor:
    return DummyRegressor(strategy="mean")
