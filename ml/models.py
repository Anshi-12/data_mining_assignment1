"""Unsupervised anomaly detectors with a common score-oriented interface."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class Detector(Protocol):
    def fit(self, X: np.ndarray) -> "Detector": ...
    def anomaly_score(self, X: np.ndarray) -> np.ndarray: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class TimedScores:
    fit_seconds: float
    inference_ms_per_row: float
    scores: np.ndarray
    predictions: np.ndarray


class SklearnDetector:
    """Adapter that exposes higher-is-more-anomalous scores."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X: np.ndarray) -> "SklearnDetector":
        self.estimator.fit(X)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return -np.asarray(self.estimator.decision_function(X), dtype=float).reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.estimator.predict(X)).reshape(-1)
        return (raw == -1).astype(int)


class NumpyAutoencoder:
    """Small fully-connected autoencoder implemented only with NumPy.

    Architecture: input -> hidden(tanh) -> bottleneck(tanh) -> hidden(tanh) -> input.
    Training minimizes mean squared reconstruction error with mini-batch Adam.
    No labels are accepted by this class.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 8,
        bottleneck_dim: int = 3,
        epochs: int = 45,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        contamination: float = 0.035,
        random_state: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.contamination = contamination
        self.random_state = random_state
        self.params: dict[str, np.ndarray] = {}
        self.threshold_: float | None = None

    def _init_params(self, rng: np.random.RandomState) -> None:
        dims = [self.input_dim, self.hidden_dim, self.bottleneck_dim, self.hidden_dim, self.input_dim]
        for i in range(4):
            fan_in, fan_out = dims[i], dims[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.params[f"W{i+1}"] = rng.normal(0.0, scale, size=(fan_in, fan_out))
            self.params[f"b{i+1}"] = np.zeros((1, fan_out), dtype=float)

    @staticmethod
    def _tanh_grad(a: np.ndarray) -> np.ndarray:
        return 1.0 - a * a

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        z1 = X @ self.params["W1"] + self.params["b1"]
        a1 = np.tanh(z1)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = np.tanh(z2)
        z3 = a2 @ self.params["W3"] + self.params["b3"]
        a3 = np.tanh(z3)
        out = a3 @ self.params["W4"] + self.params["b4"]
        return out, (X, a1, a2, a3)

    def fit(self, X: np.ndarray) -> "NumpyAutoencoder":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError("Autoencoder received an incompatible feature matrix.")
        rng = np.random.RandomState(self.random_state)
        self._init_params(rng)
        m = {name: np.zeros_like(value) for name, value in self.params.items()}
        v = {name: np.zeros_like(value) for name, value in self.params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), self.batch_size):
                batch = X[order[start : start + self.batch_size]]
                out, (x0, a1, a2, a3) = self._forward(batch)
                n = max(len(batch), 1)
                d_out = 2.0 * (out - x0) / (n * self.input_dim)

                grads: dict[str, np.ndarray] = {}
                grads["W4"] = a3.T @ d_out
                grads["b4"] = d_out.sum(axis=0, keepdims=True)
                d3 = (d_out @ self.params["W4"].T) * self._tanh_grad(a3)
                grads["W3"] = a2.T @ d3
                grads["b3"] = d3.sum(axis=0, keepdims=True)
                d2 = (d3 @ self.params["W3"].T) * self._tanh_grad(a2)
                grads["W2"] = a1.T @ d2
                grads["b2"] = d2.sum(axis=0, keepdims=True)
                d1 = (d2 @ self.params["W2"].T) * self._tanh_grad(a1)
                grads["W1"] = x0.T @ d1
                grads["b1"] = d1.sum(axis=0, keepdims=True)

                step += 1
                for name in self.params:
                    m[name] = beta1 * m[name] + (1.0 - beta1) * grads[name]
                    v[name] = beta2 * v[name] + (1.0 - beta2) * (grads[name] ** 2)
                    m_hat = m[name] / (1.0 - beta1**step)
                    v_hat = v[name] / (1.0 - beta2**step)
                    self.params[name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        train_scores = self.anomaly_score(X)
        self.threshold_ = float(np.quantile(train_scores, 1.0 - self.contamination))
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not self.params:
            raise RuntimeError("Autoencoder must be fitted before scoring.")
        X = np.asarray(X, dtype=float)
        out, _ = self._forward(X)
        return np.mean((out - X) ** 2, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Autoencoder must be fitted before prediction.")
        return (self.anomaly_score(X) >= self.threshold_).astype(int)


def build_detectors(*, contamination: float, random_state: int, input_dim: int) -> dict[str, Detector]:
    return {
        "Isolation Forest": SklearnDetector(
            IsolationForest(
                n_estimators=160,
                contamination=contamination,
                max_samples="auto",
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        "Local Outlier Factor": SklearnDetector(
            LocalOutlierFactor(
                n_neighbors=35,
                contamination=contamination,
                novelty=True,
                n_jobs=-1,
            )
        ),
        "One-Class SVM": SklearnDetector(
            OneClassSVM(kernel="rbf", gamma="scale", nu=contamination)
        ),
        "Robust Mahalanobis": SklearnDetector(
            EllipticEnvelope(contamination=contamination, random_state=random_state, support_fraction=0.8)
        ),
        "NumPy Autoencoder": NumpyAutoencoder(
            input_dim=input_dim,
            contamination=contamination,
            random_state=random_state,
        ),
    }


def fit_and_time(detector: Detector, X_train: np.ndarray, X_validation: np.ndarray) -> TimedScores:
    started = perf_counter()
    detector.fit(X_train)
    fit_seconds = perf_counter() - started

    started = perf_counter()
    scores = detector.anomaly_score(X_validation)
    predictions = detector.predict(X_validation)
    elapsed = perf_counter() - started
    inference_ms_per_row = (elapsed / max(len(X_validation), 1)) * 1000.0
    return TimedScores(
        fit_seconds=float(fit_seconds),
        inference_ms_per_row=float(inference_ms_per_row),
        scores=np.asarray(scores, dtype=float),
        predictions=np.asarray(predictions, dtype=int),
    )
