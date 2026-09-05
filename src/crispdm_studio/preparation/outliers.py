"""Training-fold-safe outlier transformer and policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True, slots=True)
class OutlierPolicy:
    method: str = "iqr_clip"
    iqr_multiplier: float = 1.5

    @property
    def rationale(self) -> str:
        return (
            "Numeric values are clipped to training-fold IQR fences rather than deleting rows. "
            "The fences must be learned inside each training fold to avoid leakage."
        )


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip each numeric feature to IQR fences learned during ``fit`` only."""

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.q1_ = np.nanpercentile(values, 25, axis=0)
        self.q3_ = np.nanpercentile(values, 75, axis=0)
        iqr = self.q3_ - self.q1_
        self.lower_bounds_ = self.q1_ - self.multiplier * iqr
        self.upper_bounds_ = self.q3_ + self.multiplier * iqr
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)
