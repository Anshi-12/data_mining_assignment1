"""Unfitted preprocessing specification for Phase 3 model training."""

from __future__ import annotations

from sklearn.preprocessing import RobustScaler


def build_robust_scaler() -> RobustScaler:
    """Return a NEW, UNFITTED RobustScaler.

    Phase 2 must never fit this scaler. In Phase 3, a fresh instance will be fit
    only on the training portion used by a detector. Ground-truth labels are never
    part of the scaler input.
    """
    return RobustScaler()
