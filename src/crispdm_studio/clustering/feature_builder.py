"""Feature selection and preprocessing for unsupervised clustering.

Unlike Phase 6 supervised modeling, clustering has no held-out label target. It is
therefore appropriate to fit imputers/encoders/scalers on the complete *unlabeled*
structurally-cleaned dataframe used by the clustering analysis. This pipeline is
separate from, and never touches, Phase 3's deferred supervised transformer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from crispdm_studio.understanding.profiler import UnderstandingResult
from crispdm_studio.understanding.type_inference import FeatureType


@dataclass(frozen=True, slots=True)
class ClusterFeatureSpec:
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]

    @property
    def eligible_columns(self) -> tuple[str, ...]:
        return self.numeric_columns + self.categorical_columns

    def build_transformer(self) -> ColumnTransformer:
        transformers = []
        if self.numeric_columns:
            transformers.append((
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                list(self.numeric_columns),
            ))
        if self.categorical_columns:
            transformers.append((
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=0.01,
                        max_categories=40,
                        sparse_output=False,
                    )),
                ]),
                list(self.categorical_columns),
            ))
        return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def select_cluster_features(dataframe: pd.DataFrame, understanding: UnderstandingResult) -> ClusterFeatureSpec:
    profiles = {p.name: p for p in understanding.columns}
    numeric: list[str] = []
    categorical: list[str] = []
    excluded: list[str] = []
    identifiers = set(understanding.identifiers)

    for column in dataframe.columns:
        if column in identifiers:
            excluded.append(column)
            continue
        series = dataframe[column]
        profile = profiles.get(column)
        if series.dropna().nunique() <= 1:
            excluded.append(column)
        elif pd.api.types.is_numeric_dtype(series.dtype):
            numeric.append(column)
        elif profile and profile.feature_type is FeatureType.NUMERIC:
            converted = pd.to_numeric(series, errors="coerce")
            if converted.notna().sum() >= max(3, int(len(series) * 0.5)):
                numeric.append(column)
            else:
                excluded.append(column)
        elif profile and profile.feature_type in {FeatureType.CATEGORICAL, FeatureType.BOOLEAN}:
            # Extremely identifier-like categoricals are not useful clustering features.
            if profile.uniqueness_ratio <= 0.80 or profile.unique <= 40:
                categorical.append(column)
            else:
                excluded.append(column)
        else:
            excluded.append(column)
    return ClusterFeatureSpec(tuple(numeric), tuple(categorical), tuple(excluded))


def fit_unsupervised_features(dataframe: pd.DataFrame, spec: ClusterFeatureSpec):
    """Fit preprocessing on all unlabeled rows; valid specifically for this unsupervised analysis."""
    transformer = spec.build_transformer()
    matrix = transformer.fit_transform(dataframe)
    matrix = np.asarray(matrix, dtype=float)
    return transformer, matrix
