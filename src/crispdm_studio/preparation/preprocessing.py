"""Phase 3 preparation orchestration and fold-safe preprocessing specification."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from crispdm_studio.preparation.cleaning import (
    DecisionStage,
    PreparationDecision,
    apply_structural_cleaning,
)
from crispdm_studio.preparation.encoding import EncodingStrategy
from crispdm_studio.preparation.missing import MissingValueStrategy, describe_missing_strategy
from crispdm_studio.preparation.outliers import IQRClipper, OutlierPolicy
from crispdm_studio.understanding.profiler import UnderstandingResult
from crispdm_studio.understanding.type_inference import FeatureType


@dataclass(frozen=True, slots=True)
class PreparationSummary:
    rows_before: int
    rows_after_structural: int
    columns_before: int
    columns_after_structural: int
    rows_removed: int
    columns_removed: int
    applied_decisions: int
    deferred_decisions: int


@dataclass(frozen=True, slots=True)
class PreprocessingSpec:
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    high_cardinality_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    missing: MissingValueStrategy
    encoding: EncodingStrategy
    outliers: OutlierPolicy
    scaling_method: str = "standard_scaler"

    def build_transformer(self) -> ColumnTransformer:
        """Build an unfitted transformer.

        CRITICAL: callers must fit this object only on a training split/fold. Phase 3
        intentionally returns it unfitted and never calls fit on the full dataset.
        """
        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if self.numeric_columns:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy=self.missing.numeric_strategy)),
                    ("outlier_clip", IQRClipper(multiplier=self.outliers.iqr_multiplier)),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("numeric", numeric_pipeline, list(self.numeric_columns)))

        if self.categorical_columns:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy=self.missing.categorical_strategy)),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown=self.encoding.handle_unknown,
                            min_frequency=self.encoding.min_frequency,
                            max_categories=self.encoding.max_categories,
                            sparse_output=False,
                        ),
                    ),
                ]
            )
            transformers.append(("categorical", categorical_pipeline, list(self.categorical_columns)))

        return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


@dataclass(frozen=True, slots=True)
class PreparationResult:
    structural_dataframe: pd.DataFrame
    decisions: tuple[PreparationDecision, ...]
    summary: PreparationSummary
    preprocessing_spec: PreprocessingSpec

    @property
    def applied_decisions(self) -> tuple[PreparationDecision, ...]:
        return tuple(d for d in self.decisions if d.stage is DecisionStage.APPLIED_NOW)

    @property
    def deferred_decisions(self) -> tuple[PreparationDecision, ...]:
        return tuple(d for d in self.decisions if d.stage is DecisionStage.DEFERRED)


def _infer_modeling_columns(
    frame: pd.DataFrame, understanding: UnderstandingResult
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    profiles = {profile.name: profile for profile in understanding.columns}
    numeric: list[str] = []
    categorical: list[str] = []
    high_cardinality: list[str] = []
    excluded: list[str] = []

    for column in frame.columns:
        profile = profiles.get(column)
        if pd.api.types.is_numeric_dtype(frame[column].dtype):
            numeric.append(column)
            continue
        if profile is None:
            excluded.append(column)
            continue
        if profile.feature_type in {FeatureType.CATEGORICAL, FeatureType.BOOLEAN}:
            categorical.append(column)
            if profile.unique > 20 or profile.uniqueness_ratio > 0.20:
                high_cardinality.append(column)
        elif profile.feature_type is FeatureType.NUMERIC:
            numeric.append(column)
        else:
            # Generic Phase 3 does not invent text/date feature engineering. These
            # columns remain in the structural dataframe but are excluded from the
            # baseline modeling transformer until a later explicit strategy exists.
            excluded.append(column)

    return tuple(numeric), tuple(categorical), tuple(high_cardinality), tuple(excluded)


def prepare_dataset(dataframe: pd.DataFrame, understanding: UnderstandingResult) -> PreparationResult:
    """Prepare structural data and return an *unfitted* learned-transform spec."""
    rows_before, columns_before = dataframe.shape
    structural = apply_structural_cleaning(dataframe, understanding)
    clean_df = structural.dataframe

    numeric, categorical, high_cardinality, excluded = _infer_modeling_columns(clean_df, understanding)
    missing = MissingValueStrategy()
    encoding = EncodingStrategy()
    outliers = OutlierPolicy()

    decisions = list(structural.decisions)

    decisions.append(
        PreparationDecision(
            action="Defer missing-value imputation",
            stage=DecisionStage.DEFERRED,
            reason=(
                describe_missing_strategy(clean_df, numeric, categorical)
                + " Medians/modes are learned parameters, so they must be fit on training folds only."
            ),
            rows_affected=int(clean_df[list(numeric + categorical)].isna().any(axis=1).sum())
            if (numeric or categorical)
            else 0,
            columns_affected=sum(int(clean_df[column].isna().any()) for column in numeric + categorical),
            columns=tuple(column for column in numeric + categorical if clean_df[column].isna().any()),
        )
    )

    decisions.append(
        PreparationDecision(
            action="Defer categorical encoding",
            stage=DecisionStage.DEFERRED,
            reason=encoding.rationale + " Category vocabularies/frequencies must be learned from training folds only.",
            columns_affected=len(categorical),
            columns=categorical,
            details={
                "method": encoding.method,
                "max_categories": encoding.max_categories,
                "min_frequency": encoding.min_frequency,
            },
        )
    )

    decisions.append(
        PreparationDecision(
            action="Handle high-cardinality categoricals inside encoder",
            stage=DecisionStage.DEFERRED,
            reason=(
                "High-cardinality categoricals are retained but OneHotEncoder will cap/group levels based on training-fold "
                "frequencies, preventing a vocabulary learned from validation/test data."
            ),
            columns_affected=len(high_cardinality),
            columns=high_cardinality,
        )
    )

    decisions.append(
        PreparationDecision(
            action="Defer numeric outlier clipping",
            stage=DecisionStage.DEFERRED,
            reason=outliers.rationale,
            columns_affected=len(numeric),
            columns=numeric,
            details={"method": outliers.method, "iqr_multiplier": outliers.iqr_multiplier},
        )
    )

    decisions.append(
        PreparationDecision(
            action="Defer numeric scaling",
            stage=DecisionStage.DEFERRED,
            reason=(
                "StandardScaler means and standard deviations are learned parameters. Scaling is therefore represented "
                "inside the sklearn pipeline and will be fit on training folds only."
            ),
            columns_affected=len(numeric),
            columns=numeric,
            details={"method": "standard_scaler"},
        )
    )

    if excluded:
        decisions.append(
            PreparationDecision(
                action="Exclude unsupported generic feature types from baseline transformer",
                stage=DecisionStage.DEFERRED,
                reason=(
                    "Datetime, free-text, and empty columns remain available in the structurally cleaned dataframe, but "
                    "Phase 3 does not invent dataset-specific feature engineering for them. The baseline transformer drops "
                    "them until an explicit modeling strategy is chosen."
                ),
                columns_affected=len(excluded),
                columns=excluded,
            )
        )

    spec = PreprocessingSpec(
        numeric_columns=numeric,
        categorical_columns=categorical,
        high_cardinality_columns=high_cardinality,
        excluded_columns=excluded,
        missing=missing,
        encoding=encoding,
        outliers=outliers,
    )
    applied = sum(decision.stage is DecisionStage.APPLIED_NOW for decision in decisions)
    deferred = len(decisions) - applied
    summary = PreparationSummary(
        rows_before=rows_before,
        rows_after_structural=clean_df.shape[0],
        columns_before=columns_before,
        columns_after_structural=clean_df.shape[1],
        rows_removed=rows_before - clean_df.shape[0],
        columns_removed=columns_before - clean_df.shape[1],
        applied_decisions=applied,
        deferred_decisions=deferred,
    )
    return PreparationResult(
        structural_dataframe=clean_df,
        decisions=tuple(decisions),
        summary=summary,
        preprocessing_spec=spec,
    )
