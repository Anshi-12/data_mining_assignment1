"""Pure Phase 5 clustering engine."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from crispdm_studio.clustering.algorithms import KCandidate, search_kmeans
from crispdm_studio.clustering.charts import ClusteringChart, inertia_chart, pca_chart, silhouette_chart
from crispdm_studio.clustering.evaluation import ClusterProfile, SegmentDescription, build_profiles, describe_segments
from crispdm_studio.clustering.feature_builder import ClusterFeatureSpec, fit_unsupervised_features, select_cluster_features
from crispdm_studio.clustering.selection import choose_candidate
from crispdm_studio.eda.insights import SkippedAnalysis
from crispdm_studio.understanding.profiler import UnderstandingResult
from crispdm_studio.hardening import CAPS, deterministic_cap


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    status: str
    feature_spec: ClusterFeatureSpec
    candidates: tuple[KCandidate, ...] = ()
    chosen_k: int | None = None
    selection_reason: str = ""
    silhouette: float | None = None
    pca_explained_variance: tuple[float, float] | None = None
    profiles: tuple[ClusterProfile, ...] = ()
    segment_descriptions: tuple[SegmentDescription, ...] = ()
    charts: tuple[ClusteringChart, ...] = ()
    skipped_analyses: tuple[SkippedAnalysis, ...] = ()
    resource_notes: tuple[str, ...] = ()

    @property
    def skipped(self) -> bool:
        return self.status == "skipped"


def _skip(spec: ClusterFeatureSpec, reason: str, candidates: tuple[KCandidate, ...] = ()) -> ClusteringResult:
    return ClusteringResult(
        status="skipped",
        feature_spec=spec,
        candidates=candidates,
        selection_reason=reason,
        skipped_analyses=(SkippedAnalysis("Clustering", reason),),
    )


def run_clustering(dataframe: pd.DataFrame, understanding: UnderstandingResult, *, min_rows: int = 8, max_k: int = 8, min_silhouette: float = 0.25) -> ClusteringResult:
    """Cluster structurally-cleaned unlabeled data without touching supervised preprocessing."""
    analysis_df, cap_note = deterministic_cap(dataframe, CAPS.clustering_max_rows)
    spec = select_cluster_features(analysis_df, understanding)
    if not spec.eligible_columns:
        return _skip(spec, "No usable continuous or safely encodable categorical features remain after structural cleaning.")
    if len(analysis_df) < min_rows:
        return _skip(spec, f"Clustering requires at least {min_rows} structurally cleaned rows; only {len(analysis_df)} are available.")

    try:
        _, matrix = fit_unsupervised_features(analysis_df, spec)
    except Exception as exc:
        return _skip(spec, f"Unsupervised preprocessing could not produce a usable feature matrix: {type(exc).__name__}.")
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return _skip(spec, "Unsupervised preprocessing produced zero usable feature dimensions.")
    unique_rows = np.unique(np.round(matrix, 12), axis=0).shape[0]
    if unique_rows < 3:
        return _skip(spec, f"Only {unique_rows} distinct transformed observation patterns remain; there is not enough structure to search for multiple stable clusters.")

    try:
        candidates = search_kmeans(matrix, max_k=min(max_k, unique_rows - 1))
        decision = choose_candidate(candidates, rows=len(analysis_df), min_silhouette=min_silhouette)
    except Exception as exc:
        return _skip(spec, f"Clustering candidate search failed safely ({type(exc).__name__}); no segments are being presented.")
    charts: list[ClusteringChart] = []
    if candidates:
        charts.extend([silhouette_chart(candidates), inertia_chart(candidates)])
    if decision.chosen is None:
        return ClusteringResult(
            status="skipped",
            feature_spec=spec,
            candidates=candidates,
            selection_reason=decision.reason,
            charts=tuple(charts),
            skipped_analyses=(SkippedAnalysis("Clustering", decision.reason),),
            resource_notes=(cap_note,) if cap_note else (),
        )

    chosen = decision.chosen
    labels = np.asarray(chosen.labels)
    pca_var = None
    if matrix.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42)
        projection = pca.fit_transform(matrix)
        explained = tuple(float(x) for x in pca.explained_variance_ratio_[:2])
        pca_var = (explained[0], explained[1])
        charts.append(pca_chart(projection[:, 0], projection[:, 1], labels.tolist(), pca_var))
    elif matrix.shape[1] == 1:
        # PCA cannot create a meaningful 2D plane from one transformed feature.
        pass

    profiles = build_profiles(analysis_df, labels, spec.numeric_columns, spec.categorical_columns)
    descriptions = describe_segments(profiles, analysis_df, spec.numeric_columns, spec.categorical_columns)
    return ClusteringResult(
        status="completed",
        feature_spec=spec,
        candidates=candidates,
        chosen_k=chosen.k,
        selection_reason=decision.reason,
        silhouette=chosen.silhouette,
        pca_explained_variance=pca_var,
        profiles=profiles,
        segment_descriptions=descriptions,
        charts=tuple(charts),
        resource_notes=(cap_note,) if cap_note else (),
    )
