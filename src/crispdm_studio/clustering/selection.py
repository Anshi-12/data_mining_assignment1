"""Cluster-count selection and meaningfulness gate."""
from __future__ import annotations

from dataclasses import dataclass

from crispdm_studio.clustering.algorithms import KCandidate


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    chosen: KCandidate | None
    reason: str


def choose_candidate(candidates: tuple[KCandidate, ...], *, rows: int, min_silhouette: float = 0.25) -> SelectionDecision:
    if not candidates:
        return SelectionDecision(None, "No valid K-Means candidate could be evaluated with silhouette scoring.")
    best = max(candidates, key=lambda c: (c.silhouette, -c.k))
    min_allowed = max(2, int(round(rows * 0.03)))
    if best.silhouette < min_silhouette:
        return SelectionDecision(
            None,
            f"Best silhouette was {best.silhouette:.3f}, below the meaningfulness threshold {min_silhouette:.2f}; separation is too weak to present as segments.",
        )
    if best.min_cluster_size < min_allowed:
        return SelectionDecision(
            None,
            f"Best k={best.k} created a smallest cluster of {best.min_cluster_size} rows; at least {min_allowed} are required to avoid presenting tiny fragments as segments.",
        )
    return SelectionDecision(
        best,
        f"k={best.k} achieved the highest silhouette ({best.silhouette:.3f}) among evaluated candidates while passing minimum cluster-size and separation guardrails.",
    )
