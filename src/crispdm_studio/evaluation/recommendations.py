"""Data-derived Phase 7 recommendation narrative."""
from __future__ import annotations

from dataclasses import dataclass

from crispdm_studio.clustering.engine import ClusteringResult
from crispdm_studio.evaluation.metrics import ClassImbalanceDiagnostic, SelectionScore
from crispdm_studio.modeling.comparison import ModelResult, ModelingResult
from crispdm_studio.understanding.profiler import UnderstandingResult


@dataclass(frozen=True, slots=True)
class Recommendation:
    what_we_learned: str
    reliability: str
    reasonable_action: str
    limitations: str
    next_steps: str

    @property
    def full_text(self) -> str:
        return "\n\n".join((
            f"What we learned — {self.what_we_learned}",
            f"How reliable it appears — {self.reliability}",
            f"What action is reasonable — {self.reasonable_action}",
            f"What limitations matter — {self.limitations}",
            f"What to do next — {self.next_steps}",
        ))


def build_recommendation(
    modeling: ModelingResult,
    selected: ModelResult | None,
    selection_scores: tuple[SelectionScore, ...],
    imbalance: ClassImbalanceDiagnostic,
    understanding: UnderstandingResult,
    clustering: ClusteringResult,
    selection_reason: str,
) -> Recommendation:
    quality = [w.message for w in understanding.quality_warnings]
    quality_text = " ".join(quality[:4]) if quality else "No Phase 2 data-quality warning crossed the configured thresholds."
    holdout_text = f"The held-out test set contains {modeling.split.test_rows} rows." if modeling.split is not None else "No holdout split was available."
    if modeling.split is not None and modeling.split.test_rows < 30:
        holdout_text += " This is a small holdout, so its point estimates have substantial sampling uncertainty."
    cv_text = f"{modeling.cv_folds}-fold CV was available." if modeling.cv_folds else "Cross-validation was not available at a defensible fold count."
    cluster_text = (
        f"Clustering found {clustering.chosen_k} segments with silhouette {clustering.silhouette:.3f}." if not clustering.skipped and clustering.chosen_k is not None and clustering.silhouette is not None
        else f"Clustering was skipped or rejected: {clustering.selection_reason}"
    )
    if selected is None:
        return Recommendation(
            what_we_learned=f"The supervised comparison did not identify a candidate that defensibly improves on the Dummy baseline. {selection_reason}",
            reliability=f"The evidence is insufficient for a model recommendation. {holdout_text} {cv_text} {imbalance.message if imbalance.applicable else ''}".strip(),
            reasonable_action="Do not operationalize a predictive model from this run; use the descriptive EDA and validated data-quality findings instead.",
            limitations=f"{quality_text} {cluster_text}",
            next_steps="Collect more representative observations, improve target definition and predictor timing, resolve major data-quality issues, then rerun the same leakage-safe modeling workflow.",
        )
    primary = modeling.primary_metric or "primary metric"
    test_value = selected.test_metrics.get(primary)
    cv = next((m for m in selected.cv_metrics if m.name == primary), None)
    score = next((s for s in selection_scores if s.model_name == selected.name), None)
    learned = f"{selected.name} is the strongest defensible candidate: held-out {primary.replace('_', ' ')}={test_value:.3f}"
    if score:
        learned += f", improving on baseline by {score.test_improvement:+.3f}."
    else:
        learned += "."
    reliability_parts = [holdout_text, cv_text]
    if cv:
        reliability_parts.append(f"Its CV {primary.replace('_', ' ')} was {cv.mean:.3f} ± {cv.std:.3f}.")
    reliability_parts.append(selected.overfitting_reason if selected.overfitting_flag and selected.overfitting_reason else "No configured train/test divergence threshold was exceeded for the selected model.")
    if imbalance.applicable:
        reliability_parts.append(imbalance.message)
    return Recommendation(
        what_we_learned=learned,
        reliability=" ".join(reliability_parts),
        reasonable_action="Use this result as a validated candidate for further business/domain review and prospective testing, not as an automatic production-deployment decision.",
        limitations=f"{quality_text} {cluster_text}",
        next_steps="Review the highest permutation-importance features for plausibility and timing, inspect the held-out error diagnostic, validate performance on fresh data, and only then consider deployment thresholds or business actions.",
    )
