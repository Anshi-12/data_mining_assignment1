"""Exploratory Data Analysis public API."""

from crispdm_studio.eda.statistics import CategoricalDistribution, NumericDistribution
from crispdm_studio.eda.insights import EDAFinding, FindingKind, SkippedAnalysis
from crispdm_studio.eda.charts import EDAChart
from crispdm_studio.eda.engine import EDAResult, run_eda

__all__ = [
    "CategoricalDistribution",
    "EDAChart",
    "EDAFinding",
    "EDAResult",
    "FindingKind",
    "NumericDistribution",
    "SkippedAnalysis",
    "run_eda",
]
