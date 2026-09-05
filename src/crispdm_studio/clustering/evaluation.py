"""Cluster profiles and data-derived segment descriptions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ClusterProfile:
    cluster: int
    size: int
    share: float
    numeric_means: dict[str, float] = field(default_factory=dict)
    numeric_medians: dict[str, float] = field(default_factory=dict)
    categorical_modes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SegmentDescription:
    cluster: int
    title: str
    text: str
    metrics: dict[str, Any] = field(default_factory=dict)


def build_profiles(dataframe: pd.DataFrame, labels: np.ndarray, numeric: tuple[str, ...], categorical: tuple[str, ...]) -> tuple[ClusterProfile, ...]:
    work = dataframe.copy()
    work["__cluster__"] = labels
    profiles: list[ClusterProfile] = []
    for cluster in sorted(work["__cluster__"].unique()):
        part = work[work["__cluster__"] == cluster]
        means = {c: float(pd.to_numeric(part[c], errors="coerce").mean()) for c in numeric if pd.to_numeric(part[c], errors="coerce").notna().any()}
        medians = {c: float(pd.to_numeric(part[c], errors="coerce").median()) for c in numeric if pd.to_numeric(part[c], errors="coerce").notna().any()}
        modes: dict[str, str] = {}
        for c in categorical:
            mode = part[c].dropna().astype(str).mode()
            if not mode.empty:
                modes[c] = str(mode.iloc[0])
        profiles.append(ClusterProfile(int(cluster), len(part), len(part) / len(work), means, medians, modes))
    return tuple(profiles)


def describe_segments(profiles: tuple[ClusterProfile, ...], dataframe: pd.DataFrame, numeric: tuple[str, ...], categorical: tuple[str, ...]) -> tuple[SegmentDescription, ...]:
    overall_medians = {c: float(pd.to_numeric(dataframe[c], errors="coerce").median()) for c in numeric}
    overall_std = {c: float(pd.to_numeric(dataframe[c], errors="coerce").std()) for c in numeric}
    descriptions: list[SegmentDescription] = []
    for p in profiles:
        traits: list[str] = []
        metrics: dict[str, Any] = {"size": p.size, "share": p.share}
        scored: list[tuple[float, str, float, float]] = []
        for c, med in p.numeric_medians.items():
            std = overall_std.get(c, 0.0)
            base = overall_medians.get(c, med)
            if std and np.isfinite(std):
                scored.append((abs(med - base) / std, c, med, base))
        for score, c, med, base in sorted(scored, reverse=True)[:3]:
            if score >= 0.25:
                direction = "above" if med > base else "below"
                traits.append(f"{c} median {med:,.4g}, {direction} overall median {base:,.4g}")
                metrics[f"{c}_median"] = med
                metrics[f"{c}_overall_median"] = base
        for c, mode in list(p.categorical_modes.items())[:2]:
            share = float((dataframe.loc[np.ones(len(dataframe), dtype=bool), c].astype(str) == mode).mean())
            traits.append(f"most common {c} is '{mode}'")
            metrics[f"{c}_mode"] = mode
            metrics[f"{c}_overall_mode_share"] = share
        if not traits:
            traits.append("no single profiled feature differs strongly from the overall dataset")
        text = f"Cluster {p.cluster} contains {p.size:,} rows ({p.share:.1%}). " + "; ".join(traits) + "."
        descriptions.append(SegmentDescription(p.cluster, f"Segment {p.cluster}", text, metrics))
    return tuple(descriptions)
