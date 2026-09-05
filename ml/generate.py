"""CLI: generate and validate deterministic locally synthesized telemetry."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.data_loader import DEFAULT_CONTAMINATION, DEFAULT_ROWS, DEFAULT_SEED, generate_anomaly_dataset
from ml.preprocessing import build_robust_scaler
from ml.validation import validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for the combined inspection frame (features + evaluation labels).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = generate_anomaly_dataset(
        n_samples=args.rows,
        contamination=args.contamination,
        random_state=args.seed,
    )
    validation = validate_dataset(dataset)
    scaler = build_robust_scaler()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dataset.combined_frame().to_csv(args.output, index=False)

    print("Anomaly Detection Studio — Phase 2 dataset validation")
    print("Source: locally synthesized telemetry (no dataset download)")
    print(f"Seed: {dataset.seed}")
    print(f"Feature matrix: {dataset.features.shape[0]} rows x {dataset.features.shape[1]} features")
    print(
        f"Ground truth: {dataset.statistics.anomaly_count} anomalies / "
        f"{dataset.statistics.rows} rows ({dataset.statistics.contamination:.2%})"
    )
    print("Anomaly archetypes:")
    for name, count in dataset.statistics.archetype_counts.items():
        share = dataset.statistics.archetype_shares_of_anomalies[name]
        print(f"  - {name}: {count} ({share:.2%} of anomalies)")
    print("Label columns passed to feature matrix: NONE")
    if args.output is not None:
        print(f"CSV written: {args.output}")
    print(f"RobustScaler fitted in Phase 2: {hasattr(scaler, 'center_')}")
    print("Validation checks:")
    for check in validation.checks:
        print(f"  PASS — {check}")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
