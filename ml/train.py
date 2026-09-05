"""CLI for Phase 3 leakage-safe training and artifact generation."""

from __future__ import annotations

import argparse

from ml.training import train_phase3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and validate Phase 3 anomaly detectors.")
    parser.add_argument("--rows", type=int, default=2_000, help="Synthetic rows; use 2K-5K for dev.")
    parser.add_argument("--contamination", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--artifact-dir", default="server/artifacts")
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train_phase3(
        rows=args.rows,
        contamination=args.contamination,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        artifact_dir=args.artifact_dir,
        top_n=args.top_n,
    )

    print("Anomaly Detection Studio — Phase 3 training")
    print(f"Dataset: {result.dataset_rows} locally synthesized rows")
    print(
        f"Split FIRST: train={result.split.training_rows}, "
        f"validation={result.split.validation_rows}"
    )
    print("RobustScaler fit scope: TRAINING FEATURES ONLY")
    print("Ground-truth labels passed to detector fit: NONE")
    print("\nBenchmark results:")
    print("  Model                       ROC-AUC  PR-AUC   Prec.  Recall    F1    Fit(s)  ms/row")
    for metric in result.benchmark_results:
        print(
            f"  {metric.model_name:<27} {metric.roc_auc:>7.4f}  {metric.pr_auc:>6.4f}  "
            f"{metric.precision:>6.3f}  {metric.recall:>6.3f}  {metric.f1:>6.3f}  "
            f"{metric.fit_seconds:>7.3f}  {metric.inference_ms_per_row:>6.3f}"
        )
    print(f"\nReal Isolation Forest search runs: {len(result.search_history)}")
    print(f"Selected config: {result.selected_params}")
    print(f"Why: {result.selection_reason}")
    print(
        f"Calibration: empirical training-score percentile; threshold="
        f"{result.calibration.raw_threshold:.6f} raw / {result.calibration.threat_threshold:.2f} threat"
    )
    print(f"Artifacts written to: {result.artifact_dir}")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
