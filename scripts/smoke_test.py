"""End-to-end release smoke check for the complete CRISP-DM pipeline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispdm_studio.config import CONFIG
from crispdm_studio.clustering import run_clustering
from crispdm_studio.eda import run_eda
from crispdm_studio.evaluation import evaluate_modeling
from crispdm_studio.ingestion import ingest_csv
from crispdm_studio.modeling import run_modeling
from crispdm_studio.preparation import prepare_dataset
from crispdm_studio.reporting import build_report
from crispdm_studio.reporting.chart_export import chart_to_html
from crispdm_studio.understanding import profile_dataset


def main() -> None:
    rows = ["customer_id,amount,score,segment,label,constant"]
    for i in range(30):
        rows.append(
            f"C{i+1},{10 + i * 3 if i != 7 else ''},{2 + i * 0.7:.1f},{'A' if i % 3 else 'B'},{'yes' if i % 2 == 0 else 'no'},x"
        )
    rows.append(rows[-1])  # exact duplicate exercises structural deduplication
    data = ("\n".join(rows) + "\n").encode()
    dataset = ingest_csv(filename="smoke.csv", mime_type="text/csv", data=data, config=CONFIG)
    understanding = profile_dataset(dataset)
    preparation = prepare_dataset(dataset.dataframe, understanding)
    transformer = preparation.preprocessing_spec.build_transformer()
    eda = run_eda(preparation.structural_dataframe, understanding)
    clustering = run_clustering(preparation.structural_dataframe, understanding, min_rows=5)

    print(f"OK ingestion: {dataset.overview.rows} rows x {dataset.overview.columns} columns")
    print(f"Understanding: {len(understanding.columns)} column profiles")
    print(
        "Preparation: "
        f"{preparation.summary.rows_after_structural} rows x {preparation.summary.columns_after_structural} columns; "
        f"{len(preparation.applied_decisions)} applied, {len(preparation.deferred_decisions)} deferred"
    )
    print(f"Deferred transformer is unfitted: {not hasattr(transformer, 'transformers_')}")
    print(
        f"EDA: {len(eda.findings)} findings, {len(eda.charts)} reusable charts, "
        f"{len(eda.skipped_analyses)} explicitly skipped analyses"
    )
    if eda.charts:
        print(f"First chart HTML export works: {'plotly' in chart_to_html(eda.charts[0]).lower()}")
    print(
        "Clustering: "
        + (
            f"k={clustering.chosen_k}, silhouette={clustering.silhouette:.3f}, {len(clustering.segment_descriptions)} segments"
            if not clustering.skipped
            else f"skipped — {clustering.selection_reason}"
        )
    )
    if clustering.charts:
        print(f"Clustering chart HTML export works: {'plotly' in chart_to_html(clustering.charts[0]).lower()}")
    print(f"Deferred transformer still unfitted after clustering: {not hasattr(transformer, 'transformers_')}")
    modeling = run_modeling(preparation.structural_dataframe, understanding, preparation, "label")
    print(
        "Modeling: "
        + (
            f"{modeling.task_type.value}, {len(modeling.model_results)} stored fitted pipelines, CV folds={modeling.cv_folds}"
            if not modeling.skipped
            else f"skipped — {modeling.skip_reason}"
        )
    )
    print(f"Deferred transformer is unfitted after modeling: {not hasattr(transformer, 'transformers_')}")
    evaluation = evaluate_modeling(modeling, eda, clustering, understanding)
    print(
        "Evaluation: "
        + (
            f"selected={evaluation.selected_model_name}, importance={len(evaluation.permutation_importance)}, charts={len(evaluation.charts)}"
            if not evaluation.skipped
            else f"no defensible model — {evaluation.skip_reason}"
        )
    )
    print(f"Deferred transformer is unfitted after evaluation: {not hasattr(transformer, 'transformers_')}")
    report = build_report(understanding, preparation, eda, clustering, modeling, evaluation)
    print(f"Report HTML export succeeded: {report.html.startswith('<!doctype html>')}; bytes={len(report.html_bytes):,}")
    print(
        "Report PDF export: "
        f"fidelity={report.pdf.fidelity}; bytes={len(report.pdf_bytes):,}; "
        f"static_charts={report.pdf.static_charts_rendered}; placeholders={report.pdf.static_charts_placeholder}; "
        f"weasyprint_succeeded={report.pdf.weasyprint_succeeded}"
    )
    if "--write-report" in sys.argv:
        (ROOT / "smoke-report.html").write_bytes(report.html_bytes)
        (ROOT / "smoke-report.pdf").write_bytes(report.pdf_bytes)
        print("Smoke report files written: smoke-report.html, smoke-report.pdf")
    if report.pdf.notices:
        for notice in report.pdf.notices:
            print(f"Report PDF notice [{notice.component}]: {notice.message}")
            if notice.install_hint:
                print(f"Report PDF install hint: {notice.install_hint}")
    print(f"Deferred transformer is unfitted after reporting: {not hasattr(transformer, 'transformers_')}")


if __name__ == "__main__":
    main()
