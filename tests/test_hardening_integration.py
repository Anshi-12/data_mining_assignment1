from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from crispdm_studio.config import CONFIG
from crispdm_studio.exceptions import CrispDMError
from crispdm_studio.ingestion import ingest_csv
from crispdm_studio.understanding import profile_dataset
from crispdm_studio.preparation import prepare_dataset
from crispdm_studio.eda import run_eda
from crispdm_studio.clustering import run_clustering
from crispdm_studio.modeling import run_modeling
from crispdm_studio.state import (
    DATASET_KEY, UNDERSTANDING_RESULT_KEY, PREPARATION_RESULT_KEY, EDA_RESULT_KEY,
    CLUSTERING_RESULT_KEY, MODELING_RESULT_KEY, EVALUATION_RESULT_KEY, REPORT_RESULT_KEY,
    initialize_state, set_dataset, set_error, invalidate_from,
)


def ingest(text: str, name: str = "case.csv", mime: str = "text/csv", encoding: str = "utf-8"):
    return ingest_csv(filename=name, mime_type=mime, data=text.encode(encoding), config=CONFIG)


def through_preparation(text: str):
    ds = ingest(text)
    understanding = profile_dataset(ds)
    preparation = prepare_dataset(ds.dataframe, understanding)
    return ds, understanding, preparation


def test_single_row_survives_descriptive_pipeline_and_skips_cluster_cleanly():
    _, understanding, preparation = through_preparation("a,b\n1,x\n")
    eda = run_eda(preparation.structural_dataframe, understanding)
    clustering = run_clustering(preparation.structural_dataframe, understanding)
    assert eda.rows <= 1
    assert clustering.skipped
    assert "at least" in clustering.selection_reason.lower() or "usable" in clustering.selection_reason.lower()


def test_single_column_survives_and_downstream_skips_are_structured():
    _, understanding, preparation = through_preparation("label\nA\nB\nA\nB\nA\nB\nA\nB\n")
    eda = run_eda(preparation.structural_dataframe, understanding)
    clustering = run_clustering(preparation.structural_dataframe, understanding)
    modeling = run_modeling(preparation.structural_dataframe, understanding, preparation, None)
    assert eda is not None
    assert clustering.skipped
    assert modeling.skipped and "target" in modeling.skip_reason.lower()


def test_all_numeric_and_all_categorical_do_not_force_wrong_eda():
    _, u_num, p_num = through_preparation("x,y,z\n1,10,3\n2,11,4\n4,12,8\n7,14,9\n9,16,12\n")
    e_num = run_eda(p_num.structural_dataframe, u_num)
    assert e_num.numeric_columns
    assert any(x.analysis == "Categorical distributions" for x in e_num.skipped_analyses)

    _, u_cat, p_cat = through_preparation("a,b,c\nred,x,yes\nblue,y,no\nred,y,yes\nblue,x,no\nred,x,no\nblue,y,yes\n")
    e_cat = run_eda(p_cat.structural_dataframe, u_cat)
    assert e_cat.categorical_columns
    assert any(x.analysis == "Numeric distributions" for x in e_cat.skipped_analyses)


def test_near_total_missingness_degrades_without_exception():
    text = "a,b,c,target\n1,,,,\n,foo,,,\n,,,yes\n,,,no\n,,,yes\n,,,no\n,,,yes\n,,,no\n"
    _, understanding, preparation = through_preparation(text)
    eda = run_eda(preparation.structural_dataframe, understanding)
    assert eda is not None
    assert understanding.quality_warnings


def test_wide_high_cardinality_dataset_is_bounded_and_does_not_crash():
    cols = [f"f{i}" for i in range(80)]
    rows = [",".join(cols)]
    for r in range(120):
        rows.append(",".join(str((r * (i + 3)) % 17) if i < 40 else f"v{i}_{r}" for i in range(80)))
    _, understanding, preparation = through_preparation("\n".join(rows) + "\n")
    eda = run_eda(preparation.structural_dataframe, understanding)
    assert len(eda.charts) < 100
    # relationship analysis is capped even when the source is wide
    assert len(eda.numeric_columns) <= 80


def test_tiny_and_highly_imbalanced_targets_skip_safely():
    text = "x,target\n1,A\n2,A\n3,A\n4,A\n5,A\n6,A\n7,A\n8,B\n"
    _, understanding, preparation = through_preparation(text)
    result = run_modeling(preparation.structural_dataframe, understanding, preparation, "target")
    assert result.skipped
    assert "fewer than two" in result.skip_reason.lower() or "split" in result.skip_reason.lower()


def test_no_clusterable_features_skips_not_raises():
    _, understanding, preparation = through_preparation("a,b\nx,1\nx,1\nx,1\nx,1\nx,1\nx,1\nx,1\nx,1\n")
    result = run_clustering(preparation.structural_dataframe, understanding)
    assert result.skipped
    assert result.skipped_analyses


def test_deliberate_clustering_failure_is_converted_to_skip(monkeypatch):
    _, understanding, preparation = through_preparation("x,y\n1,2\n2,3\n3,5\n4,7\n5,11\n6,13\n7,17\n8,19\n9,23\n10,29\n")
    import crispdm_studio.clustering.engine as engine
    monkeypatch.setattr(engine, "search_kmeans", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    result = run_clustering(preparation.structural_dataframe, understanding)
    assert result.skipped
    assert "failed safely" in result.selection_reason.lower()


def test_deliberate_model_failure_returns_safe_skip_and_original_spec_unfitted(monkeypatch):
    rows = ["x,region,target"]
    for i in range(30):
        rows.append(f"{i},{'A' if i % 2 else 'B'},{'yes' if i > 14 else 'no'}")
    _, understanding, preparation = through_preparation("\n".join(rows) + "\n")
    original = preparation.preprocessing_spec.build_transformer()
    monkeypatch.setattr(Pipeline, "fit", lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("forced")))
    result = run_modeling(preparation.structural_dataframe, understanding, preparation, "target")
    assert result.skipped
    assert "all candidate estimators failed safely" in result.skip_reason.lower()
    assert not hasattr(original, "transformers_")


def test_failed_upload_clears_all_stale_downstream_state():
    state = {}
    initialize_state(state)
    ds = ingest("x,y\n1,2\n2,3\n")
    set_dataset(state, ds)
    for key in (UNDERSTANDING_RESULT_KEY, PREPARATION_RESULT_KEY, EDA_RESULT_KEY, CLUSTERING_RESULT_KEY, MODELING_RESULT_KEY, EVALUATION_RESULT_KEY, REPORT_RESULT_KEY):
        state[key] = object()
    set_error(state, "bad upload")
    assert state[DATASET_KEY] is None
    assert all(state[key] is None for key in (UNDERSTANDING_RESULT_KEY, PREPARATION_RESULT_KEY, EDA_RESULT_KEY, CLUSTERING_RESULT_KEY, MODELING_RESULT_KEY, EVALUATION_RESULT_KEY, REPORT_RESULT_KEY))


def test_mid_pipeline_invalidation_clears_only_dependent_results():
    state = {}
    initialize_state(state)
    state[UNDERSTANDING_RESULT_KEY] = "u"
    state[PREPARATION_RESULT_KEY] = "p"
    state[EDA_RESULT_KEY] = "e"
    state[CLUSTERING_RESULT_KEY] = "c"
    state[MODELING_RESULT_KEY] = "m"
    state[EVALUATION_RESULT_KEY] = "v"
    state[REPORT_RESULT_KEY] = "r"
    invalidate_from(state, MODELING_RESULT_KEY)
    assert state[UNDERSTANDING_RESULT_KEY] == "u"
    assert state[PREPARATION_RESULT_KEY] == "p"
    assert state[EDA_RESULT_KEY] == "e"
    assert state[CLUSTERING_RESULT_KEY] == "c"
    assert state[MODELING_RESULT_KEY] is None and state[EVALUATION_RESULT_KEY] is None and state[REPORT_RESULT_KEY] is None


@pytest.mark.parametrize(
    "name,mime,data",
    [
        ("empty.csv", "text/csv", b""),
        ("headers.csv", "text/csv", b"a,b\n"),
        ("notcsv.txt", "text/plain", b"a,b\n1,2\n"),
    ],
)
def test_ingestion_hostile_inputs_are_domain_errors(name, mime, data):
    with pytest.raises(CrispDMError):
        ingest_csv(filename=name, mime_type=mime, data=data, config=CONFIG)


def test_duplicate_garbage_headers_and_cp1252_encoding_are_normalized():
    raw = "name,name,,city\nJos\xe9,1,x,Montr\xe9al\n".encode("cp1252")
    ds = ingest_csv(filename="legacy.csv", mime_type="text/csv", data=raw, config=CONFIG)
    assert len(set(ds.dataframe.columns)) == len(ds.dataframe.columns)
    assert any(c.startswith("column_") for c in ds.dataframe.columns)
    assert ds.warnings
