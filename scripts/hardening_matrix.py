"""Generate the Phase 9 hostile-input hardening matrix from executable checks."""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.pipeline import Pipeline
from crispdm_studio.config import CONFIG
from crispdm_studio.exceptions import CrispDMError
from crispdm_studio.ingestion import ingest_csv
from crispdm_studio.understanding import profile_dataset
from crispdm_studio.preparation import prepare_dataset
from crispdm_studio.eda import run_eda
from crispdm_studio.clustering import run_clustering
from crispdm_studio.modeling import run_modeling
from crispdm_studio.state import initialize_state, set_dataset, set_error, DATASET_KEY, REPORT_RESULT_KEY

rows=[]
def add(case, expected, actual, ok=True): rows.append((case, expected, ("PASS — " if ok else "FAIL — ")+actual))
def ingest(text, name="case.csv", mime="text/csv", encoding="utf-8"):
    return ingest_csv(filename=name,mime_type=mime,data=text.encode(encoding),config=CONFIG)
def prep(text):
    d=ingest(text); u=profile_dataset(d); p=prepare_dataset(d.dataframe,u); return d,u,p

for case,name,mime,data,msg in [
    ("Empty file","empty.csv","text/csv",b"","friendly ingestion rejection"),
    ("Zero-row/header-only","headers.csv","text/csv",b"a,b\n","friendly zero-row rejection"),
    ("Non-CSV upload","bad.txt","text/plain",b"a,b\n1,2\n","friendly file-type rejection"),
]:
    try: ingest_csv(filename=name,mime_type=mime,data=data,config=CONFIG); add(case,msg,"unexpectedly accepted",False)
    except CrispDMError as e: add(case,msg,e.user_message)

try:
    d,u,p=prep("a,b\n1,x\n"); e=run_eda(p.structural_dataframe,u); c=run_clustering(p.structural_dataframe,u)
    add("Single-row dataset","descriptive stages survive; clustering skips",f"EDA returned; clustering skipped={c.skipped}",c.skipped)
except Exception as e: add("Single-row dataset","safe degradation",type(e).__name__,False)

try:
    d,u,p=prep("label\nA\nB\nA\nB\nA\nB\nA\nB\n"); c=run_clustering(p.structural_dataframe,u); m=run_modeling(p.structural_dataframe,u,p,None)
    add("Single-column dataset","downstream unsupported stages skip",f"clustering skipped={c.skipped}; modeling skipped={m.skipped}",c.skipped and m.skipped)
except Exception as e: add("Single-column dataset","safe degradation",type(e).__name__,False)

for case,text,kind in [
    ("All-numeric","x,y,z\n1,10,3\n2,11,4\n4,12,8\n7,14,9\n9,16,12\n","numeric"),
    ("All-categorical","a,b,c\nred,x,yes\nblue,y,no\nred,y,yes\nblue,x,no\nred,x,no\nblue,y,yes\n","categorical"),
]:
    d,u,p=prep(text); e=run_eda(p.structural_dataframe,u)
    ok=bool(e.numeric_columns) if kind=="numeric" else bool(e.categorical_columns)
    add(case,"adaptive EDA only runs applicable analyses",f"numeric={len(e.numeric_columns)}, categorical={len(e.categorical_columns)}, skipped={len(e.skipped_analyses)}",ok)

text="a,b,c,target\n1,,,,\n,foo,,,\n,,,yes\n,,,no\n,,,yes\n,,,no\n,,,yes\n,,,no\n"
d,u,p=prep(text); e=run_eda(p.structural_dataframe,u)
add("Near-total missingness","quality warnings + safe descriptive result",f"quality warnings={len(u.quality_warnings)}; EDA returned",bool(u.quality_warnings))

cols=[f"f{i}" for i in range(80)]; source=[",".join(cols)]
for r in range(120): source.append(",".join(str((r*(i+3))%17) if i<40 else f"v{i}_{r}" for i in range(80)))
d,u,p=prep("\n".join(source)+"\n"); e=run_eda(p.structural_dataframe,u)
add("Wide/high-cardinality","bounded charts/relationships; no crash",f"80 source columns; {len(e.charts)} charts",len(e.charts)<100)

raw="name,name,,city\nJos\xe9,1,x,Montr\xe9al\n".encode("cp1252")
d=ingest_csv(filename="legacy.csv",mime_type="text/csv",data=raw,config=CONFIG)
add("Duplicate/garbage headers + CP1252","normalize headers and decode",f"columns={list(d.dataframe.columns)}; warnings={len(d.warnings)}",bool(d.warnings))

d,u,p=prep("x,target\n1,A\n2,A\n3,A\n4,A\n5,A\n6,A\n7,A\n8,B\n"); m=run_modeling(p.structural_dataframe,u,p,"target")
add("Highly imbalanced/tiny target","skip unsafe stratified modeling",m.skip_reason or "",m.skipped)

d,u,p=prep("a,b\nx,1\nx,1\nx,1\nx,1\nx,1\nx,1\nx,1\nx,1\n"); c=run_clustering(p.structural_dataframe,u)
add("No clusterable features","structured clustering skip",c.selection_reason,c.skipped)

# forced clustering failure
import crispdm_studio.clustering.engine as ce
orig=ce.search_kmeans
try:
    ce.search_kmeans=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("forced"))
    d,u,p=prep("x,y\n1,2\n2,3\n3,5\n4,7\n5,11\n6,13\n7,17\n8,19\n9,23\n10,29\n")
    c=run_clustering(p.structural_dataframe,u); add("Forced clustering failure","convert exception to structured skip",c.selection_reason,c.skipped)
finally: ce.search_kmeans=orig


# forced model failure: all estimators fail, but engine returns a structured skip
rows_src=["x,region,target"]+[f"{i},{'A' if i%2 else 'B'},{'yes' if i>14 else 'no'}" for i in range(30)]
d,u,p=prep("\n".join(rows_src)+"\n")
orig_fit=Pipeline.fit
try:
    Pipeline.fit=lambda self,*a,**k: (_ for _ in ()).throw(RuntimeError("forced"))
    m=run_modeling(p.structural_dataframe,u,p,"target")
    add("Forced model failure","exclude failed estimators / clean skip if all fail",m.skip_reason or "",m.skipped and bool(m.model_failures))
finally:
    Pipeline.fit=orig_fit

# forced static-chart export failure: canonical HTML remains usable and PDF degrades
from crispdm_studio.eda import run_eda
from crispdm_studio.evaluation import evaluate_modeling
from crispdm_studio.reporting.context import build_report_context
from crispdm_studio.reporting.html_report import render_html_report
import crispdm_studio.reporting.pdf_report as pr
from crispdm_studio.reporting.chart_export import ChartExportError
rows_src=["x,y,target"]+[f"{i},{i%5},{'yes' if i>14 else 'no'}" for i in range(30)]
d,u,p=prep("\n".join(rows_src)+"\n"); e=run_eda(p.structural_dataframe,u); c=run_clustering(p.structural_dataframe,u,min_rows=5); m=run_modeling(p.structural_dataframe,u,p,"target"); v=evaluate_modeling(m,e,c,u)
ctx=build_report_context(u,p,e,c,m,v); html=render_html_report(ctx)
orig_chart=pr.chart_to_png_bytes
try:
    pr.chart_to_png_bytes=lambda *a,**k: (_ for _ in ()).throw(ChartExportError("forced static renderer outage"))
    pdf=pr.render_pdf_report(html,ctx)
    add("Forced export failure","HTML survives; PDF uses placeholders/text fallback",f"html={html.startswith('<!doctype html>')}; fidelity={pdf.fidelity}; placeholders={pdf.static_charts_placeholder}",html.startswith("<!doctype html>") and pdf.pdf_bytes.startswith(b"%PDF") and pdf.fidelity in {"chart-fallback","text-only-fallback"})
finally:
    pr.chart_to_png_bytes=orig_chart

# stale state after bad upload
state={}; initialize_state(state); d=ingest("x,y\n1,2\n2,3\n"); set_dataset(state,d); state[REPORT_RESULT_KEY]=object(); set_error(state,"bad upload")
add("New/failed upload after old report","clear dataset and all downstream state",f"dataset={state[DATASET_KEY]}; report={state[REPORT_RESULT_KEY]}",state[DATASET_KEY] is None and state[REPORT_RESULT_KEY] is None)

out=["# Phase 9 hardening matrix","","| Hostile input / failure | Expected behavior | Actual result |","|---|---|---|"]
for case,expected,actual in rows:
    out.append(f"| {case} | {expected} | {actual.replace('|','/')} |")
path=ROOT/"HARDENING_MATRIX.md"; path.write_text("\n".join(out)+"\n",encoding="utf-8")
print("\n".join(out))
if any(actual.startswith("FAIL") for _,_,actual in rows): raise SystemExit(1)
