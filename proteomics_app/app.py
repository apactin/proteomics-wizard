from __future__ import annotations

from pathlib import Path
import traceback
import json
import time
from typing import Optional
import os
import re

import streamlit as st

from proteomics_app.pipeline import (
    PipelineInputs,
    make_run_dir,
    save_uploaded_file_as,
    run_pipeline,
)

from proteomics_app.viewer.chrom_viewer import render_viewer
from proteomics_app.viewer.stats_viewer import render_stats


st.set_page_config(layout="wide")
st.title("Proteomics Pipeline Wizard")

st.markdown(
    """
This wizard runs a 4-step pipeline end-to-end:

1) Compile Log2FC CSVs → `merged_log2fc.xlsx`  
2) Stereoselectivity + UniProt annotation → `merged_log2fc_results.xlsx`  
3) Peptide mapping → `merged_log2fc_results_sites.xlsx`  
4) Skyline splitting + Parquet traces → final workbook + `chrom_traces.parquet`
"""
)

# -------------------------
# Persistence: saved runs folder
# -------------------------
APP_DIR = Path(__file__).resolve().parent
SAVED_RUNS_DIR = Path(os.environ.get("RUNS_DIR", str(APP_DIR / "saved_runs")))
SAVED_RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _now_run_id() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _list_saved_runs() -> list[Path]:
    runs = [p for p in SAVED_RUNS_DIR.iterdir() if p.is_dir()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def _save_run_manifest(run_dir: Path, outs) -> Path:
    """
    Saves a manifest.json inside a stable folder under saved_runs/.
    We copy outputs there so they remain loadable even if the original run_dir was temporary.
    """
    run_id = run_dir.name if run_dir else _now_run_id()
    save_dir = SAVED_RUNS_DIR / run_id
    (save_dir / "outputs").mkdir(parents=True, exist_ok=True)

    # Copy outputs to stable location
    def _copy(p: Path) -> Optional[str]:
        p = Path(p)
        if not p.exists():
            return None
        dst = save_dir / "outputs" / p.name
        dst.write_bytes(p.read_bytes())
        return str(dst)

    outputs = {
        "merged_xlsx": _copy(outs.merged_xlsx),
        "stereo_xlsx": _copy(outs.stereo_xlsx),
        "sites_xlsx": _copy(outs.sites_xlsx),
        "split_xlsx": _copy(outs.split_xlsx),
        "traces_parquet": _copy(outs.traces_parquet),
    }

    manifest = {
        "run_id": run_id,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "outputs": outputs,
    }
    _write_json(save_dir / "manifest.json", manifest)
    return save_dir / "manifest.json"


def _load_run_from_manifest(manifest_path: Path):
    """
    Returns an object with the same attribute names as your outs object.
    (We keep it simple so we don't have to import your Outputs type.)
    """
    m = _read_json(manifest_path)
    out_paths = m.get("outputs", {})

    class LoadedOuts:
        merged_xlsx = Path(out_paths["merged_xlsx"]) if out_paths.get("merged_xlsx") else None
        stereo_xlsx = Path(out_paths["stereo_xlsx"]) if out_paths.get("stereo_xlsx") else None
        sites_xlsx = Path(out_paths["sites_xlsx"]) if out_paths.get("sites_xlsx") else None
        split_xlsx = Path(out_paths["split_xlsx"]) if out_paths.get("split_xlsx") else None
        traces_parquet = Path(out_paths["traces_parquet"]) if out_paths.get("traces_parquet") else None

    return LoadedOuts(), m


# -------------------------
# Step 1 flexible helpers
# -------------------------
def _infer_ms_and_window(filename: str) -> tuple[str, str]:
    """
    Return (ms_level, window) where:
      ms_level in {"ms2","ms3",""}
      window in {"0s","5s","30s",""}
    IMPORTANT: avoid '0s' matching inside '30s'.
    """
    fn = (filename or "").lower()

    ms = "ms3" if "ms3" in fn else ("ms2" if "ms2" in fn else "")

    # Check 30s before 0s, and use token-ish matching
    for w in ("30s", "5s", "0s"):
        if re.search(rf"(^|[^0-9a-z]){w}([^0-9a-z]|$)", fn):
            return ms, w

    return ms, ""




def _validate_step1_files(files) -> tuple[bool, str]:
    """
    Minimal validation:
      - At least one MS2 CSV and at least one MS3 CSV
    (Allows any number of injections per condition.)
    """
    ms2 = 0
    ms3 = 0
    unknown = 0
    for f in files or []:
        name = (getattr(f, "name", "") or "")
        ms, _ = _infer_ms_and_window(name)
        if ms == "ms2":
            ms2 += 1
        elif ms == "ms3":
            ms3 += 1
        else:
            unknown += 1

    if ms2 == 0 or ms3 == 0:
        return (
            False,
            f"Step 1 needs at least one MS2 CSV and one MS3 CSV. Detected: MS2={ms2}, MS3={ms3}, unknown={unknown}. "
            "Make sure filenames include 'MS2' or 'MS3' (and ideally 0s/5s/30s).",
        )
    return True, ""


def _save_step1_csvs(files, uploads_dir: Path) -> list[Path]:
    """
    Save all uploaded Step 1 CSVs into uploads_dir, keeping original names.
    Returns list of saved file Paths.
    """
    saved: list[Path] = []
    for f in files or []:
        name = (getattr(f, "name", "") or "").strip()
        if not name:
            continue
        dest = uploads_dir / name
        save_uploaded_file_as(f, dest)
        saved.append(dest)
    return saved


# -------------------------
# Session state
# -------------------------
if "run_ctx" not in st.session_state:
    st.session_state.run_ctx = None
if "logs" not in st.session_state:
    st.session_state.logs = ""
if "outputs" not in st.session_state:
    st.session_state.outputs = None
if "loaded_manifest" not in st.session_state:
    st.session_state.loaded_manifest = None  # path to manifest.json or None


def log(msg: str) -> None:
    st.session_state.logs += msg.rstrip() + "\n"


# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Run settings")
    base_dir_txt = st.text_input("Optional base run directory", value="")
    run_name_txt = st.text_input("Optional run name", value="")  # nice-to-have
    debug = st.checkbox("Debug logging", value=True)
    make_new = st.button("New run folder")

    if make_new or st.session_state.run_ctx is None:
        run_base = Path(base_dir_txt).expanduser() if base_dir_txt.strip() else None
        st.session_state.run_ctx = make_run_dir(
            run_base,
            prefix="proteomics",
            run_name=run_name_txt.strip() or None,
        )
        st.session_state.logs = ""
        st.session_state.outputs = None
        st.session_state.loaded_manifest = None

    run_ctx = st.session_state.run_ctx
    st.caption(f"Run folder:\n`{run_ctx.run_dir}`")
    st.caption(f"Uploads:\n`{run_ctx.uploads_dir}`")
    st.caption(f"Outputs:\n`{run_ctx.outputs_dir}`")

    st.divider()
    st.header("Saved runs")
    saved = _list_saved_runs()
    if saved:
        labels = [p.name for p in saved]
        pick = st.selectbox("Load previous run", labels, index=0)
        if st.button("Load selected run"):
            mp = SAVED_RUNS_DIR / pick / "manifest.json"
            if not mp.exists():
                st.error("Selected run is missing manifest.json")
            else:
                outs_loaded, meta = _load_run_from_manifest(mp)
                st.session_state.outputs = outs_loaded
                st.session_state.loaded_manifest = str(mp)
                st.success(f"Loaded run: {pick}")
    else:
        st.caption("No saved runs yet.")

    st.divider()
    st.header("Logs")
    st.code(st.session_state.logs or "(no logs yet)", language="text")


run_ctx = st.session_state.run_ctx
uploads_dir: Path = Path(run_ctx.uploads_dir)
outputs_dir: Path = Path(run_ctx.outputs_dir)

# -------------------------
# Tabs: Pipeline vs Viewer
# -------------------------
tab_info, tab_pipeline, tab_viewer, tab_stats = st.tabs(["Info", "Pipeline", "Viewer", "Stats"])


# -------------------------
# Info tab
# -------------------------
with tab_info:
    st.header("Proteomics Pipeline Overview")

    st.markdown(
        """
This application implements an end-to-end LC/MS proteomics analysis pipeline
designed to quantify stereoselective covalent labeling events and visualize
chromatographic evidence supporting site-level calls.

This tab provides a high-level overview of the pipeline logic, data flow,
and interpretation of outputs.
"""
    )

    st.divider()

    # -------------------------
    st.subheader("Pipeline Structure")

    st.markdown(
        """
The pipeline is organized into **four sequential stages**, each producing
intermediate outputs that feed into downstream analysis:

1. **Log2FC Compilation**
2. **Stereoselectivity Scoring + UniProt Annotation**
3. **Peptide-to-Site Mapping**
4. **Chromatographic Peak Splitting + Trace Extraction**

Each stage can be inspected independently once outputs are generated.
"""
    )

    st.divider()

    # -------------------------
    st.subheader("Step 1: Log2FC Compilation (Input CSVs)")

    st.markdown(
        """
**Purpose:**  
Aggregate replicate-level LC/MS quantification files into a unified
wide-format dataset containing Log2 fold-change values.

**Key concepts:**
- Supports both MS2 and MS3 quantification
- Automatically infers MS level and dynamic exclusion window (0s / 5s / 30s)
- Handles multiple injections per condition

**Primary output:**  
`merged_log2fc.xlsx`
"""
    )

    st.divider()

    # -------------------------
    st.subheader("Step 2: Stereoselectivity Analysis")

    st.markdown(
        """
**Purpose:**  
Compute site-level stereoselectivity scores and integrate protein annotations.

**Key concepts:**
- Window-aware scoring across available MS2/MS3 data
- Adaptive hit calling based on data completeness
- UniProt metadata enrichment

**Primary output:**  
`merged_log2fc_results.xlsx`
"""
    )

    st.divider()

    # -------------------------
    st.subheader("Step 3: Peptide Mapping")

    st.markdown(
        """
**Purpose:**  
Map quantified peptides to specific modification sites and protein positions.

**Key concepts:**
- Uses FragPipe peptide mapping outputs
- Resolves ambiguous peptide-to-site relationships
- Produces a site-centric view of the dataset

**Primary output:**  
`merged_log2fc_results_sites.xlsx`
"""
    )

    st.divider()

    # -------------------------
    st.subheader("Step 4: Chromatographic Peak Splitting")

    st.markdown(
        """
**Purpose:**  
Analyze Skyline chromatograms to detect peak splitting and extract
chromatographic traces for visualization.

**Key concepts:**
- Identifies single vs split peaks
- Computes quantitative splitting metrics
- Extracts full chromatographic traces into a columnar format

**Primary outputs:**
- Final split workbook (`*_split.xlsx`)
- Chromatographic traces (`chrom_traces.parquet`)
"""
    )

    st.divider()

    # -------------------------
    st.subheader("Viewer Tab")

    st.markdown(
        """
The **Viewer** tab enables interactive inspection of chromatographic evidence
supporting individual site calls.

**Features include:**
- Filtering by hit category and scoring metrics
- Interactive trace visualization
- Cross-navigation between related sites and conditions

This tab is intended for **manual validation and exploratory analysis**.
"""
    )

    st.divider()

    # -------------------------
    st.subheader("Stats Tab")

    st.markdown(
        """
The **Stats** tab provides summary-level views of the dataset, including:

- Hit counts and overlap summaries
- Distributions of stereoselectivity scores
- Comparisons across MS modes and windows

These views are designed to support high-level interpretation
and dataset quality assessment.
"""
    )

    st.divider()

    # -------------------------
    st.subheader("Interpreting Results")

    st.markdown(
        """
When interpreting results from this pipeline, consider:

- Consistency across MS modes and time windows
- Chromatographic support for site-level calls
- Potential confounding effects from peptide coverage or peak interference

Final conclusions should integrate **both quantitative scores and
chromatographic evidence**.
"""
    )

# -------------------------
# Pipeline tab
# -------------------------
with tab_pipeline:
    st.header("Step 1: Upload Log2FC CSVs (batch)")

    step1_files = st.file_uploader(
        "Upload any number of LFC CSVs (filenames should include MS2/MS3 and ideally 0s/5s/30s)",
        type=["csv"],
        accept_multiple_files=True,
    )

    st.caption(
        "Example file names include MS3_30s.csv, MS3_5s.csv, MS3_0s.csv, MS2_30s.csv, MS2_5s.csv. One file per injection. "
    )

    ok1, msg1 = _validate_step1_files(step1_files)
    if step1_files and not ok1:
        st.warning(msg1)

    st.header("Step 2: Automatic stereoselectivity scoring and UniProt annotation")

    st.caption(
        "No file upload required for this step. "
    )

    st.header("Step 3: Upload peptide mapping files (batch)")

    step3_files = st.file_uploader(
        "Upload peptide mapping inputs (combined_peptide_MS3.tsv, peptide_list_MS3.txt, combined_peptide_MS2.tsv, peptide_list_MS2.txt)",
        type=["tsv", "txt"],
        accept_multiple_files=True,
    )

    st.caption(
        "These are files generated by FragPipe. "
        "Example path: FP_MS2_Log2/combined_peptide.tsv and FP_MS2_Log2/skyline_files/peptide_list.txt. "
        "Rename files to include _MS2 or _MS3 before uploading. There should be 2 MS2 files and 2 MS3 files (4 files total). "
    )

    st.header("Step 4: Upload Skyline exports + peak boundaries (batch)")

    step4_files = st.file_uploader(
        "Upload Skyline TSV exports + MS2/MS3 peak boundary CSV reports",
        type=["tsv", "txt", "csv"],
        accept_multiple_files=True,
    )

    st.caption(
        "These files need to be generated in Skyline. "
        "File #1: Export -> Chromatograms -> Select all files and include all -> save as skyline_MS2_export.tsv (or MS3). "
        "File #2: Export -> Report -> select Peak Boundaries -> save as MS2_peaks_report.csv (or MS3)." \
        "There should be 2 MS2 files and 2 MS3 files (4 files total). "
    )

    def _index_uploads(files):
        """Return dict of lowercase filename -> UploadedFile."""
        out = {}
        for f in files or []:
            name = (getattr(f, "name", "") or "").strip()
            if not name:
                continue
            out[name.lower()] = f
        return out

    # Build indices (Step 1 still indexed, but only Step 3/4 use missing logic)
    idx3 = _index_uploads(step3_files)
    idx4 = _index_uploads(step4_files)

    REQ_STEP3 = {
        "combined_peptide_MS3.tsv": ["combined_peptide_ms3.tsv", "combined_peptide_ms3"],
        "peptide_list_MS3.txt": ["peptide_list_ms3.txt", "peptide_list_ms3"],
        "combined_peptide_MS2.tsv": ["combined_peptide_ms2.tsv", "combined_peptide_ms2"],
        "peptide_list_MS2.txt": ["peptide_list_ms2.txt", "peptide_list_ms2"],
    }

    REQ_STEP4 = {
        "skyline_MS2_export.tsv": ["skyline_ms2_export.tsv", "skyline_ms2_export", "skyline_ms2"],
        "skyline_MS3_export.tsv": ["skyline_ms3_export.tsv", "skyline_ms3_export", "skyline_ms3"],
        "MS2_peaks_report.csv": ["ms2_peaks_report.csv", "ms2_peaks_report", "ms2_peaks"],
        "MS3_peaks_report.csv": ["ms3_peaks_report.csv", "ms3_peaks_report", "ms3_peaks"],
    }

    def _find_uploaded(idx: dict, patterns: list[str]):
        """
        Try exact match first, then substring match.
        Returns UploadedFile or None.
        """
        for p in patterns:
            if p in idx:
                return idx[p]
        for fn, f in idx.items():
            for p in patterns:
                if p in fn:
                    return f
        return None

    def _missing_for(idx: dict, req: dict) -> list[str]:
        missing = []
        for canonical, patterns in req.items():
            if _find_uploaded(idx, patterns) is None:
                missing.append(canonical)
        return missing

    missing3 = _missing_for(idx3, REQ_STEP3)
    missing4 = _missing_for(idx4, REQ_STEP4)

    all_ok = ok1 and (len(missing3) == 0) and (len(missing4) == 0)

    if (step1_files or step3_files or step4_files) and (not all_ok):
        st.warning(
            "Missing required files:\n"
            + (f"Step 1: {msg1}\n" if (step1_files and not ok1) else ("" if ok1 else "Step 1: Please upload MS2+MS3 CSVs.\n"))
            + ("\n".join([f"Step 3: {m}" for m in missing3]) + "\n" if missing3 else "")
            + ("\n".join([f"Step 4: {m}" for m in missing4]) + "\n" if missing4 else "")
        )

    st.divider()

    run_button = st.button("Run pipeline", type="primary", disabled=not all_ok)

    if run_button:
        st.session_state.logs = ""
        st.session_state.outputs = None
        st.session_state.loaded_manifest = None

        try:
            log("Saving uploads into run folder...")

            # ---- Step 1 save (flexible) ----
            step1_saved = _save_step1_csvs(step1_files, uploads_dir)
            if len(step1_saved) == 0:
                raise RuntimeError("No Step 1 CSVs were uploaded/saved.")

            # ---- Step 3 save ----
            combined_ms3 = save_uploaded_file_as(
                _find_uploaded(idx3, REQ_STEP3["combined_peptide_MS3.tsv"]),
                uploads_dir / "combined_peptide_MS3.tsv",
            )
            list_ms3 = save_uploaded_file_as(
                _find_uploaded(idx3, REQ_STEP3["peptide_list_MS3.txt"]),
                uploads_dir / "peptide_list_MS3.txt",
            )
            combined_ms2 = save_uploaded_file_as(
                _find_uploaded(idx3, REQ_STEP3["combined_peptide_MS2.tsv"]),
                uploads_dir / "combined_peptide_MS2.tsv",
            )
            list_ms2 = save_uploaded_file_as(
                _find_uploaded(idx3, REQ_STEP3["peptide_list_MS2.txt"]),
                uploads_dir / "peptide_list_MS2.txt",
            )

            # ---- Step 4 save ----
            sky_ms2 = save_uploaded_file_as(
                _find_uploaded(idx4, REQ_STEP4["skyline_MS2_export.tsv"]),
                uploads_dir / "skyline_MS2_export.tsv",
            )
            sky_ms3 = save_uploaded_file_as(
                _find_uploaded(idx4, REQ_STEP4["skyline_MS3_export.tsv"]),
                uploads_dir / "skyline_MS3_export.tsv",
            )
            peaks_ms2 = save_uploaded_file_as(
                _find_uploaded(idx4, REQ_STEP4["MS2_peaks_report.csv"]),
                uploads_dir / "MS2_peaks_report.csv",
            )
            peaks_ms3 = save_uploaded_file_as(
                _find_uploaded(idx4, REQ_STEP4["MS3_peaks_report.csv"]),
                uploads_dir / "MS3_peaks_report.csv",
            )

            # NOTE: requires the updated PipelineInputs that supports step1_csvs
            pi = PipelineInputs(
                step1_csvs=step1_saved,
                combined_peptide_ms3=combined_ms3,
                peptide_list_ms3=list_ms3,
                combined_peptide_ms2=combined_ms2,
                peptide_list_ms2=list_ms2,
                skyline_ms2_tsv=sky_ms2,
                skyline_ms3_tsv=sky_ms3,
                ms2_peaks_csv=peaks_ms2,
                ms3_peaks_csv=peaks_ms3,
            )

            status = st.status("Running pipeline...", expanded=True)

            def ui_log(m: str) -> None:
                log(m)
                status.write(m)

            outs = run_pipeline(inputs=pi, outputs_dir=outputs_dir, log=ui_log, debug=debug)

            st.session_state.outputs = outs
            status.update(label="✅ Pipeline complete", state="complete", expanded=False)
            st.success("Pipeline finished successfully.")

            # Save a loadable snapshot of this run
            try:
                manifest_path = _save_run_manifest(Path(run_ctx.run_dir), outs)
                st.session_state.loaded_manifest = str(manifest_path)
                st.info(f"Saved run for re-loading: `{Path(manifest_path).parent.name}`")
            except Exception as e:
                log(f"⚠️ Could not save run snapshot: {e}")

        except Exception as e:
            tb = traceback.format_exc()
            log("❌ Pipeline failed.")
            log(str(e))
            log(tb)
            st.error(f"Pipeline failed: {e}")

    # Outputs list stays in Pipeline tab
    outs = st.session_state.outputs
    if outs is not None:
        st.header("Outputs")

        def dl(label: str, path: Path):
            path = Path(path)
            if path.exists():
                st.write(f"✅ **{label}**: `{path.name}`")
                st.download_button(
                    label=f"Download {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/octet-stream",
                )
            else:
                st.write(f"❌ **{label}** missing: `{path.name}`")

        dl("Step 1 merged workbook", outs.merged_xlsx)
        dl("Step 2 stereoselectivity workbook", outs.stereo_xlsx)
        dl("Step 3 sites + peptide mapping workbook", outs.sites_xlsx)
        dl("Step 4 splitting workbook", outs.split_xlsx)
        dl("Step 4 traces parquet", outs.traces_parquet)

# -------------------------
# Viewer tab
# -------------------------
with tab_viewer:
    outs = st.session_state.outputs

    if st.session_state.loaded_manifest:
        run_name = Path(st.session_state.loaded_manifest).parent.name
        st.caption(f"Loaded run: `{run_name}`")

    if outs is None:
        st.info("No run loaded. Run the pipeline in the Pipeline tab, or load a saved run from the sidebar.")
    else:
        # If loaded outs can contain None fields, check before rendering
        if not getattr(outs, "split_xlsx", None) or not getattr(outs, "traces_parquet", None):
            st.error("This run does not have Step 4 outputs (split workbook + traces parquet).")
        else:
            st.header("Viewer")
            render_viewer(
                xlsx_path=outs.split_xlsx,
                traces_path=outs.traces_parquet,
                embedded=True,
            )

# -------------------------
# Stats tab
# -------------------------
with tab_stats:
    outs = st.session_state.outputs
    if outs is None:
        st.info("No run loaded. Run the pipeline or load a saved run from the sidebar.")
    else:
        # Use Step 4 workbook if present; fallback to Step 2 stereo workbook
        render_stats(
            stereo_xlsx=getattr(outs, "stereo_xlsx", None),
            split_xlsx=getattr(outs, "split_xlsx", None),
            embedded=True,
        )
