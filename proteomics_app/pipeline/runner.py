from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, List
import inspect
import re

from proteomics_app.steps import compile_data, stereoselectivity, peptide_search, splitting

LogFn = Callable[[str], None]


def _default_logger(msg: str) -> None:
    print(msg)


def _call_run(fn, /, *args, **kwargs):
    """
    Call fn(*args, **filtered_kwargs) where filtered_kwargs only includes params
    that fn actually accepts. Prevents 'unexpected keyword argument' errors when
    step signatures differ.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **filtered)


def _infer_identifier_from_filename(path: Path) -> str:
    """
    Infer window identifier from a CSV filename.
    Supports patterns like:
      MS3_0s.csv, ms3_5s_rep2.csv, something_MS2_30s.csv, etc.

    Returns "" if it cannot infer.
    """
    name = path.name.lower()

    ms = None
    if "ms3" in name:
        ms = "MS3"
    elif "ms2" in name:
        ms = "MS2"

    if ms is None:
        return ""

    # Prefer explicit window tokens
    if re.search(r"\b0s\b", name):
        return f"{ms}_0s"
    if re.search(r"\b5s\b", name):
        return f"{ms}_5s"
    if re.search(r"\b30s\b", name):
        return f"{ms}_30s"

    # Fallback: look for _0s/_5s/_30s without word boundaries
    for w in ("0s", "5s", "30s"):
        if w in name:
            return f"{ms}_{w}"

    return ""


def _make_unique_identifiers(specs: List[compile_data.InputSpec]) -> List[compile_data.InputSpec]:
    """
    If multiple CSVs map to the same identifier (e.g., multiple MS3_5s injections),
    rename them uniquely: MS3_5s__1, MS3_5s__2, ...
    """
    counts = {}
    out: List[compile_data.InputSpec] = []
    for s in specs:
        base = s.identifier
        counts[base] = counts.get(base, 0) + 1
        n = counts[base]
        if n == 1:
            out.append(s)
        else:
            out.append(compile_data.InputSpec(identifier=f"{base}__{n}", filepath=s.filepath))
    return out


@dataclass(frozen=True)
class PipelineOutputs:
    merged_xlsx: Path
    stereo_xlsx: Path
    sites_xlsx: Path
    split_xlsx: Path
    traces_parquet: Path
    run_dir: Path


@dataclass(frozen=True)
class PipelineInputs:
    # -------------------------
    # Step 1 inputs (NEW)
    # -------------------------
    # Option A: pass explicit InputSpec list (full control)
    step1_specs: Optional[List[compile_data.InputSpec]] = None
    # Option B: pass arbitrary CSV file list (identifiers inferred from filenames)
    step1_csvs: Optional[List[Path]] = None

    # -------------------------
    # Step 1 legacy (BACKWARDS COMPAT)
    # -------------------------
    ms3_0s_csv: Optional[Path] = None
    ms3_5s_csv: Optional[Path] = None
    ms3_30s_csv: Optional[Path] = None
    ms2_5s_csv: Optional[Path] = None
    ms2_30s_csv: Optional[Path] = None

    # Step 3 peptide mapping inputs
    combined_peptide_ms3: Path = Path()
    peptide_list_ms3: Path = Path()
    combined_peptide_ms2: Path = Path()
    peptide_list_ms2: Path = Path()

    # Step 4 skyline inputs
    skyline_ms2_tsv: Path = Path()
    skyline_ms3_tsv: Path = Path()
    ms2_peaks_csv: Path = Path()
    ms3_peaks_csv: Path = Path()


def _build_compile_specs(inputs: PipelineInputs, log: LogFn) -> List[compile_data.InputSpec]:
    """
    Priority:
      1) inputs.step1_specs (explicit)
      2) inputs.step1_csvs (infer identifiers)
      3) legacy 5 fields (MS3_0s, MS3_5s, MS3_30s, MS2_5s, MS2_30s)
    """
    if inputs.step1_specs:
        specs = inputs.step1_specs
        log(f"Step 1: using explicit step1_specs ({len(specs)} CSVs)")
        return _make_unique_identifiers(specs)

    if inputs.step1_csvs:
        specs: List[compile_data.InputSpec] = []
        for p in inputs.step1_csvs:
            p = Path(p)
            ident = _infer_identifier_from_filename(p)
            if not ident:
                # fall back to filename stem if window can't be inferred
                ident = p.stem
            specs.append(compile_data.InputSpec(identifier=ident, filepath=str(p)))
        specs = _make_unique_identifiers(specs)
        log(f"Step 1: using inferred step1_csvs ({len(specs)} CSVs)")
        return specs

    # Legacy
    legacy = [
        ("MS3_0s", inputs.ms3_0s_csv),
        ("MS3_5s", inputs.ms3_5s_csv),
        ("MS3_30s", inputs.ms3_30s_csv),
        ("MS2_5s", inputs.ms2_5s_csv),
        ("MS2_30s", inputs.ms2_30s_csv),
    ]
    missing = [k for k, v in legacy if v is None]
    if missing:
        raise ValueError(
            "Step 1 inputs missing. Provide either step1_specs, step1_csvs, or all legacy fields. "
            f"Missing legacy: {missing}"
        )

    specs = [compile_data.InputSpec(identifier=k, filepath=str(Path(v))) for k, v in legacy]  # type: ignore[arg-type]
    log("Step 1: using legacy 5 CSV inputs")
    return specs


def run_pipeline(
    *,
    inputs: PipelineInputs,
    outputs_dir: Path,
    log: Optional[LogFn] = None,
    debug: bool = True,
) -> PipelineOutputs:
    log = log or _default_logger
    outdir = Path(outputs_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    merged_xlsx = outdir / "merged_log2fc.xlsx"
    stereo_xlsx = outdir / "merged_log2fc_results.xlsx"
    sites_xlsx = outdir / "merged_log2fc_results_sites.xlsx"
    split_xlsx = outdir / "merged_log2fc_results_sites_with_skyline_splitting.xlsx"
    traces_parquet = outdir / "chrom_traces.parquet"

    # -----------------------
    # Step 1: compile_data
    # -----------------------
    log("Step 1/4 — compile_data: merging CSVs into wide XLSX")

    compile_specs = _build_compile_specs(inputs, log)

    _call_run(
        compile_data.run,
        inputs=compile_specs,
        out_dir=outdir,
        log=log,
        debug=debug,
    )

    if not merged_xlsx.exists():
        raise RuntimeError(f"compile_data did not produce {merged_xlsx.name} in {outdir}")

    # -----------------------
    # Step 2: stereoselectivity
    # -----------------------
    log("Step 2/4 — stereoselectivity: collapse duplicates + UniProt + hitlists")

    stereo_inputs = stereoselectivity.StereoInputs(
        input_xlsx=merged_xlsx,
        input_sheet=0,
    )

    _call_run(
        stereoselectivity.run,
        stereo_inputs,
        outdir,
        log=log,
        debug=debug,
    )

    if not stereo_xlsx.exists():
        raise RuntimeError(f"stereoselectivity did not produce {stereo_xlsx.name} in {outdir}")

    # -----------------------
    # Step 3: peptide_search
    # -----------------------
    log("Step 3/4 — peptide_search: map peptides + modified variants")

    ps_inputs = peptide_search.PeptideSearchInputs(
        input_xlsx=stereo_xlsx,
        peptide_table_ms3=inputs.combined_peptide_ms3,
        peptide_list_ms3=inputs.peptide_list_ms3,
        peptide_table_ms2=inputs.combined_peptide_ms2,
        peptide_list_ms2=inputs.peptide_list_ms2,
    )

    _call_run(
        peptide_search.run,
        ps_inputs,
        outdir,
        output_filename=sites_xlsx.name,
        log=log,
    )

    if not sites_xlsx.exists():
        raise RuntimeError(f"peptide_search did not produce {sites_xlsx.name} in {outdir}")

    # -----------------------
    # Step 4: splitting
    # -----------------------
    log("Step 4/4 — splitting: analyze skyline traces + peak boundary guided calling")

    split_inputs = splitting.SplittingInputs(
        sites_xlsx=sites_xlsx,
        sky_ms2_tsv=inputs.skyline_ms2_tsv,
        sky_ms3_tsv=inputs.skyline_ms3_tsv,
        ms2_peaks_csv=inputs.ms2_peaks_csv,
        ms3_peaks_csv=inputs.ms3_peaks_csv,
    )

    _call_run(
        splitting.run,
        split_inputs,
        outdir,
        output_xlsx_name=split_xlsx.name,
        traces_parquet_name=traces_parquet.name,
        log=log,
    )

    if not split_xlsx.exists():
        raise RuntimeError(f"splitting did not produce {split_xlsx.name} in {outdir}")
    if not traces_parquet.exists():
        raise RuntimeError("splitting did not produce chrom_traces.parquet (pyarrow installed?)")

    log("✅ Pipeline complete.")

    return PipelineOutputs(
        merged_xlsx=merged_xlsx,
        stereo_xlsx=stereo_xlsx,
        sites_xlsx=sites_xlsx,
        split_xlsx=split_xlsx,
        traces_parquet=traces_parquet,
        run_dir=outdir,
    )
