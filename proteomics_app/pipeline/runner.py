from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import inspect

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
    # Step 1 CSVs
    ms3_0s_csv: Path
    ms3_5s_csv: Path
    ms3_30s_csv: Path
    ms2_5s_csv: Path
    ms2_30s_csv: Path

    # Step 3 peptide mapping inputs
    combined_peptide_ms3: Path
    peptide_list_ms3: Path
    combined_peptide_ms2: Path
    peptide_list_ms2: Path

    # Step 4 skyline inputs
    skyline_ms2_tsv: Path
    skyline_ms3_tsv: Path
    ms2_peaks_csv: Path
    ms3_peaks_csv: Path


def run_pipeline(
    *,
    inputs: PipelineInputs,
    outputs_dir: Path,
    log: Optional[LogFn] = None,
    debug: bool = True,
) -> PipelineOutputs:
    """
    Run all 4 steps in order inside outputs_dir.

    This runner is compatible with the refactored step entrypoints:
      - compile_data.run(inputs=[InputSpec...], out_dir=..., ...)
      - stereoselectivity.run(inputs=StereoInputs(...), out_dir=..., ...)
      - peptide_search.run(inputs=PeptideSearchInputs(...), out_dir=..., ...)
      - splitting.run(inputs=SplittingInputs(...), out_dir=..., ...)
    """
    log = log or _default_logger
    outdir = Path(outputs_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Canonical filenames in outputs_dir
    merged_xlsx = outdir / "merged_log2fc.xlsx"
    stereo_xlsx = outdir / "merged_log2fc_results.xlsx"
    sites_xlsx = outdir / "merged_log2fc_results_sites.xlsx"
    split_xlsx = outdir / "merged_log2fc_results_sites_with_skyline_splitting.xlsx"
    traces_parquet = outdir / "chrom_traces.parquet"

    # -----------------------
    # Step 1: compile_data
    # -----------------------
    log("Step 1/4 — compile_data: merging CSVs into wide XLSX")

    compile_specs = [
        compile_data.InputSpec(identifier="MS3_0s", filepath=str(inputs.ms3_0s_csv)),
        compile_data.InputSpec(identifier="MS3_5s", filepath=str(inputs.ms3_5s_csv)),
        compile_data.InputSpec(identifier="MS3_30s", filepath=str(inputs.ms3_30s_csv)),
        compile_data.InputSpec(identifier="MS2_5s", filepath=str(inputs.ms2_5s_csv)),
        compile_data.InputSpec(identifier="MS2_30s", filepath=str(inputs.ms2_30s_csv)),
    ]

    # Your compile_data.run writes merged_log2fc.xlsx into out_dir (per your logs)
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
        stereo_inputs,   # positional: inputs
        outdir,          # positional: out_dir
        log=log,
        debug=debug,
    )

    if not stereo_xlsx.exists():
        raise RuntimeError(f"stereoselectivity did not produce {stereo_xlsx.name} in {outdir}")

    # -----------------------
    # Step 3: peptide_search  ✅ FIXED (field names)
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
        ps_inputs,  # positional: inputs
        outdir,     # positional: out_dir
        output_filename=sites_xlsx.name,
        log=log,
    )

    if not sites_xlsx.exists():
        raise RuntimeError(f"peptide_search did not produce {sites_xlsx.name} in {outdir}")


    # -----------------------
    # Step 4: splitting      ✅ FIXED
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
        split_inputs,  # positional: inputs
        outdir,        # positional: out_dir
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
