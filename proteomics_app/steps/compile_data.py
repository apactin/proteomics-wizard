#!/usr/bin/env python3
"""
compile_data.py

Merges multiple FragPipe-Analyst CSVs into one Excel workbook.

Expected per-input CSV columns (exact match):
- Index (e.g., A0AVT1_C347)
- Protein ID
- Gene Name
- R_vs_S_log2 fold change
- R_vs_S_p.adj

Output Excel columns:
Site, Gene Name, Uniprot ID,
<IDENTIFIER>_Log2FC, ... then <IDENTIFIER>_p.adj ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import pandas as pd


# ---- Expected input column names (EXACT match) ----
INDEX_COL = "Index"
UNIPROT_COL_IN = "Protein ID"
GENE_COL_IN = "Gene Name"
LOG2FC_COL_IN = "R_vs_S_log2 fold change"
PADJ_COL_IN = "R_vs_S_p.adj"

# ---- Output column names ----
SITE_COL_OUT = "Site"
GENE_COL_OUT = "Gene Name"
UNIPROT_COL_OUT = "Uniprot ID"


# =======================
# Public API (for Streamlit)
# =======================

@dataclass(frozen=True)
class InputSpec:
    identifier: str
    filepath: Path


@dataclass(frozen=True)
class CompileOutputs:
    merged_xlsx: Path
    n_rows: int
    n_inputs: int


LogFn = Callable[[str], None]


def extract_site_token(index_val: object) -> Optional[str]:
    """
    Extracts 'C347' from something like 'A0AVT1_C347'.

    IMPORTANT: Do not use \\b word boundaries here because '_' is a word character,
    so '_C347' does NOT create a word boundary before 'C'.
    """
    if pd.isna(index_val):
        return None
    s = str(index_val).strip()
    if not s:
        return None

    m = re.search(r"C\d+", s)
    return m.group(0) if m else None


def _empty_parsed_df(identifier: str) -> pd.DataFrame:
    """Return an empty dataframe with the expected schema for a parsed file."""
    fc_col = f"{identifier}_Log2FC"
    padj_col = f"{identifier}_p.adj"
    return pd.DataFrame(columns=["_merge_key", SITE_COL_OUT, GENE_COL_OUT, UNIPROT_COL_OUT, fc_col, padj_col])


def _check_required_columns(df: pd.DataFrame, spec: InputSpec, log: Optional[LogFn]) -> None:
    required = [INDEX_COL, UNIPROT_COL_IN, GENE_COL_IN, LOG2FC_COL_IN, PADJ_COL_IN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        if log:
            log(f"[ERROR] Missing required columns in {spec.filepath.name}: {missing}")
            log(f"[ERROR] Columns found ({len(df.columns)}): {list(df.columns)}")
        raise KeyError(f"{spec.filepath} is missing required columns: {missing}")


def _load_one_csv(
    spec: InputSpec,
    *,
    log: Optional[LogFn] = None,
    write_debug_files: bool = False,
    debug_dir: Optional[Path] = None,
) -> pd.DataFrame:
    fp = Path(spec.filepath)
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp.resolve()}")

    if log:
        log(f"\n========== Loading: {fp.name} (identifier={spec.identifier}) ==========")

    df = pd.read_csv(fp)

    if log:
        log(f"[INFO] Raw rows: {len(df)}")
        log(f"[INFO] Raw columns ({len(df.columns)}): {list(df.columns)}")

    _check_required_columns(df, spec, log)

    if log:
        example_vals = df[INDEX_COL].dropna().astype(str).head(5).tolist()
        log(f"[INFO] First 5 non-null {INDEX_COL} examples: {example_vals}")

    df["_Csite"] = df[INDEX_COL].apply(extract_site_token)
    n_found = int(df["_Csite"].notna().sum())
    if log:
        log(f"[INFO] Extracted C-site token for {n_found}/{len(df)} rows")

    if n_found == 0:
        if log:
            log("[WARN] No C-site tokens found. This file contributes no rows.")
        if write_debug_files and debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            out_debug = debug_dir / f"DEBUG_{spec.identifier}_raw_index_preview.tsv"
            df[[INDEX_COL]].head(200).to_csv(out_debug, sep="\t", index=False)
            if log:
                log(f"[DEBUG] Wrote {out_debug} (first 200 Index values).")
        return _empty_parsed_df(spec.identifier)

    before = len(df)
    df = df.loc[df["_Csite"].notna()].copy()
    if log:
        log(f"[INFO] After filtering rows without extracted site: {len(df)}/{before}")

    # Standardize outputs
    df[UNIPROT_COL_OUT] = df[UNIPROT_COL_IN].astype(str).str.strip()
    df[GENE_COL_OUT] = df[GENE_COL_IN].astype(str).str.strip()
    df[SITE_COL_OUT] = df[GENE_COL_OUT] + " " + df["_Csite"].astype(str)

    # File-specific output columns
    fc_col = f"{spec.identifier}_Log2FC"
    padj_col = f"{spec.identifier}_p.adj"

    df[fc_col] = pd.to_numeric(df[LOG2FC_COL_IN], errors="coerce")
    df[padj_col] = pd.to_numeric(df[PADJ_COL_IN], errors="coerce")

    if log:
        log(f"[INFO] Non-null Log2FC values in {fc_col}: {int(df[fc_col].notna().sum())}/{len(df)}")
        log(f"[INFO] Non-null p.adj values in {padj_col}: {int(df[padj_col].notna().sum())}/{len(df)}")

    # Merge key
    df["_merge_key"] = df[UNIPROT_COL_OUT] + "|" + df["_Csite"].astype(str)

    # Deduplicate within file
    df_out = (
        df[["_merge_key", SITE_COL_OUT, GENE_COL_OUT, UNIPROT_COL_OUT, fc_col, padj_col]]
        .sort_values(by=[fc_col], na_position="last")
        .drop_duplicates(subset=["_merge_key"], keep="first")
        .reset_index(drop=True)
    )

    if log:
        log(f"[INFO] Rows after deduplication: {len(df_out)}")
        log(f"[INFO] Preview rows:\n{df_out.head(5).to_string(index=False)}")

    if write_debug_files and debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        out_debug = debug_dir / f"DEBUG_{spec.identifier}_parsed_preview.tsv"
        df_out.head(200).to_csv(out_debug, sep="\t", index=False)
        if log:
            log(f"[DEBUG] Wrote {out_debug} (first 200 parsed rows).")

    return df_out


def merge_all(
    inputs: Sequence[InputSpec],
    *,
    log: Optional[LogFn] = None,
    write_debug_files: bool = False,
    debug_dir: Optional[Path] = None,
) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None

    if not inputs:
        raise ValueError("No inputs were provided.")

    for i, spec in enumerate(inputs, start=1):
        cur = _load_one_csv(spec, log=log, write_debug_files=write_debug_files, debug_dir=debug_dir)
        fc_col = f"{spec.identifier}_Log2FC"
        padj_col = f"{spec.identifier}_p.adj"

        if log:
            log(f"\n----- Merging file {i}/{len(inputs)}: {spec.identifier} -----")
            log(f"[INFO] Current parsed rows: {len(cur)}")

        if merged is None:
            merged = cur.copy()
            if log:
                log(f"[INFO] Initialized merged table with {len(merged)} rows")
        else:
            before_rows = len(merged)
            merged = merged.merge(
                cur[["_merge_key", fc_col, padj_col]],
                on="_merge_key",
                how="outer",
                validate="one_to_one",
            )
            if log:
                log(f"[INFO] Merged rows: {before_rows} -> {len(merged)} (outer join)")

            # Backfill metadata for rows introduced by later files
            meta_cols = [SITE_COL_OUT, GENE_COL_OUT, UNIPROT_COL_OUT]
            merged = merged.merge(
                cur[["_merge_key", *meta_cols]],
                on="_merge_key",
                how="left",
                suffixes=("", "_new"),
            )
            for c in meta_cols:
                newc = f"{c}_new"
                if newc in merged.columns:
                    merged[c] = merged[c].where(merged[c].notna(), merged[newc])
                    merged.drop(columns=[newc], inplace=True)

        if log and merged is not None:
            log(f"[INFO] Merged preview:\n{merged.head(5).to_string(index=False)}")

    assert merged is not None

    # Ensure all expected per-file columns exist
    fc_cols = [f"{spec.identifier}_Log2FC" for spec in inputs]
    padj_cols = [f"{spec.identifier}_p.adj" for spec in inputs]
    for c in fc_cols + padj_cols:
        if c not in merged.columns:
            merged[c] = pd.NA

    # Arrange final columns: metadata → all Log2FC → all p.adj
    out_cols = [SITE_COL_OUT, GENE_COL_OUT, UNIPROT_COL_OUT]
    for spec in inputs:
        out_cols.append(f"{spec.identifier}_Log2FC")
    for spec in inputs:
        out_cols.append(f"{spec.identifier}_p.adj")

    out = merged.loc[:, out_cols].copy()
    out = out.sort_values(by=[GENE_COL_OUT, SITE_COL_OUT, UNIPROT_COL_OUT], kind="stable").reset_index(drop=True)
    return out


def run(
    inputs: Sequence[InputSpec],
    out_dir: Path,
    *,
    output_filename: str = "merged_log2fc.xlsx",
    log: Optional[LogFn] = None,
    write_debug_files: bool = False,
) -> CompileOutputs:
    """
    Streamlit-friendly entry point.

    - inputs: list of InputSpec(identifier, filepath)
    - out_dir: where outputs will be written
    Returns CompileOutputs with path to merged Excel.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = out_dir / "debug" if write_debug_files else None

    if log:
        log("Starting merge...")

    merged = merge_all(inputs, log=log, write_debug_files=write_debug_files, debug_dir=debug_dir)

    out_path = out_dir / output_filename
    merged.to_excel(out_path, index=False)

    if log:
        log(f"\n✅ Wrote: {out_path.resolve()} ({len(merged)} rows)")

    return CompileOutputs(merged_xlsx=out_path, n_rows=int(len(merged)), n_inputs=int(len(inputs)))


# =======================
# Optional CLI wrapper
# =======================

def _default_inputs_from_cwd() -> List[InputSpec]:
    # Preserves your previous default behavior, but now as an optional CLI convenience.
    return [
        InputSpec(identifier="MS3_0s", filepath=Path("MS3_0s.csv")),
        InputSpec(identifier="MS3_5s", filepath=Path("MS3_5s.csv")),
        InputSpec(identifier="MS3_30s", filepath=Path("MS3_30s.csv")),
        InputSpec(identifier="MS2_5s", filepath=Path("MS2_5s.csv")),
        InputSpec(identifier="MS2_30s", filepath=Path("MS2_30s.csv")),
    ]


def main() -> None:
    inputs = _default_inputs_from_cwd()
    out_dir = Path(".")
    run(inputs, out_dir, output_filename="merged_log2fc.xlsx", log=print, write_debug_files=True)


if __name__ == "__main__":
    main()
