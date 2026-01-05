#!/usr/bin/env python3
"""
peptide_search.py

Enrich Unique_Sites with peptide context + modified peptide variants, MS3-first with MS2 fallback.
Then propagate the same peptide columns to hit lists.

Refactor goals:
- No work at import time
- Expose run(...) for Streamlit wizard
- Keep behavior identical to your original script

Expected sheets in input workbook:
  - Unique_Sites
  - Hits_MS2
  - Hits_MS3
  - Hits_Consistent_MS2_MS3
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

LogFn = Callable[[str], None]


# -----------------------
# Config / IO models
# -----------------------

@dataclass(frozen=True)
class PeptideSearchConfig:
    unique_sites_sheet: str = "Unique_Sites"
    hit_sheets: Tuple[str, ...] = ("Hits_MS2", "Hits_MS3", "Hits_Consistent_MS2_MS3")

    uniprot_col: str = "Uniprot ID"
    site_col: str = "Site"

    # peptide table columns
    pt_protein_col: str = "Protein ID"
    pt_start_col: str = "Start"
    pt_end_col: str = "End"
    pt_peptide_col: str = "Peptide Sequence"

    peptide_table_sheet_if_excel: str = "combined_peptide"  # if table is .xlsx/.xls

    # join behavior when enriching hit sheets
    join_on_uniprot_when_available: bool = True

    # output naming (if caller doesn't provide explicit output path)
    output_suffix: str = "_sites"


@dataclass(frozen=True)
class PeptideSearchInputs:
    input_xlsx: Path

    # MS3 first, MS2 fallback
    peptide_table_ms3: Path
    peptide_list_ms3: Path

    peptide_table_ms2: Path
    peptide_list_ms2: Path


@dataclass(frozen=True)
class PeptideSearchOutputs:
    output_xlsx: Path
    n_unique_sites: int
    n_hits_sheets_enriched: int


def _log(msg: str, log: Optional[LogFn]) -> None:
    if log:
        log(msg)
    else:
        print(msg)


# -----------------------
# PEPTIDE MAPPING HELPERS
# -----------------------

def extract_resnum_from_site(site: str) -> Optional[int]:
    """Take token after space (e.g. 'C120') and extract digits (e.g. 120)."""
    if site is None or (isinstance(site, float) and pd.isna(site)):
        return None
    parts = str(site).strip().split()
    if len(parts) < 2:
        return None
    res_token = parts[1]
    m = re.search(r"(\d+)", res_token)
    return int(m.group(1)) if m else None


def load_combined_peptide_table(path: Path, sheet_if_excel: str) -> pd.DataFrame:
    """Load TSV (default) or Excel."""
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_if_excel, engine="openpyxl")
    return pd.read_csv(path, sep="\t")


def clean_modified_peptide_line(line: str) -> str:
    """Strip modifications: keep only uppercase A–Z letters."""
    return "".join(re.findall(r"[A-Z]", line))


def parse_peptide_list(txt_path: Path) -> Dict[str, List[Tuple[str, str]]]:
    """
    Map uniprot accession -> list of (raw_modified_line, cleaned_sequence)
    Header format expected: >>sp|Q01469|FABP5_HUMAN
    """
    mapping: Dict[str, List[Tuple[str, str]]] = {}
    current_uniprot: Optional[str] = None
    header_re = re.compile(r"^>>\s*\w+\|([A-Z0-9]+)\|")

    with txt_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">>"):
                m = header_re.match(line)
                current_uniprot = m.group(1) if m else None
                if current_uniprot and current_uniprot not in mapping:
                    mapping[current_uniprot] = []
                continue
            if current_uniprot:
                cleaned = clean_modified_peptide_line(line)
                if cleaned:
                    mapping[current_uniprot].append((line, cleaned))
    return mapping


def find_modified_matches(
    peptide_map: Dict[str, List[Tuple[str, str]]],
    uniprot_id: str,
    peptide_seq: str,
) -> List[str]:
    """Return raw modified peptide lines where cleaned sequence == peptide_seq."""
    if not uniprot_id or not peptide_seq:
        return []
    return [raw for (raw, cleaned) in peptide_map.get(uniprot_id, []) if cleaned == peptide_seq]


def choose_best_peptide(peptides_with_ranges: List[Tuple[str, int, int]], resnum: int) -> Tuple[str, int, int]:
    """
    Pick a single "best" peptide per site.
    Heuristic:
      1) shortest peptide length
      2) smallest protein-span (End-Start)
      3) smallest distance of residue from peptide center
    """
    def key_fn(t: Tuple[str, int, int]):
        pep, start, end = t
        pep_len = len(pep)
        span = end - start
        center = (start + end) / 2.0
        center_dist = abs(resnum - center)
        return (pep_len, span, center_dist)

    return sorted(peptides_with_ranges, key=key_fn)[0]


def prep_peptide_table(
    df: pd.DataFrame,
    source_name: str,
    *,
    pt_protein_col: str,
    pt_start_col: str,
    pt_end_col: str,
    pt_peptide_col: str,
) -> Dict[str, pd.DataFrame]:
    """Validate columns + numeric Start/End + group by protein."""
    for col in [pt_protein_col, pt_start_col, pt_end_col, pt_peptide_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {source_name}. Columns: {list(df.columns)}")

    df = df.copy()
    df[pt_start_col] = pd.to_numeric(df[pt_start_col], errors="coerce")
    df[pt_end_col] = pd.to_numeric(df[pt_end_col], errors="coerce")
    df = df.dropna(subset=[pt_protein_col, pt_start_col, pt_end_col, pt_peptide_col])

    return {pid: sub.copy() for pid, sub in df.groupby(pt_protein_col)}


def map_one_site(
    uniprot: str,
    resnum: int,
    pt_by_protein: Dict[str, pd.DataFrame],
    peptide_map: Dict[str, List[Tuple[str, str]]],
    *,
    pt_start_col: str,
    pt_end_col: str,
    pt_peptide_col: str,
) -> Tuple[str, Optional[int], Optional[int], str, str, str]:
    """
    Return:
      best_peptide, best_start, best_end, all_peptides_str, all_modpep_str, note
    """
    sub = pt_by_protein.get(uniprot)
    if sub is None:
        return "", None, None, "", "", "No rows in combined_peptide for this Protein ID"

    sub2 = sub[(sub[pt_start_col] <= resnum) & (sub[pt_end_col] >= resnum)].copy()
    if sub2.empty:
        return "", None, None, "", "", "Protein matched, but residue not within any Start/End range"

    peptides_with_ranges: List[Tuple[str, int, int]] = []
    for _, r2 in sub2.iterrows():
        pep = str(r2[pt_peptide_col]).strip()
        start = int(r2[pt_start_col])
        end = int(r2[pt_end_col])
        if pep:
            peptides_with_ranges.append((pep, start, end))

    peptides_with_ranges = sorted(set(peptides_with_ranges), key=lambda x: (x[1], x[2], x[0]))
    best_pep, best_start, best_end = choose_best_peptide(peptides_with_ranges, resnum)

    all_peps_unique = [p for (p, _, _) in peptides_with_ranges]
    all_peps_str = "; ".join(all_peps_unique)

    all_mod_matches: List[str] = []
    for pep in all_peps_unique:
        all_mod_matches.extend(find_modified_matches(peptide_map, uniprot, pep))

    # De-dup preserving order
    seen = set()
    all_mod_unique = []
    for m in all_mod_matches:
        if m not in seen:
            seen.add(m)
            all_mod_unique.append(m)

    all_mod_str = "; ".join(all_mod_unique)
    note = "" if all_mod_unique else "No matching modified peptide lines in peptide_list (after cleaning)"
    return best_pep, best_start, best_end, all_peps_str, all_mod_str, note


# -----------------------
# Core enrichment logic
# -----------------------

def enrich_workbook(
    sheets: Dict[str, pd.DataFrame],
    *,
    cfg: PeptideSearchConfig,
    pt_ms3: Dict[str, pd.DataFrame],
    pepmap_ms3: Dict[str, List[Tuple[str, str]]],
    pt_ms2: Dict[str, pd.DataFrame],
    pepmap_ms2: Dict[str, List[Tuple[str, str]]],
    log: Optional[LogFn] = None,
) -> Tuple[Dict[str, pd.DataFrame], int]:
    """Return updated sheets dict + number of hit sheets enriched."""
    if cfg.unique_sites_sheet not in sheets:
        raise ValueError(
            f"Sheet '{cfg.unique_sites_sheet}' not found. Sheets: {list(sheets.keys())}"
        )

    df = sheets[cfg.unique_sites_sheet].copy()

    for col in [cfg.uniprot_col, cfg.site_col]:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found in {cfg.unique_sites_sheet}. Columns: {list(df.columns)}"
            )

    # Residue #
    df["Residue #"] = df[cfg.site_col].map(extract_resnum_from_site)

    # Enrichment columns
    best_peptides: List[str] = []
    best_starts: List[Optional[int]] = []
    best_ends: List[Optional[int]] = []
    all_peptides_col: List[str] = []
    all_modpep_col: List[str] = []
    map_notes: List[str] = []
    peptide_source_used: List[str] = []
    modified_source_used: List[str] = []

    for _, row in df.iterrows():
        uniprot = str(row.get(cfg.uniprot_col, "")).strip()
        site = str(row.get(cfg.site_col, "")).strip()
        resnum = row.get("Residue #", None)

        # Defaults
        best_pep = ""
        bstart = None
        bend = None
        all_peps = ""
        all_mods = ""
        note = ""
        used_pep_src = ""
        used_mod_src = ""

        if (not uniprot) or (uniprot.lower() == "nan"):
            note = "Missing Uniprot ID"
        elif pd.isna(resnum):
            note = f"Could not parse residue # from Site='{site}'"
        else:
            resnum = int(resnum)

            # Try MS3 first
            best_pep, bstart, bend, all_peps, all_mods, note = map_one_site(
                uniprot,
                resnum,
                pt_ms3,
                pepmap_ms3,
                pt_start_col=cfg.pt_start_col,
                pt_end_col=cfg.pt_end_col,
                pt_peptide_col=cfg.pt_peptide_col,
            )
            used_pep_src = "MS3" if (best_pep or all_peps) else ""
            used_mod_src = "MS3" if all_mods else ""

            # If MS3 found no modified peptide lines, try MS2
            if not all_mods:
                best_pep2, bstart2, bend2, all_peps2, all_mods2, note2 = map_one_site(
                    uniprot,
                    resnum,
                    pt_ms2,
                    pepmap_ms2,
                    pt_start_col=cfg.pt_start_col,
                    pt_end_col=cfg.pt_end_col,
                    pt_peptide_col=cfg.pt_peptide_col,
                )

                # Use MS2 if it improves things (finds mods or at least maps peptides if MS3 couldn't)
                if all_mods2 or (not best_pep and (best_pep2 or all_peps2)):
                    best_pep, bstart, bend, all_peps, all_mods, note = (
                        best_pep2, bstart2, bend2, all_peps2, all_mods2, note2
                    )
                    used_pep_src = "MS2" if (best_pep or all_peps) else used_pep_src
                    used_mod_src = "MS2" if all_mods else used_mod_src

        best_peptides.append(best_pep)
        best_starts.append(bstart)
        best_ends.append(bend)
        all_peptides_col.append(all_peps)
        all_modpep_col.append(all_mods)
        map_notes.append(note)
        peptide_source_used.append(used_pep_src)
        modified_source_used.append(used_mod_src)

    # Attach enrichment columns to Unique_Sites
    df["Best Peptide"] = best_peptides
    df["Peptide Start"] = best_starts
    df["Peptide End"] = best_ends
    df["All Peptides"] = all_peptides_col
    df["Matching modified peptides"] = all_modpep_col
    df["Peptide mapping notes"] = map_notes
    df["Peptide source used"] = peptide_source_used
    df["Modified source used"] = modified_source_used

    sheets = dict(sheets)
    sheets[cfg.unique_sites_sheet] = df

    # Build lookup table from Unique_Sites to enrich hit sheets
    propagate_cols = [
        cfg.site_col,
        cfg.uniprot_col,
        "Residue #",
        "Best Peptide",
        "Peptide Start",
        "Peptide End",
        "All Peptides",
        "Matching modified peptides",
        "Peptide mapping notes",
        "Peptide source used",
        "Modified source used",
    ]
    lookup = df[[c for c in propagate_cols if c in df.columns]].copy()
    lookup[cfg.site_col] = lookup[cfg.site_col].astype(str).str.strip()
    if cfg.uniprot_col in lookup.columns:
        lookup[cfg.uniprot_col] = lookup[cfg.uniprot_col].astype(str).str.strip()

    n_enriched = 0
    for hs in cfg.hit_sheets:
        if hs not in sheets:
            continue

        hdf = sheets[hs].copy()
        if cfg.site_col not in hdf.columns:
            sheets[hs] = hdf
            continue

        hdf[cfg.site_col] = hdf[cfg.site_col].astype(str).str.strip()

        use_uniprot = (
            cfg.join_on_uniprot_when_available
            and (cfg.uniprot_col in hdf.columns)
            and (cfg.uniprot_col in lookup.columns)
        )

        if use_uniprot:
            hdf[cfg.uniprot_col] = hdf[cfg.uniprot_col].astype(str).str.strip()
            merged = hdf.merge(
                lookup,
                how="left",
                on=[cfg.site_col, cfg.uniprot_col],
                suffixes=("", "_from_unique"),
            )
        else:
            merged = hdf.merge(
                lookup.drop(columns=[cfg.uniprot_col], errors="ignore"),
                how="left",
                on=[cfg.site_col],
                suffixes=("", "_from_unique"),
            )

        sheets[hs] = merged
        n_enriched += 1

    _log(f"[peptide_search] Enriched Unique_Sites + {n_enriched} hit sheets.", log)
    return sheets, n_enriched


def read_workbook_all_sheets(xlsx_path: Path) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    return {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}


def write_workbook_all_sheets(out_path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, sdf in sheets.items():
            sdf.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)


# -----------------------
# Public run() entry point
# -----------------------

def run(
    inputs: PeptideSearchInputs,
    out_dir: Path,
    *,
    cfg: Optional[PeptideSearchConfig] = None,
    output_filename: Optional[str] = None,
    log: Optional[LogFn] = None,
) -> PeptideSearchOutputs:
    """
    Streamlit-friendly entry point.

    - Reads the input workbook
    - Loads peptide tables + peptide lists (MS3 and MS2)
    - Enriches Unique_Sites and hit sheets
    - Writes output workbook into out_dir
    """
    cfg = cfg or PeptideSearchConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not inputs.input_xlsx.exists():
        raise FileNotFoundError(f"Missing input workbook: {inputs.input_xlsx}")

    for p in [inputs.peptide_table_ms3, inputs.peptide_list_ms3, inputs.peptide_table_ms2, inputs.peptide_list_ms2]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    _log(f"[peptide_search] Reading workbook: {inputs.input_xlsx}", log)
    sheets = read_workbook_all_sheets(inputs.input_xlsx)

    # Load sources
    _log("[peptide_search] Loading MS3 peptide table + list...", log)
    pt_ms3 = prep_peptide_table(
        load_combined_peptide_table(inputs.peptide_table_ms3, cfg.peptide_table_sheet_if_excel),
        inputs.peptide_table_ms3.name,
        pt_protein_col=cfg.pt_protein_col,
        pt_start_col=cfg.pt_start_col,
        pt_end_col=cfg.pt_end_col,
        pt_peptide_col=cfg.pt_peptide_col,
    )
    pepmap_ms3 = parse_peptide_list(inputs.peptide_list_ms3)

    _log("[peptide_search] Loading MS2 peptide table + list...", log)
    pt_ms2 = prep_peptide_table(
        load_combined_peptide_table(inputs.peptide_table_ms2, cfg.peptide_table_sheet_if_excel),
        inputs.peptide_table_ms2.name,
        pt_protein_col=cfg.pt_protein_col,
        pt_start_col=cfg.pt_start_col,
        pt_end_col=cfg.pt_end_col,
        pt_peptide_col=cfg.pt_peptide_col,
    )
    pepmap_ms2 = parse_peptide_list(inputs.peptide_list_ms2)

    # Enrich
    sheets2, n_enriched = enrich_workbook(
        sheets,
        cfg=cfg,
        pt_ms3=pt_ms3,
        pepmap_ms3=pepmap_ms3,
        pt_ms2=pt_ms2,
        pepmap_ms2=pepmap_ms2,
        log=log,
    )

    # Output name
    if output_filename:
        out_path = out_dir / output_filename
    else:
        out_path = out_dir / f"{inputs.input_xlsx.stem}{cfg.output_suffix}{inputs.input_xlsx.suffix}"

    write_workbook_all_sheets(out_path, sheets2)
    _log(f"[peptide_search] ✅ Wrote enriched workbook: {out_path}", log)

    n_unique = len(sheets2.get(cfg.unique_sites_sheet, pd.DataFrame()))
    return PeptideSearchOutputs(output_xlsx=out_path, n_unique_sites=int(n_unique), n_hits_sheets_enriched=int(n_enriched))


# -----------------------
# Optional CLI wrapper (keeps original behavior)
# -----------------------

def main() -> None:
    base_dir = Path(__file__).resolve().parent

    cfg = PeptideSearchConfig()

    inputs = PeptideSearchInputs(
        input_xlsx=base_dir / "merged_log2fc_results.xlsx",
        peptide_table_ms3=base_dir / "combined_peptide_MS3.tsv",
        peptide_list_ms3=base_dir / "peptide_list_MS3.txt",
        peptide_table_ms2=base_dir / "combined_peptide_MS2.tsv",
        peptide_list_ms2=base_dir / "peptide_list_MS2.txt",
    )

    out = run(inputs, out_dir=base_dir, cfg=cfg, log=print)
    print(f"✅ Done: {out.output_xlsx}")


if __name__ == "__main__":
    main()
