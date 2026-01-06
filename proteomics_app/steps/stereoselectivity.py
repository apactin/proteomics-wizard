#!/usr/bin/env python3
"""
stereoselectivity.py

Streamlit-friendly refactor of your stereoselectivity analysis.

Key behavior preserved:
- Works on merged wide-format XLSX with per-file columns like:
    MS3_0s_Log2FC, MS3_0s_p.adj, MS2_5s_Log2FC, ...
- n_DE_{mode} is number of raw non-NaN windows present (independent of sign cleaning).
- Adaptive hit filtering based on windows available in dataset:
    require ceil(n_windows_available / 2)
- UniProt enrichment via rest.uniprot.org with on-disk JSON cache.

Enhancement added:
- "Rich" UniProt caching:
    - caches raw entry JSON:   <cache_root>/uniprot_raw/<ID>.json
    - caches summary JSON:     <cache_root>/uniprot_summary/<ID>.json
- Writes per-run UniProt annotation artifact:
    - <out_dir>/uniprot_annotations.jsonl
    - (optional) <out_dir>/uniprot_annotations.parquet

This module exposes:
  run(inputs, out_dir, ...) -> Outputs
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from openpyxl.utils import get_column_letter
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LogFn = Callable[[str], None]


# ---------------- Configuration ----------------

@dataclass(frozen=True)
class StereoConfig:
    de_windows: Tuple[str, ...] = ("0s", "5s", "30s")
    debug: bool = True

    # thresholds for hits
    mu_abs_min: float = 0.7
    sd_max: float = 0.8

    # S-score factors
    cov_norm_denominator: float = 3.0     # for f_cov(n): min(n/denom, 1)
    sign_flip_penalty: float = 0.3        # f_conf(True)=penalty, False=1

    # UniProt
    uniprot_timeout_s: int = 12
    uniprot_retry_total: int = 4
    uniprot_backoff: float = 0.5

    # Rich UniProt annotation
    uniprot_rich: bool = True
    uniprot_save_raw_json: bool = True
    uniprot_write_run_artifact: bool = True
    # Excel-friendly joiner for list-like fields
    uniprot_list_joiner: str = "; "


@dataclass(frozen=True)
class StereoInputs:
    input_xlsx: Path
    input_sheet: str | int = 0   # 0 or sheet name


@dataclass(frozen=True)
class StereoOutputs:
    output_xlsx: Path
    n_input_rows: int
    n_unique_rows: int
    n_hits_ms2: int
    n_hits_ms3: int
    n_hits_consistent: int


# ---------------- Small utilities ----------------

def _dbg(msg: str, cfg: StereoConfig, log: Optional[LogFn]) -> None:
    if cfg.debug:
        if log:
            log(msg)
        else:
            print(msg)

def _autodetect_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    norm = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in norm:
            return norm[key]
    return None

def _coerce_numeric_inplace(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def _make_session(cfg: StereoConfig) -> requests.Session:
    s = requests.Session()
    r = Retry(
        total=cfg.uniprot_retry_total,
        backoff_factor=cfg.uniprot_backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.headers["User-Agent"] = "stereoselectivity-lite/2.1"
    return s

def _cache_json(path: Path, obj: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    except Exception:
        pass

def _load_cache(path: Path) -> Optional[dict]:
    try:
        return json.load(open(path, encoding="utf-8")) if path.exists() else None
    except Exception:
        return None

def _safe_get(d: object, *keys: str) -> Optional[object]:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur

def _uniq_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        x = (x or "").strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# ---------------- Core analysis helpers ----------------

def clean_and_majority(vals: Sequence[object]) -> Tuple[np.ndarray, bool]:
    """
    Remove outlier sign flips: if a majority sign exists, keep only that sign.
    Returns (cleaned_values, sign_flip_flag).
    sign_flip_flag is True if signs are ambiguous/mixed (no clear majority).
    """
    arr = np.array([v for v in vals if pd.notna(v)], dtype=float)
    if len(arr) < 2:
        return arr, False

    signs = np.sign(arr)
    pos = np.sum(signs > 0)
    neg = np.sum(signs < 0)

    if pos != neg:
        majority_sign = 1 if pos > neg else -1
        cleaned = arr[signs == majority_sign]
        return cleaned, False
    else:
        return arr, True


def get_mode_columns(df_in: pd.DataFrame, mode: str, suffix: str, de_windows: Sequence[str]) -> List[str]:
    """
    Collect columns for a given mode and suffix in DE_WINDOWS order.
    Example: mode="MS3", suffix="Log2FC" -> ["MS3_0s_Log2FC", "MS3_5s_Log2FC", ...]
    """
    cols = []
    for de in de_windows:
        c = f"{mode}_{de}_{suffix}"
        if c in df_in.columns:
            cols.append(c)
    return cols


def required_windows_present(n_windows_available: int) -> int:
    """Adaptive threshold: ceil(n/2)."""
    if n_windows_available <= 0:
        return 0
    return int(math.ceil(n_windows_available / 2.0))


def _f_sd(sd: float) -> float:
    return 1 / (1 + sd) if pd.notna(sd) else np.nan

def _f_cov(n: int, denom: float) -> float:
    return min(n / denom, 1.0)

def _f_conf(flip: bool, penalty: float) -> float:
    return penalty if flip else 1.0


# ---------------- UniProt enrichment (rich caching) ----------------

def _extract_uniprot_disulfides(entry_json: dict) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for feat in entry_json.get("features", []) or []:
        ftype = str(feat.get("type", "")).lower()
        if ftype != "disulfide bond":
            continue
        loc = feat.get("location", {}) or {}
        start = (
            _safe_get(loc, "start", "value")
            or _safe_get(loc, "begin", "value")
            or loc.get("begin")
            or loc.get("start")
        )
        end = (
            _safe_get(loc, "end", "value")
            or _safe_get(loc, "finish", "value")
            or loc.get("end")
            or loc.get("finish")
        )
        try:
            if isinstance(start, dict):
                start = start.get("value")
            if isinstance(end, dict):
                end = end.get("value")
            if start and end:
                pairs.append((int(start), int(end)))
        except Exception:
            continue
    return pairs


def _extract_uniprot_rich_summary(pid: str, entry_json: dict, seq: Optional[str], pairs: List[Tuple[int, int]]) -> Dict:
    # protein name
    protein_name = (
        _safe_get(entry_json, "proteinDescription", "recommendedName", "fullName", "value")
        or _safe_get(entry_json, "proteinDescription", "submissionNames", 0, "fullName", "value")  # type: ignore
        or _safe_get(entry_json, "proteinDescription", "alternativeNames", 0, "fullName", "value")  # type: ignore
        or None
    )
    if isinstance(protein_name, dict):
        protein_name = protein_name.get("value")

    # genes
    gene_names: List[str] = []
    for g in entry_json.get("genes", []) or []:
        gn = _safe_get(g, "geneName", "value")
        if isinstance(gn, str):
            gene_names.append(gn)
        for syn in g.get("synonyms", []) or []:
            sv = _safe_get(syn, "value")
            if isinstance(sv, str):
                gene_names.append(sv)

    gene_names = _uniq_keep_order(gene_names)

    # organism
    organism = _safe_get(entry_json, "organism", "scientificName")
    if not isinstance(organism, str):
        organism = None

    # keywords
    keywords: List[str] = []
    for kw in entry_json.get("keywords", []) or []:
        name = kw.get("name")
        if isinstance(name, str):
            keywords.append(name)
    keywords = _uniq_keep_order(keywords)

    # GO terms (often appear in UniProt cross-references)
    go_bp: List[str] = []
    go_mf: List[str] = []
    go_cc: List[str] = []

    # UniProt REST JSON schema has varied a bit over time; try both common locations
    xrefs = entry_json.get("uniProtKBCrossReferences") or entry_json.get("dbReferences") or []
    if isinstance(xrefs, list):
        for xr in xrefs:
            db = xr.get("database") or xr.get("type")
            if str(db).upper() != "GO":
                continue
            props = xr.get("properties") or {}
            # properties can be dict or list of dicts depending on schema
            term = None
            if isinstance(props, dict):
                term = props.get("GoTerm") or props.get("term")
            elif isinstance(props, list):
                for p in props:
                    if p.get("key") in ("GoTerm", "term"):
                        term = p.get("value")
                        break
            if not isinstance(term, str):
                continue

            # GO term looks like: "P:..."; "F:..."; "C:..."
            if term.startswith("P:"):
                go_bp.append(term[2:].strip())
            elif term.startswith("F:"):
                go_mf.append(term[2:].strip())
            elif term.startswith("C:"):
                go_cc.append(term[2:].strip())
            else:
                # if unknown, keep in BP as catch-all
                go_bp.append(term.strip())

    go_bp = _uniq_keep_order(go_bp)
    go_mf = _uniq_keep_order(go_mf)
    go_cc = _uniq_keep_order(go_cc)

    # Subcellular location + pathway + function-ish notes commonly in comments
    subcell: List[str] = []
    pathways: List[str] = []
    families: List[str] = []
    ec_numbers: List[str] = []

    # EC numbers sometimes in recommendedName
    ec_list = _safe_get(entry_json, "proteinDescription", "recommendedName", "ecNumbers")
    if isinstance(ec_list, list):
        for ec in ec_list:
            ev = ec.get("value")
            if isinstance(ev, str):
                ec_numbers.append(ev)

    for c in entry_json.get("comments", []) or []:
        ctype = str(c.get("commentType", "")).upper()

        if ctype == "SUBCELLULAR LOCATION":
            # Schema can include "subcellularLocations": [{"location":{"value":...}, ...}]
            scl = c.get("subcellularLocations") or []
            if isinstance(scl, list):
                for item in scl:
                    loc = _safe_get(item, "location", "value")
                    if isinstance(loc, str):
                        subcell.append(loc)

        elif ctype == "PATHWAY":
            # Schema can include "pathways": [{"name": {"value": ...}}] or "text": [...]
            pls = c.get("pathways") or []
            if isinstance(pls, list):
                for p in pls:
                    nm = _safe_get(p, "name", "value") or p.get("name")
                    if isinstance(nm, str):
                        pathways.append(nm)
            # some schema uses "text" blocks
            txt = c.get("texts") or c.get("text") or []
            if isinstance(txt, list):
                for t in txt:
                    tv = t.get("value")
                    if isinstance(tv, str) and "pathway" in tv.lower():
                        pathways.append(tv)

        elif ctype == "SIMILARITY":
            # often contains protein family mentions in free text
            txt = c.get("texts") or c.get("text") or []
            if isinstance(txt, list):
                for t in txt:
                    tv = t.get("value")
                    if isinstance(tv, str):
                        # keep the whole line; stats viewer can tokenize later
                        families.append(tv)

    subcell = _uniq_keep_order(subcell)
    pathways = _uniq_keep_order(pathways)
    families = _uniq_keep_order(families)
    ec_numbers = _uniq_keep_order(ec_numbers)

    # sequence-derived
    aa_len = len(seq) if isinstance(seq, str) else None
    c_count = seq.count("C") if isinstance(seq, str) else None

    disulfide_bonds = len(pairs)
    disulfide_c = disulfide_bonds * 2 if disulfide_bonds else 0
    c_frac = (c_count / aa_len) if aa_len and c_count is not None else None
    pairs_str = ", ".join(f"Cys{a}–Cys{b}" for a, b in pairs) if pairs else "None"

    # Return a stable, small summary schema
    return {
        "UniProt_ID": pid,
        "UniProt_Protein_Name": protein_name,
        "UniProt_Gene_Names": gene_names,  # list
        "UniProt_Organism": organism,
        "UniProt_Keywords": keywords,      # list
        "UniProt_GO_BP": go_bp,            # list
        "UniProt_GO_MF": go_mf,            # list
        "UniProt_GO_CC": go_cc,            # list
        "UniProt_Subcellular_Location": subcell,  # list
        "UniProt_EC": ec_numbers,          # list
        "UniProt_Pathway": pathways,       # list
        "UniProt_Family": families,        # list (free-text lines)
        # existing numeric features you already used
        "AA_len": aa_len,
        "Cys_count": c_count,
        "Disulfide_Bond_Count": disulfide_bonds,
        "Disulfide_Cys_Count": disulfide_c,
        "Disulfide_Pairs_String": pairs_str,
        "Cys_fraction": c_frac,
    }


def fetch_uniprot(
    pid: str,
    *,
    session: requests.Session,
    cache_root: Path,
    timeout_s: int,
    cfg: StereoConfig,
    log: Optional[LogFn] = None,
) -> Dict:
    """
    Fetch UniProt info with robust caching.

    Cache layout:
      <cache_root>/uniprot_summary/<pid>.json   (stable schema; what we merge/use)
      <cache_root>/uniprot_raw/<pid>.json       (raw UniProt entry json; optional)
    """
    pid = str(pid).strip()
    if not pid:
        return {}

    summary_file = cache_root / "uniprot_summary" / f"{pid}.json"
    cached_summary = _load_cache(summary_file)
    if cached_summary:
        return cached_summary

    raw_file = cache_root / "uniprot_raw" / f"{pid}.json"

    fasta_url = f"https://rest.uniprot.org/uniprotkb/{pid}.fasta"
    json_url = f"https://rest.uniprot.org/uniprotkb/{pid}"

    seq: Optional[str] = None
    entry_json: dict = {}
    pairs: List[Tuple[int, int]] = []

    # FASTA for sequence-derived metrics
    try:
        r = session.get(fasta_url, headers={"Accept": "text/plain"}, timeout=timeout_s)
        if r.ok and ">" in r.text:
            seq = "".join(line.strip() for line in r.text.splitlines() if not line.startswith(">"))
    except Exception as e:
        if log:
            log(f"⚠️ FASTA fetch failed for {pid}: {e}")

    # JSON entry for rich annotations + disulfides
    try:
        r = session.get(json_url, headers={"Accept": "application/json"}, timeout=timeout_s)
        if r.ok:
            entry_json = r.json()
            if cfg.uniprot_save_raw_json:
                _cache_json(raw_file, entry_json)
            pairs = _extract_uniprot_disulfides(entry_json)
        else:
            if log:
                log(f"⚠️ UniProt JSON HTTP {r.status_code} for {pid}")
    except Exception as e:
        if log:
            log(f"⚠️ JSON fetch failed for {pid}: {e}")

    # Build stable summary
    if cfg.uniprot_rich and entry_json:
        summary = _extract_uniprot_rich_summary(pid, entry_json, seq, pairs)
    else:
        # minimal fallback (original behavior)
        aa_len = len(seq) if seq else None
        c_count = seq.count("C") if seq else None
        disulfide_bonds = len(pairs)
        disulfide_c = disulfide_bonds * 2 if disulfide_bonds else 0
        c_frac = (c_count / aa_len) if aa_len and c_count else None
        pairs_str = ", ".join(f"Cys{a}–Cys{b}" for a, b in pairs) if pairs else "None"
        summary = {
            "UniProt_ID": pid,
            "AA_len": aa_len,
            "Cys_count": c_count,
            "Disulfide_Bond_Count": disulfide_bonds,
            "Disulfide_Cys_Count": disulfide_c,
            "Disulfide_Pairs_String": pairs_str,
            "Cys_fraction": c_frac,
        }

    _cache_json(summary_file, summary)
    return summary


def _resolve_cache_root(cache_dir: Optional[Path]) -> Path:
    """
    Priority:
      1) explicit cache_dir arg
      2) UNIPROT_CACHE_DIR env
      3) PROTEOMICS_CACHE_DIR env (uses <that>/uniprot)
      4) ~/.stereoselectivity_cache
    """
    if cache_dir is not None:
        return Path(cache_dir)

    env_uniprot = os.environ.get("UNIPROT_CACHE_DIR", "").strip()
    if env_uniprot:
        return Path(env_uniprot)

    env_root = os.environ.get("PROTEOMICS_CACHE_DIR", "").strip()
    if env_root:
        return Path(env_root) / "uniprot"

    return Path.home() / ".stereoselectivity_cache"


def _write_uniprot_run_artifacts(out_dir: Path, df_u: pd.DataFrame, log: Optional[LogFn]) -> None:
    """
    Writes:
      - uniprot_annotations.jsonl
      - (optional) uniprot_annotations.parquet
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSONL: always available
    jsonl_path = out_dir / "uniprot_annotations.jsonl"
    try:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in df_u.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if log:
            log(f"✅ Wrote UniProt run artifact: {jsonl_path}")
    except Exception as e:
        if log:
            log(f"⚠️ Failed writing {jsonl_path}: {e}")

    # Parquet: best-effort
    parquet_path = out_dir / "uniprot_annotations.parquet"
    try:
        df_u.to_parquet(parquet_path, index=False)
        if log:
            log(f"✅ Wrote UniProt run artifact: {parquet_path}")
    except Exception:
        # silently ignore if parquet engine not installed
        pass


# ---------------- Excel formatting ----------------

def autosize_columns(worksheet, df_sheet: pd.DataFrame, min_width: int = 10, max_width: int = 50) -> None:
    """Auto-fit columns based on longest entry or header. Robust to duplicate column names."""
    for i, col_name in enumerate(df_sheet.columns, 1):
        column_letter = get_column_letter(i)
        series = df_sheet.iloc[:, i - 1]
        values = series.astype(str).fillna("").tolist()
        max_length = max([len(str(col_name))] + [len(v) for v in values]) if values else len(str(col_name))
        adjusted_width = min(max(max_length + 2, min_width), max_width)
        worksheet.column_dimensions[column_letter].width = adjusted_width


# ---------------- Main run() entry point ----------------

def run(
    inputs: StereoInputs,
    out_dir: Path,
    *,
    cfg: Optional[StereoConfig] = None,
    cache_dir: Optional[Path] = None,
    output_filename: str = "merged_log2fc_results.xlsx",
    log: Optional[LogFn] = None,
) -> StereoOutputs:
    cfg = cfg or StereoConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_root = _resolve_cache_root(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    session = _make_session(cfg)

    # ---------- Load ----------
    df = pd.read_excel(inputs.input_xlsx, sheet_name=inputs.input_sheet)
    df = df.replace(" ", np.nan)
    df.columns = [c.strip().replace("\u00A0", " ") for c in df.columns]

    _dbg("Columns read from Excel:", cfg, log)
    for c in df.columns:
        _dbg(f"  - '{c}'", cfg, log)
    _dbg(f"Loaded data: {df.shape}", cfg, log)

    # Coerce numeric columns
    for c in df.columns:
        if c.endswith("_Log2FC") or c.endswith("_p.adj"):
            _coerce_numeric_inplace(df, c)

    # ---------- Per-mode metrics ----------
    for mode in ["MS2", "MS3"]:
        lfc_cols = get_mode_columns(df, mode, "Log2FC", cfg.de_windows)
        if not lfc_cols:
            _dbg(f"[WARN] No Log2FC columns found for {mode}. Skipping mode.", cfg, log)
            continue

        mus: List[float] = []
        sds: List[float] = []
        flips: List[bool] = []
        ns: List[int] = []

        for _, row in df[lfc_cols].iterrows():
            vals = row.values
            cleaned, flip = clean_and_majority(vals)

            if len(cleaned) > 0:
                mus.append(float(np.nanmean(cleaned)))
                sds.append(float(np.nanstd(cleaned, ddof=1)) if len(cleaned) > 1 else 0.0)
            else:
                mus.append(np.nan)
                sds.append(np.nan)

            flips.append(bool(flip))
            ns.append(int(np.sum(pd.notna(vals))))  # raw presence

        df[f"mu_site_{mode}"] = mus
        df[f"sd_site_{mode}"] = sds
        df[f"sign_flip_{mode}"] = flips
        df[f"n_DE_{mode}"] = ns

    # ---------- S-score ----------
    for mode in ["MS2", "MS3"]:
        mu_col = f"mu_site_{mode}"
        if mu_col not in df.columns:
            continue
        df[f"S_site_{mode}"] = (
            df[mu_col].abs()
            * df[f"sd_site_{mode}"].apply(_f_sd)
            * df[f"n_DE_{mode}"].apply(lambda n: _f_cov(n, cfg.cov_norm_denominator))
            * df[f"sign_flip_{mode}"].apply(lambda f: _f_conf(bool(f), cfg.sign_flip_penalty))
        )

    # ---------- Collapse duplicates ----------
    gene_col = _autodetect_column(df, {"gene name", "gene"})
    site_col = _autodetect_column(df, {"site"})
    uniprot_col = _autodetect_column(df, {"uniprot id", "uniprot", "accession"})

    if not site_col:
        raise SystemExit("No Site column found (expected something like 'Site').")

    group_cols = [gene_col, site_col] if gene_col else [site_col]

    lfc_cols_existing = [c for c in df.columns if c.endswith("_Log2FC")]
    padj_cols_existing = [c for c in df.columns if c.endswith("_p.adj")]

    extra_meta_cols_existing = [c for c in df.columns if c in [
        "Predicted_note", "Predicted_peak_split", "Predicted_intensity_ratio"
    ]]

    agg: Dict[str, object] = {}

    # computed metrics
    for mode in ["MS2", "MS3"]:
        for col, fn in [
            (f"mu_site_{mode}", "mean"),
            (f"sd_site_{mode}", "median"),
            (f"S_site_{mode}", "mean"),
            (f"n_DE_{mode}", "max"),
            (f"sign_flip_{mode}", "any"),
        ]:
            if col in df.columns:
                agg[col] = fn

    # per-window values
    for c in lfc_cols_existing:
        agg[c] = "mean"
    for c in padj_cols_existing:
        agg[c] = "mean"
    for c in extra_meta_cols_existing:
        agg[c] = "first"

    df_unique = df.groupby(group_cols, as_index=False).agg(agg)

    # Carry UniProt ID through collapse (mode / most frequent per site)
    if uniprot_col:
        common_uniprot = (
            df[group_cols + [uniprot_col]]
            .dropna(subset=[uniprot_col])
            .groupby(group_cols)[uniprot_col]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
            .reset_index()
        )
        df_unique = df_unique.merge(common_uniprot, on=group_cols, how="left")

    _dbg(f"Collapsed {len(df)} → {len(df_unique)}", cfg, log)

    # ---------- UniProt enrichment ----------
    uniprot_col_unique = _autodetect_column(df_unique, {"uniprot id", "uniprot", "accession"})
    uinfo_df = None

    if uniprot_col_unique:
        ids = df_unique[uniprot_col_unique].dropna().astype(str).unique()
        _dbg(f"Fetching UniProt data for {len(ids)} proteins...", cfg, log)

        records = [
            fetch_uniprot(pid, session=session, cache_root=cache_root, timeout_s=cfg.uniprot_timeout_s, cfg=cfg, log=log)
            for pid in ids
        ]
        uinfo_df = pd.DataFrame(records)

        # align join key name
        if "UniProt_ID" in uinfo_df.columns and uniprot_col_unique != "UniProt_ID":
            uinfo_df.rename(columns={"UniProt_ID": uniprot_col_unique}, inplace=True)

        # Convert list-like columns to Excel-friendly strings for merge/export
        list_cols = [c for c in uinfo_df.columns if c.startswith("UniProt_") and uinfo_df[c].apply(lambda x: isinstance(x, list)).any()]
        for c in list_cols:
            uinfo_df[c] = uinfo_df[c].apply(lambda x: cfg.uniprot_list_joiner.join(x) if isinstance(x, list) else (x if pd.notna(x) else ""))

        # Ensure expected numeric columns exist
        expected_cols = [
            uniprot_col_unique,
            "AA_len",
            "Cys_count",
            "Disulfide_Bond_Count",
            "Disulfide_Cys_Count",
            "Disulfide_Pairs_String",
            "Cys_fraction",
        ]
        for col in expected_cols:
            if col not in uinfo_df.columns:
                uinfo_df[col] = np.nan

        # Merge EVERYTHING we fetched (including rich string columns) into df_unique
        df_unique = df_unique.merge(uinfo_df, on=uniprot_col_unique, how="left")
        _dbg("✅ UniProt enrichment merged successfully.", cfg, log)

        # Write run-level artifacts (use original list form if available)
        if cfg.uniprot_write_run_artifact and uinfo_df is not None:
            # For artifacts, prefer list columns if we can reconstruct them from cached summaries
            # BUT: we already stringified list cols for Excel; that's still fine for stats aggregation.
            _write_uniprot_run_artifacts(out_dir, uinfo_df.copy(), log)

    else:
        _dbg("No UniProt column found, skipping enrichment.", cfg, log)

    # ---------- Filter hits (adaptive) ----------
    def filter_hits(df_in: pd.DataFrame, mode: str) -> pd.DataFrame:
        mu = f"mu_site_{mode}"
        sd = f"sd_site_{mode}"
        n  = f"n_DE_{mode}"
        if mu not in df_in.columns:
            return df_in.iloc[0:0].copy()

        n_avail = len(get_mode_columns(df_in, mode, "Log2FC", cfg.de_windows))
        n_req = required_windows_present(n_avail)
        _dbg(f"[filter_hits] {mode}: windows_available={n_avail} -> windows_required={n_req}", cfg, log)

        return df_in[
            (df_in[mu].abs() >= cfg.mu_abs_min)
            & (df_in[sd] <= cfg.sd_max)
            & (df_in[n] >= n_req)
        ].copy()

    hits_MS2_full = filter_hits(df_unique, "MS2")
    hits_MS3_full = filter_hits(df_unique, "MS3")

    if not hits_MS2_full.empty and "S_site_MS2" in hits_MS2_full.columns:
        hits_MS2_full = hits_MS2_full.sort_values("S_site_MS2", ascending=False)
    if not hits_MS3_full.empty and "S_site_MS3" in hits_MS3_full.columns:
        hits_MS3_full = hits_MS3_full.sort_values("S_site_MS3", ascending=False)

    _dbg(f"Hits_MS2: {len(hits_MS2_full)}, Hits_MS3: {len(hits_MS3_full)}", cfg, log)

    # Consistent hits: appear in both MS2 + MS3 hit sets
    idx_consistent = set(hits_MS2_full.index) & set(hits_MS3_full.index)
    hits_consistent_full = df_unique.loc[list(idx_consistent)].copy()
    _dbg(f"Consistent hits across MS2 and MS3: {len(hits_consistent_full)}", cfg, log)

    # ---------- Export ----------
    rename_map = {
        "mu_site_MS2": "Mean_LFC_MS2",
        "sd_site_MS2": "StdDev_MS2",
        "n_DE_MS2": "Num_Windows_MS2",
        "sign_flip_MS2": "Sign_Flip_MS2",
        "S_site_MS2": "S_Score_MS2",
        "mu_site_MS3": "Mean_LFC_MS3",
        "sd_site_MS3": "StdDev_MS3",
        "n_DE_MS3": "Num_Windows_MS3",
        "sign_flip_MS3": "Sign_Flip_MS3",
        "S_site_MS3": "S_Score_MS3",
    }

    ms2_metrics = [c for c in ["mu_site_MS2", "sd_site_MS2", "n_DE_MS2", "sign_flip_MS2", "S_site_MS2"] if c in df_unique.columns]
    ms3_metrics = [c for c in ["mu_site_MS3", "sd_site_MS3", "n_DE_MS3", "sign_flip_MS3", "S_site_MS3"] if c in df_unique.columns]

    # sort per-window columns
    lfc_cols_existing_sorted: List[str] = []
    padj_cols_existing_sorted: List[str] = []
    for mode in ["MS3", "MS2"]:
        for de in cfg.de_windows:
            c1 = f"{mode}_{de}_Log2FC"
            c2 = f"{mode}_{de}_p.adj"
            if c1 in df_unique.columns:
                lfc_cols_existing_sorted.append(c1)
            if c2 in df_unique.columns:
                padj_cols_existing_sorted.append(c2)

    # include UniProt columns (old + new rich columns)
    uniprot_cols_existing = [c for c in df_unique.columns if c.lower().startswith("uniprot_") or any(
        k in c.lower() for k in ["uniprot", "aa_", "cys", "disulfide", "pairs", "fraction"]
    )]
    extra_meta_cols = [c for c in extra_meta_cols_existing if c in df_unique.columns]

    cols_common = [*group_cols]
    if uniprot_col_unique and uniprot_col_unique not in cols_common:
        cols_common.append(uniprot_col_unique)

    cols_hits_MS2 = [*cols_common, *lfc_cols_existing_sorted, *padj_cols_existing_sorted, *extra_meta_cols, *ms2_metrics, *uniprot_cols_existing]
    cols_hits_MS3 = [*cols_common, *lfc_cols_existing_sorted, *padj_cols_existing_sorted, *extra_meta_cols, *ms3_metrics, *uniprot_cols_existing]
    cols_hits_consistent = [*cols_common, *lfc_cols_existing_sorted, *padj_cols_existing_sorted, *extra_meta_cols, *ms2_metrics, *ms3_metrics, *uniprot_cols_existing]

    cols_hits_MS2 = [c for c in cols_hits_MS2 if c in hits_MS2_full.columns]
    cols_hits_MS3 = [c for c in cols_hits_MS3 if c in hits_MS3_full.columns]
    cols_hits_consistent = [c for c in cols_hits_consistent if c in hits_consistent_full.columns]

    hits_MS2_export = hits_MS2_full[cols_hits_MS2].rename(columns=rename_map)
    hits_MS3_export = hits_MS3_full[cols_hits_MS3].rename(columns=rename_map)
    hits_consistent_export = hits_consistent_full[cols_hits_consistent].rename(columns=rename_map)

    unique_cols_keep = [*cols_common, *lfc_cols_existing_sorted, *padj_cols_existing_sorted, *extra_meta_cols, *ms2_metrics, *ms3_metrics, *uniprot_cols_existing]
    unique_cols_keep = [c for c in unique_cols_keep if c in df_unique.columns]
    df_unique_export = df_unique[unique_cols_keep].rename(columns=rename_map)

    out_path = out_dir / output_filename
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Input_With_Metrics", index=False)
        df_unique_export.to_excel(w, sheet_name="Unique_Sites", index=False)
        hits_MS2_export.to_excel(w, sheet_name="Hits_MS2", index=False)
        hits_MS3_export.to_excel(w, sheet_name="Hits_MS3", index=False)
        hits_consistent_export.to_excel(w, sheet_name="Hits_Consistent_MS2_MS3", index=False)

        workbook = w.book
        for sheet_name, df_sheet in {
            "Input_With_Metrics": df,
            "Unique_Sites": df_unique_export,
            "Hits_MS2": hits_MS2_export,
            "Hits_MS3": hits_MS3_export,
            "Hits_Consistent_MS2_MS3": hits_consistent_export,
        }.items():
            ws = workbook[sheet_name]
            autosize_columns(ws, df_sheet)

    _dbg(f"Results written to {out_path}", cfg, log)

    return StereoOutputs(
        output_xlsx=out_path,
        n_input_rows=int(len(df)),
        n_unique_rows=int(len(df_unique_export)),
        n_hits_ms2=int(len(hits_MS2_export)),
        n_hits_ms3=int(len(hits_MS3_export)),
        n_hits_consistent=int(len(hits_consistent_export)),
    )


# ---------------- Optional CLI wrapper ----------------

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    inp = StereoInputs(
        input_xlsx=script_dir / "merged_log2fc.xlsx",
        input_sheet=0,
    )
    out = run(inp, out_dir=script_dir, log=print)
    print(f"\n✅ Done: {out.output_xlsx} | Unique={out.n_unique_rows} | Hits MS2={out.n_hits_ms2} | Hits MS3={out.n_hits_ms3}")


if __name__ == "__main__":
    main()
