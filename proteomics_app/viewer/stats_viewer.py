from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Any, Dict, List, Tuple

import io
import json
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt 

try:
    from matplotlib_venn import venn2  # pip install matplotlib-venn
    HAS_VENN = True
except Exception:
    HAS_VENN = False


@st.cache_data(show_spinner=False)
def _read_sheet(xlsx_path: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")


def _get_site_set(df: pd.DataFrame) -> Set[str]:
    if df is None or df.empty or "Site" not in df.columns:
        return set()
    return set(df["Site"].dropna().astype(str))


def _site_to_call(unique: pd.DataFrame, *, call_col: str) -> dict[str, str]:
    if unique is None or unique.empty:
        return {}
    if "Site" not in unique.columns or call_col not in unique.columns:
        return {}

    m = unique[["Site", call_col]].dropna(subset=["Site"]).copy()
    m["Site"] = m["Site"].astype(str)

    def _norm_call(x: Any) -> str:
        s = ("" if x is None else str(x)).strip()
        if not s:
            return ""
        s_low = s.lower()
        if s_low.startswith("single"):
            return "Single"
        if s_low.startswith("partial"):
            return "Partial"
        if s_low.startswith("complete"):
            return "Complete"
        return s.title()

    m[call_col] = m[call_col].apply(_norm_call)
    return dict(zip(m["Site"], m[call_col]))


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _download_fig(fig, *, filename: str, label: str = "Download PNG") -> None:
    """
    IMPORTANT: must be called BEFORE st.pyplot(..., clear_figure=True) and before closing the fig.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.download_button(label=label, data=buf.getvalue(), file_name=filename, mime="image/png")


def _venn_ms2_ms3(ms2: set[str], ms3: set[str], title: str, *, download_name: Optional[str] = None):
    if not HAS_VENN:
        st.info("Install `matplotlib-venn` to enable Venn diagrams.")
        st.write({"MS2 hits": len(ms2), "MS3 hits": len(ms3), "Shared": len(ms2 & ms3)})
        return

    fig, ax = plt.subplots()
    v = venn2([ms2, ms3], set_labels=("MS2 hits", "MS3 hits"), ax=ax)
    ax.set_title(title)

    if getattr(v, "subset_labels", None):
        for t in v.subset_labels or []:
            if t:
                t.set_fontsize(14)
                t.set_fontweight("bold")

    if getattr(v, "set_labels", None):
        for t in v.set_labels or []:
            if t:
                t.set_fontsize(12)

    if download_name:
        _download_fig(fig, filename=download_name)

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# -------------------------
# UniProt annotation artifact integration
# -------------------------

def _autodetect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in norm:
            return norm[cand.lower()]
    return None


@st.cache_data(show_spinner=False)
def _load_uniprot_annotations_jsonl(jsonl_path: str) -> pd.DataFrame:
    p = Path(jsonl_path)
    if not p.exists():
        return pd.DataFrame()

    rows: List[dict] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    except Exception:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def _find_uniprot_id_col(df: pd.DataFrame) -> Optional[str]:
    return _autodetect_col(df, [
        "Uniprot ID",
        "UniProt ID",
        "UniProt_ID",
        "uniprot id",
        "accession",
        "Uniprot",
        "UniProt",
    ])


def _merge_uniprot_annotations(unique: pd.DataFrame, *, xlsx_path: Path) -> Tuple[pd.DataFrame, Optional[Path]]:
    """
    Merges UniProt annotations onto Unique_Sites.

    Priority:
      1) If Unique_Sites already contains UniProt_* columns, do nothing.
      2) Otherwise, load <dir>/uniprot_annotations.jsonl and merge by UniProt ID.
    """
    if unique is None or unique.empty:
        return unique, None

    # Already has UniProt rich columns?
    has_rich = any(str(c).startswith("UniProt_") for c in unique.columns)
    if has_rich:
        return unique, None

    uid_col = _find_uniprot_id_col(unique)
    if not uid_col:
        return unique, None

    cand = xlsx_path.parent / "uniprot_annotations.jsonl"
    if not cand.exists():
        return unique, None

    ann = _load_uniprot_annotations_jsonl(str(cand))
    if ann.empty:
        return unique, cand

    ann_uid = _find_uniprot_id_col(ann)
    if not ann_uid:
        # common in artifact: UniProt_ID
        if "UniProt_ID" in ann.columns:
            ann_uid = "UniProt_ID"
        else:
            return unique, cand

    # Normalize join keys as strings
    u = unique.copy()
    a = ann.copy()
    u[uid_col] = u[uid_col].astype(str)
    a[ann_uid] = a[ann_uid].astype(str)

    # Align column name for merge
    if ann_uid != uid_col:
        a = a.rename(columns={ann_uid: uid_col})

    merged = u.merge(a, on=uid_col, how="left")
    return merged, cand


def _split_terms_cell(x: Any) -> List[str]:
    """
    Robust splitter for terms stored either as:
      - list
      - "a; b; c"
      - "a | b | c"
      - "a, b, c"
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    if not s:
        return []
    # prefer '; ' separator used by stereoselectivity.py
    for sep in ["; ", ";", " | ", "|", ", "]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep)]
            return [p for p in parts if p]
    return [s]


# -------------------------
# Gene/protein trend helpers
# -------------------------

def _coarse_function_class(text: str) -> str:
    """
    Coarse classification based on combined UniProt text blob:
    protein name + keywords + GO + function/pathway/family lines.
    """
    t = ("" if text is None else str(text)).lower()

    rules = [
        ("Ribosome / translation", ["ribosomal", "translation", "trna", "elongation factor", "initiation factor"]),
        ("Protein folding / chaperone", ["chaperone", "heat shock", "hsp", "foldase", "protein folding"]),
        ("Proteasome / ubiquitin", ["ubiquitin", "proteasome", "e3", "deubiquitin", "ubiquitin ligase"]),
        ("Cytoskeleton / trafficking", ["actin", "tubulin", "myosin", "dynein", "kinesin", "coatomer", "clathrin"]),
        ("Redox / oxidative stress", ["thioredoxin", "peroxiredoxin", "glutathione", "oxidoreductase", "redox"]),
        ("DNA repair / replication", ["dna repair", "helicase", "polymerase", "ligase", "replication", "chromatin"]),
        ("RNA binding / splicing", ["rna-binding", "splice", "splicing", "snrnp", "ribonucl", "cstf", "hnrnp"]),
        ("Mitochondrial", ["mitochond", "tca", "oxidative phosphorylation", "electron transport", "respiratory chain"]),
        ("Metabolism / enzyme", ["dehydrogenase", "kinase", "phosphatase", "synthetase", "transferase", "lyase", "isomerase", "hydrolase", "enzyme", "ec:"]),
        ("Membrane / transport", ["transporter", "channel", "solute carrier", "slc", "atpase", "pump", "membrane"]),
        ("Signaling / regulation", ["g protein", "kinase", "phosphatase", "adapter", "scaffold", "signaling", "receptor"]),
    ]

    for cls, keys in rules:
        if any(k in t for k in keys):
            return cls
    return "Other / Unknown"


def _pick_annotation_text(unique: pd.DataFrame) -> pd.Series:
    """
    Choose the best available annotation text column(s) to classify proteins.
    Prefer UniProt_* fields if present.
    """
    # Prefer rich UniProt fields (new)
    preferred = [
        "UniProt_Protein_Name",
        "UniProt_Keywords",
        "UniProt_GO_BP",
        "UniProt_GO_MF",
        "UniProt_GO_CC",
        "UniProt_Subcellular_Location",
        "UniProt_EC",
        "UniProt_Pathway",
        "UniProt_Family",
    ]
    # Fallbacks (older)
    fallbacks = [
        "Protein name",
        "Protein Name",
        "Recommended protein name",
        "Entry Name",
        "Function",
        "UniProt_Function",
        "Gene Name",
        "Uniprot ID",
        "UniProt ID",
        "Disulfide_Pairs_String",
    ]

    cols = [c for c in preferred if c in unique.columns]
    if not cols:
        cols = [c for c in fallbacks if c in unique.columns]

    if not cols:
        return pd.Series([""] * len(unique), index=unique.index)

    txt = unique[cols].astype(str).fillna("").agg(" | ".join, axis=1)
    return txt


def _top_terms(
    df: pd.DataFrame,
    col: str,
    mask: pd.Series,
    *,
    top_n: int = 25,
) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=["term", "count"])

    sub = df.loc[mask].copy()
    terms: List[str] = []
    for v in sub[col].tolist():
        terms.extend(_split_terms_cell(v))

    if not terms:
        return pd.DataFrame(columns=["term", "count"])

    vc = pd.Series(terms).value_counts()
    out = vc.head(int(top_n)).reset_index()
    out.columns = ["term", "count"]
    return out


def _term_enrichment_table(
    df: pd.DataFrame,
    col: str,
    mask_hit: pd.Series,
    mask_bg: pd.Series,
    *,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Simple, reviewer-friendly enrichment:
      frac_hit / frac_bg
    plus raw counts.
    """
    if col not in df.columns:
        return pd.DataFrame()

    # collect counts
    def _collect(mask: pd.Series) -> pd.Series:
        terms: List[str] = []
        for v in df.loc[mask, col].tolist():
            terms.extend(_split_terms_cell(v))
        if not terms:
            return pd.Series(dtype=int)
        return pd.Series(terms).value_counts()

    vc_hit = _collect(mask_hit)
    vc_bg = _collect(mask_bg)

    if vc_hit.empty and vc_bg.empty:
        return pd.DataFrame()

    # candidates: top by hit count primarily
    candidates = vc_hit.head(int(top_n)).index.tolist()
    if len(candidates) < top_n:
        # fill from bg if needed
        extra = [t for t in vc_bg.head(int(top_n)).index.tolist() if t not in candidates]
        candidates.extend(extra[: max(0, int(top_n) - len(candidates))])

    hit_total = float(vc_hit.sum() or 1.0)
    bg_total = float(vc_bg.sum() or 1.0)

    rows = []
    for term in candidates:
        n_hit = int(vc_hit.get(term, 0))
        n_bg = int(vc_bg.get(term, 0))
        frac_hit = n_hit / hit_total
        frac_bg = n_bg / bg_total
        enr = (frac_hit / frac_bg) if frac_bg > 0 else np.inf
        rows.append({
            "term": term,
            "hit_count": n_hit,
            "bg_count": n_bg,
            "hit_frac": frac_hit,
            "bg_frac": frac_bg,
            "enrichment_ratio": enr,
        })

    out = pd.DataFrame(rows).sort_values(["enrichment_ratio", "hit_count"], ascending=[False, False])
    return out


def render_stats(
    *,
    stereo_xlsx: Optional[Path] = None,
    split_xlsx: Optional[Path] = None,
    embedded: bool = True,
) -> None:
    """
    Renders statistics + plots from pipeline outputs.
    - stereo_xlsx: merged_log2fc_results.xlsx (Step 2)
    - split_xlsx: merged_log2fc_results_sites_with_skyline_splitting.xlsx (Step 4)
    """
    if embedded:
        st.header("Stats")
    else:
        st.title("Stats")

    # Prefer split_xlsx if available (contains Unique_Sites with split calls)
    xlsx = split_xlsx if (split_xlsx and Path(split_xlsx).exists()) else stereo_xlsx
    if not xlsx or not Path(xlsx).exists():
        st.info("No workbook available.")
        return

    xlsx = Path(xlsx)
    run_tag = f"{xlsx.stem}_{_stamp()}"

    # -------------------------
    # Load Unique_Sites (and hits)
    # -------------------------
    try:
        unique = _read_sheet(str(xlsx), "Unique_Sites")
    except Exception:
        unique = pd.DataFrame()

    # Merge UniProt annotation artifact if present
    unique, ann_path = _merge_uniprot_annotations(unique, xlsx_path=xlsx)
    if ann_path is not None and ann_path.exists():
        st.caption(f"UniProt annotations loaded from: `{ann_path.name}`")

    call_col = "sky_split_call_combined_worst"
    site_call = _site_to_call(unique, call_col=call_col) if not unique.empty else {}

    ms2, ms3 = set(), set()
    for sheet, tgt in [("Hits_MS2", "ms2"), ("Hits_MS3", "ms3")]:
        try:
            df = _read_sheet(str(xlsx), sheet)
        except Exception:
            df = pd.DataFrame()
        if tgt == "ms2":
            ms2 = _get_site_set(df)
        else:
            ms3 = _get_site_set(df)

    # -------------------------
    # Hit overlap plots
    # -------------------------
    with st.expander("MS2 vs MS3 hit overlap", expanded=True):
        if ms2 or ms3:
            _venn_ms2_ms3(ms2, ms3, "Hit overlap (all hits, by Site)", download_name=f"venn_all_{run_tag}.png")
            st.write({"MS2 hits": len(ms2), "MS3 hits": len(ms3), "Shared": len(ms2 & ms3)})
        else:
            st.info("Hit sheets not found (expected Hits_MS2 / Hits_MS3).")

        if site_call and (ms2 or ms3):
            ms2_single = {s for s in ms2 if site_call.get(s) == "Single"}
            ms3_single = {s for s in ms3 if site_call.get(s) == "Single"}
            st.markdown("### Single-only overlap")
            _venn_ms2_ms3(
                ms2_single,
                ms3_single,
                "Hit overlap (Single sites only, by Site)",
                download_name=f"venn_single_{run_tag}.png",
            )
            st.write(
                {
                    "MS2 Single hits": len(ms2_single),
                    "MS3 Single hits": len(ms3_single),
                    "Shared Single": len(ms2_single & ms3_single),
                }
            )
        elif not site_call:
            st.info("Single-only overlap unavailable (Unique_Sites or split-call column missing).")

    # If no unique sheet, stop here
    if unique.empty:
        st.divider()
        st.caption(f"Workbook used: `{xlsx.name}`")
        return

    # -------------------------
    # Normalize split type and basic masks
    # -------------------------
    def _norm_call(x: Any) -> str:
        s = ("" if x is None else str(x)).strip()
        if not s:
            return "Missing"
        s_low = s.lower()
        if s_low.startswith("single"):
            return "Single"
        if s_low.startswith("partial"):
            return "Partial"
        if s_low.startswith("complete"):
            return "Complete"
        return s.title()

    unique["_split_type"] = unique[call_col].apply(_norm_call) if call_col in unique.columns else "Missing"

    split_levels = ["Single", "Partial", "Complete", "Missing"]
    colors = {
        "Single": "#2ca02c",
        "Partial": "#ff7f0e",
        "Complete": "#d62728",
        "Missing": "#7f7f7f",
    }

    # Membership masks (for group stats)
    if "Site" in unique.columns:
        u_site = unique["Site"].astype(str)
        mask_ms2 = u_site.isin(ms2) if ms2 else pd.Series([False] * len(unique), index=unique.index)
        mask_ms3 = u_site.isin(ms3) if ms3 else pd.Series([False] * len(unique), index=unique.index)
        mask_shared = mask_ms2 & mask_ms3
    else:
        mask_ms2 = mask_ms3 = mask_shared = pd.Series([False] * len(unique), index=unique.index)

    st.subheader("Unique sites summary")

    # -------------------------
    # NEW: UniProt class stats expander
    # -------------------------
    with st.expander("UniProt class stats (keywords / GO / subcellular / EC)", expanded=False):
        st.caption("Uses UniProt_* columns from Unique_Sites or merges `uniprot_annotations.jsonl` if present.")

        cols_interest = [
            "UniProt_Keywords",
            "UniProt_GO_BP",
            "UniProt_GO_MF",
            "UniProt_GO_CC",
            "UniProt_Subcellular_Location",
            "UniProt_EC",
            "UniProt_Pathway",
        ]
        available = [c for c in cols_interest if c in unique.columns]
        if not available:
            st.info("No UniProt_* annotation columns available. (Run stereoselectivity with rich UniProt export.)")
        else:
            group_opts = {
                "All Unique sites": pd.Series([True] * len(unique), index=unique.index),
                "MS2 hits": mask_ms2,
                "MS3 hits": mask_ms3,
                "Shared hits": mask_shared,
                "Shared hits (Single only)": (mask_shared & (unique["_split_type"] == "Single")),
            }

            col_pick = st.selectbox("Annotation field", available, index=0, key="uniprot_class_field")
            bg_name = st.selectbox("Background group", list(group_opts.keys()), index=0, key="uniprot_bg_group")
            hit_name = st.selectbox("Hit group", list(group_opts.keys()), index=3 if "Shared hits" in group_opts else 0, key="uniprot_hit_group")
            top_n = st.slider("Top N terms", 10, 60, 25, key="uniprot_topn")

            mask_bg = group_opts[bg_name]
            mask_hit = group_opts[hit_name]

            tab1, tab2 = st.tabs(["Top terms", "Enrichment (hit vs background)"])

            with tab1:
                cols2 = st.columns(2)
                with cols2[0]:
                    st.write(f"**{bg_name}**")
                    t_bg = _top_terms(unique, col_pick, mask_bg, top_n=int(top_n))
                    st.dataframe(t_bg, width="stretch", hide_index=True)
                with cols2[1]:
                    st.write(f"**{hit_name}**")
                    t_hit = _top_terms(unique, col_pick, mask_hit, top_n=int(top_n))
                    st.dataframe(t_hit, width="stretch", hide_index=True)

            with tab2:
                enr = _term_enrichment_table(unique, col_pick, mask_hit, mask_bg, top_n=int(top_n))
                if enr.empty:
                    st.info("Not enough terms to compute enrichment.")
                else:
                    show = enr.copy()
                    # pretty formatting
                    show["hit_frac"] = (100.0 * show["hit_frac"]).round(2)
                    show["bg_frac"] = (100.0 * show["bg_frac"]).round(2)
                    show["enrichment_ratio"] = show["enrichment_ratio"].replace([np.inf], np.nan)
                    st.dataframe(
                        show[["term", "hit_count", "bg_count", "hit_frac", "bg_frac", "enrichment_ratio"]],
                        width="stretch",
                        hide_index=True,
                    )

    # -------------------------
    # Plot: Split call distribution (simple) + download
    # -------------------------
    with st.expander("Split call distribution", expanded=True):
        counts = unique["_split_type"].value_counts().reindex(split_levels, fill_value=0)

        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values, color=[colors[k] for k in counts.index])
        ax.set_xlabel("Split call")
        ax.set_ylabel("Count")
        ax.set_title("Split call distribution (Unique_Sites)")
        plt.xticks(rotation=20, ha="right")

        _download_fig(fig, filename=f"split_calls_{run_tag}.png")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    # -------------------------
    # Extra stat A: Split-type enrichment table across groups
    # -------------------------
    with st.expander("Split-type enrichment across groups (counts + %)", expanded=False):
        def _split_counts(df_in: pd.DataFrame) -> pd.Series:
            s = df_in["_split_type"].value_counts()
            return s.reindex(split_levels, fill_value=0)

        groups = {
            "All Unique": unique,
            "MS2 hits": unique[mask_ms2].copy(),
            "MS3 hits": unique[mask_ms3].copy(),
            "Shared": unique[mask_shared].copy(),
            "Shared (Single only)": unique[mask_shared & (unique["_split_type"] == "Single")].copy(),
        }

        rows = []
        for name, df_g in groups.items():
            cts = _split_counts(df_g)
            total = int(cts.sum()) if int(cts.sum()) > 0 else 1
            row = {"Group": name, "Total": int(cts.sum())}
            for lvl in split_levels:
                row[f"{lvl}_n"] = int(cts[lvl])
                row[f"{lvl}_pct"] = round(100.0 * float(cts[lvl]) / float(total), 1)
            rows.append(row)

        out = pd.DataFrame(rows)
        show_cols = ["Group", "Total"] + [f"{lvl}_n" for lvl in split_levels] + [f"{lvl}_pct" for lvl in split_levels]
        st.dataframe(out[show_cols], width="stretch", hide_index=True)

    # -------------------------
    # Extra stat B: Top genes by hit burden
    # -------------------------
    with st.expander("Top genes by hit burden", expanded=False):
        if "Gene Name" not in unique.columns or "Site" not in unique.columns:
            st.info("Missing 'Gene Name' and/or 'Site' in Unique_Sites.")
        else:
            df_g = unique.copy()
            df_g["Gene Name"] = df_g["Gene Name"].fillna("").astype(str)
            df_g["Site"] = df_g["Site"].fillna("").astype(str)

            df_g["is_ms2_hit"] = mask_ms2.values
            df_g["is_ms3_hit"] = mask_ms3.values
            df_g["is_shared"] = mask_shared.values
            df_g["is_single"] = (df_g["_split_type"] == "Single")

            def _shared_single_sum(idx) -> int:
                sub = df_g.loc[idx, :]
                return int(((sub["is_shared"]) & (sub["is_single"])).sum())

            g = (
                df_g.groupby("Gene Name", dropna=False)
                .agg(
                    n_unique=("Site", "count"),
                    n_ms2=("is_ms2_hit", "sum"),
                    n_ms3=("is_ms3_hit", "sum"),
                    n_shared=("is_shared", "sum"),
                    n_shared_single=("is_shared", lambda s: _shared_single_sum(s.index)),
                )
                .reset_index()
            )
            g = g[g["Gene Name"].str.strip() != ""].copy()

            metric = st.selectbox(
                "Rank genes by",
                ["n_shared", "n_shared_single", "n_ms3", "n_ms2", "n_unique"],
                index=0,
                key="stats_gene_rank_metric",
            )
            top_n = st.slider("Top N genes", 5, 50, 20, key="stats_gene_topn")
            g2 = g.sort_values(metric, ascending=False).head(int(top_n)).copy()

            st.dataframe(g2, width="stretch", hide_index=True)

            if not g2.empty:
                fig, ax = plt.subplots(figsize=(7, max(3, 0.25 * len(g2))))
                ax.barh(g2["Gene Name"][::-1], g2[metric][::-1])
                ax.set_xlabel(metric)
                ax.set_ylabel("Gene Name")
                ax.set_title(f"Top genes by {metric}")
                _download_fig(fig, filename=f"top_genes_{metric}_{run_tag}.png")
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

    # -------------------------
    # Gene-level functional trends
    # -------------------------
    with st.expander("Gene-level functional trends (broad protein classes + selectivity)", expanded=False):
        if "Gene Name" not in unique.columns or "Site" not in unique.columns:
            st.info("Missing 'Gene Name' and/or 'Site' in Unique_Sites.")
        else:
            df = unique.copy()
            df["Gene Name"] = df["Gene Name"].fillna("").astype(str)
            df["Site"] = df["Site"].fillna("").astype(str)

            annot_text = _pick_annotation_text(df)
            df["_annot_text"] = annot_text
            df["_func_class"] = df["_annot_text"].apply(_coarse_function_class)

            df["_is_ms2_hit"] = mask_ms2.values
            df["_is_ms3_hit"] = mask_ms3.values
            df["_is_shared"] = mask_shared.values
            df["_is_single"] = (df["_split_type"] == "Single")

            agg = (
                df.groupby(["Gene Name", "_func_class"], dropna=False)
                .agg(
                    n_sites=("Site", "count"),
                    frac_single=("_is_single", "mean"),
                    n_shared=("_is_shared", "sum"),
                    n_ms3=("_is_ms3_hit", "sum"),
                    n_ms2=("_is_ms2_hit", "sum"),
                    max_S_MS3=("S_Score_MS3", "max") if "S_Score_MS3" in df.columns else ("Site", "count"),
                    max_S_MS2=("S_Score_MS2", "max") if "S_Score_MS2" in df.columns else ("Site", "count"),
                )
                .reset_index()
            )
            agg = agg[agg["Gene Name"].str.strip() != ""].copy()
            agg["frac_single"] = pd.to_numeric(agg["frac_single"], errors="coerce")

            st.write("**Gene summary (selectivity + burden)**")
            st.dataframe(
                agg.sort_values(["n_shared", "n_sites"], ascending=False).head(50),
                width="stretch",
                hide_index=True,
            )

            def _class_fraction(df_in: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
                sub = df_in.loc[mask].copy() if mask is not None else df_in.copy()
                vc = sub["_func_class"].value_counts(dropna=False)
                out = vc.reset_index()
                out.columns = ["func_class", "count"]
                out["group"] = label
                out["fraction"] = out["count"] / float(out["count"].sum() or 1.0)
                return out

            class_all = _class_fraction(df, pd.Series([True] * len(df), index=df.index), "All Unique sites")
            class_shared = _class_fraction(df, df["_is_shared"], "Shared hits")
            class_single_shared = _class_fraction(df, df["_is_shared"] & df["_is_single"], "Shared hits (Single only)")
            cls_df = pd.concat([class_all, class_shared, class_single_shared], ignore_index=True)

            fig, ax = plt.subplots(figsize=(9, 4))
            top_classes = (
                class_shared.sort_values("fraction", ascending=False)["func_class"].head(10).tolist()
                if not class_shared.empty
                else class_all.sort_values("fraction", ascending=False)["func_class"].head(10).tolist()
            )
            cls_df2 = cls_df.copy()
            cls_df2["func_class2"] = cls_df2["func_class"].where(cls_df2["func_class"].isin(top_classes), other="Other (lumped)")

            cls_df2 = (
                cls_df2.groupby(["group", "func_class2"], as_index=False)
                .agg(count=("count", "sum"))
            )
            cls_df2["fraction"] = cls_df2.groupby("group")["count"].transform(lambda s: s / float(s.sum() or 1.0))

            pivot = cls_df2.pivot(index="func_class2", columns="group", values="fraction").fillna(0.0)
            pivot = pivot.reindex(sorted(pivot.index.tolist()), axis=0)

            x = np.arange(len(pivot.index))
            width = 0.28
            cols = pivot.columns.tolist()

            for i, colname in enumerate(cols):
                ax.bar(x + (i - (len(cols) - 1) / 2.0) * width, pivot[colname].values, width=width, label=colname)

            ax.set_xticks(x)
            ax.set_xticklabels(pivot.index.tolist(), rotation=25, ha="right")
            ax.set_ylabel("Fraction of sites")
            ax.set_title("Broad functional class composition (site-level)")
            ax.legend(fontsize=9, frameon=False, ncol=1)

            _download_fig(fig, filename=f"functional_class_composition_{run_tag}.png")
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

            score_col = "S_Score_MS3" if "S_Score_MS3" in df.columns else ("S_Score_MS2" if "S_Score_MS2" in df.columns else None)
            if score_col:
                df_sc = df.copy()
                df_sc[score_col] = pd.to_numeric(df_sc[score_col], errors="coerce")
                gsc = (
                    df_sc.groupby(["Gene Name", "_func_class"], dropna=False)
                    .agg(n_sites=("Site", "count"), max_score=(score_col, "max"), frac_single=("_is_single", "mean"))
                    .reset_index()
                )
                gsc = gsc.dropna(subset=["max_score"])
                gsc = gsc[gsc["Gene Name"].str.strip() != ""].copy()

                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                top_fc = gsc["_func_class"].value_counts().head(8).index.tolist()
                for fc in top_fc:
                    d = gsc[gsc["_func_class"] == fc]
                    ax.scatter(d["n_sites"], d["max_score"], s=18, alpha=0.75, label=fc)
                d_other = gsc[~gsc["_func_class"].isin(top_fc)]
                if not d_other.empty:
                    ax.scatter(d_other["n_sites"], d_other["max_score"], s=18, alpha=0.4, label="Other / Unknown")

                ax.set_xscale("log")
                ax.set_xlabel("Sites per gene (log scale)")
                ax.set_ylabel(f"Max {score_col} per gene")
                ax.set_title("Gene burden vs strength (colored by broad class)")
                ax.legend(fontsize=8, frameon=False, ncol=2)

                _download_fig(fig, filename=f"gene_burden_vs_score_{run_tag}.png")
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

            ctab = pd.crosstab(df["_func_class"], df["_split_type"])
            for lvl in split_levels:
                if lvl not in ctab.columns:
                    ctab[lvl] = 0
            ctab = ctab[split_levels]
            frac = ctab.div(ctab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

            top_classes = ctab.sum(axis=1).sort_values(ascending=False).head(12).index.tolist()
            frac2 = frac.loc[top_classes].copy()

            fig, ax = plt.subplots(figsize=(9, max(3.5, 0.35 * len(frac2))))
            bottoms = np.zeros(len(frac2))
            ylabels = frac2.index.tolist()

            for lvl in split_levels:
                vals = frac2[lvl].values
                ax.barh(ylabels, vals, left=bottoms, color=colors[lvl], label=lvl)
                bottoms = bottoms + vals

            ax.set_xlim(0, 1)
            ax.set_xlabel("Fraction of sites")
            ax.set_title("Split-type composition by broad functional class (top classes)")
            ax.legend(fontsize=9, frameon=False, ncol=4)

            _download_fig(fig, filename=f"split_by_function_class_{run_tag}.png")
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

    # -------------------------
    # Extra stat E: Log2FC window presence + sign agreement (sanity)
    # -------------------------
    with st.expander("Window consistency sanity (presence + sign agreement)", expanded=False):
        if "Site" not in unique.columns:
            st.info("Missing 'Site' column in Unique_Sites.")
        else:
            lfc_cols = [c for c in unique.columns if str(c).endswith("_Log2FC")]
            if not lfc_cols:
                st.info("No *_Log2FC columns found in Unique_Sites.")
            else:
                df_w = unique.copy()
                for c in lfc_cols:
                    df_w[c] = pd.to_numeric(df_w[c], errors="coerce")

                df_w["n_windows_present"] = df_w[lfc_cols].notna().sum(axis=1)

                def _sign_agree(row) -> float:
                    vals = row.values
                    vals = vals[pd.notna(vals)]
                    if len(vals) == 0:
                        return float("nan")
                    signs = pd.Series(vals).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                    signs = signs[signs != 0]
                    if len(signs) == 0:
                        return float("nan")
                    pos = (signs > 0).sum()
                    neg = (signs < 0).sum()
                    maj = max(pos, neg)
                    return float(maj) / float(len(signs)) if len(signs) else float("nan")

                df_w["sign_agreement"] = df_w[lfc_cols].apply(_sign_agree, axis=1)

                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots()
                    ax.hist(
                        df_w["n_windows_present"].dropna().values,
                        bins=range(0, int(df_w["n_windows_present"].max() or 0) + 2),
                    )
                    ax.set_title("Number of Log2FC windows present")
                    ax.set_xlabel("n_windows_present")
                    ax.set_ylabel("Count")
                    _download_fig(fig, filename=f"windows_present_hist_{run_tag}.png")
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)

                with c2:
                    vals = pd.to_numeric(df_w["sign_agreement"], errors="coerce").dropna()
                    if vals.empty:
                        st.info("Not enough data to compute sign agreement.")
                    else:
                        fig, ax = plt.subplots()
                        ax.hist(vals.values, bins=20)
                        ax.set_title("Sign agreement across windows")
                        ax.set_xlabel("fraction matching majority sign")
                        ax.set_ylabel("Count")
                        _download_fig(fig, filename=f"sign_agreement_hist_{run_tag}.png")
                        st.pyplot(fig, clear_figure=True)
                        plt.close(fig)

                worst = df_w.sort_values("sign_agreement", ascending=True).head(25)
                cols_show = [c for c in ["Gene Name", "Site", "_split_type", "n_windows_present", "sign_agreement"] if c in worst.columns]
                st.write("Worst sign agreement sites (review candidates)")
                st.dataframe(worst[cols_show], width="stretch", hide_index=True)

    # -------------------------
    # Helper: stacked histogram (stacked by split type) with correct saving
    # -------------------------
    def _stacked_hist(df: pd.DataFrame, col: str, title: str, fname: str, bins: int = 50) -> None:
        if col not in df.columns:
            return
        v = pd.to_numeric(df[col], errors="coerce")
        df2 = df.assign(_v=v).dropna(subset=["_v"])
        if df2.empty:
            return

        data, label_list, color_list = [], [], []
        for lvl in split_levels:
            arr = df2.loc[df2["_split_type"] == lvl, "_v"].values
            if len(arr) == 0:
                continue
            data.append(arr)
            label_list.append(lvl)
            color_list.append(colors[lvl])

        fig, ax = plt.subplots()
        ax.hist(data, bins=bins, stacked=True, label=label_list, color=color_list)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend(ncol=4, fontsize=9, frameon=False)

        _download_fig(fig, filename=fname)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    # -------------------------
    # Plot: S-score distributions (stacked by split type) + download
    # -------------------------
    with st.expander("S-score distributions (stacked by split type)", expanded=True):
        if "S_Score_MS2" in unique.columns:
            _stacked_hist(unique, "S_Score_MS2", "S_Score_MS2 distribution by split type", f"S_Score_MS2_{run_tag}.png")
        if "S_Score_MS3" in unique.columns:
            _stacked_hist(unique, "S_Score_MS3", "S_Score_MS3 distribution by split type", f"S_Score_MS3_{run_tag}.png")

    # -------------------------
    # Plot + Extra stat D: S-score concordance + correlations + download
    # -------------------------
    with st.expander("S-score concordance (MS2 vs MS3) colored by split type", expanded=True):
        if {"S_Score_MS2", "S_Score_MS3"} <= set(unique.columns):
            df = unique.copy()
            df["S_Score_MS2"] = pd.to_numeric(df["S_Score_MS2"], errors="coerce")
            df["S_Score_MS3"] = pd.to_numeric(df["S_Score_MS3"], errors="coerce")
            df = df.dropna(subset=["S_Score_MS2", "S_Score_MS3"])

            if df.empty:
                st.info("No rows with both S-scores.")
            else:
                pearson = df[["S_Score_MS2", "S_Score_MS3"]].corr(method="pearson").iloc[0, 1]
                spearman = df[["S_Score_MS2", "S_Score_MS3"]].corr(method="spearman").iloc[0, 1]
                st.write({"N_sites": int(len(df)), "pearson_r": float(pearson), "spearman_rho": float(spearman)})

                fig, ax = plt.subplots()
                for lvl in split_levels:
                    d = df[df["_split_type"] == lvl]
                    if d.empty:
                        continue
                    ax.scatter(d["S_Score_MS2"], d["S_Score_MS3"], s=10, label=lvl, color=colors[lvl], alpha=0.8)
                ax.set_xlabel("S_Score_MS2")
                ax.set_ylabel("S_Score_MS3")
                ax.set_title("S-score concordance by split type")
                ax.legend(ncol=4, fontsize=9, frameon=False)

                _download_fig(fig, filename=f"Sscore_scatter_{run_tag}.png")
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
        else:
            st.info("Missing S_Score_MS2 and/or S_Score_MS3 columns in Unique_Sites.")

    # -------------------------
    # Extra stat C: Discordant sites list (actionable review)
    # -------------------------
    with st.expander("Discordant sites (high MS3/low MS2 and high MS2/low MS3)", expanded=False):
        if {"S_Score_MS2", "S_Score_MS3"} <= set(unique.columns):
            df_d = unique.copy()
            df_d["S_Score_MS2"] = pd.to_numeric(df_d["S_Score_MS2"], errors="coerce")
            df_d["S_Score_MS3"] = pd.to_numeric(df_d["S_Score_MS3"], errors="coerce")
            df_d = df_d.dropna(subset=["S_Score_MS2", "S_Score_MS3"])

            if df_d.empty:
                st.info("No rows with both S-scores.")
            else:
                q_hi = 0.90
                q_lo = 0.10
                ms2_hi = float(df_d["S_Score_MS2"].quantile(q_hi))
                ms2_lo = float(df_d["S_Score_MS2"].quantile(q_lo))
                ms3_hi = float(df_d["S_Score_MS3"].quantile(q_hi))
                ms3_lo = float(df_d["S_Score_MS3"].quantile(q_lo))

                st.caption(
                    f"Thresholds: MS2 high>Q{int(q_hi*100)}={ms2_hi:.3g}, low<Q{int(q_lo*100)}={ms2_lo:.3g}; "
                    f"MS3 high>Q{int(q_hi*100)}={ms3_hi:.3g}, low<Q{int(q_lo*100)}={ms3_lo:.3g}"
                )

                high_ms3_low_ms2 = df_d[(df_d["S_Score_MS3"] >= ms3_hi) & (df_d["S_Score_MS2"] <= ms2_lo)].copy()
                high_ms2_low_ms3 = df_d[(df_d["S_Score_MS2"] >= ms2_hi) & (df_d["S_Score_MS3"] <= ms3_lo)].copy()

                cols = [c for c in ["Gene Name", "Site", "_split_type", "S_Score_MS2", "S_Score_MS3", "Uniprot ID", "UniProt ID", "Best Peptide"] if c in df_d.columns]

                st.write("**High MS3 / Low MS2**")
                if high_ms3_low_ms2.empty:
                    st.info("None under current thresholds.")
                else:
                    st.dataframe(high_ms3_low_ms2[cols].sort_values("S_Score_MS3", ascending=False), width="stretch", hide_index=True)

                st.write("**High MS2 / Low MS3**")
                if high_ms2_low_ms3.empty:
                    st.info("None under current thresholds.")
                else:
                    st.dataframe(high_ms2_low_ms3[cols].sort_values("S_Score_MS2", ascending=False), width="stretch", hide_index=True)
        else:
            st.info("Missing S_Score_MS2 and/or S_Score_MS3 columns in Unique_Sites.")

    st.divider()
    st.caption(f"Workbook used: `{xlsx.name}`")
