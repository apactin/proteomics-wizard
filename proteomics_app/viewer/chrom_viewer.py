from __future__ import annotations

from pathlib import Path
import re
import math
from typing import Optional, Tuple, Any, Sequence

import pandas as pd
import streamlit as st

# -------------------------
# Defaults / constants
# -------------------------
DEFAULT_UNIQUE_SHEET = "Unique_Sites"

LFC_COLS = [
    "MS3_0s_Log2FC",
    "MS3_5s_Log2FC",
    "MS3_30s_Log2FC",
    "MS2_5s_Log2FC",
    "MS2_30s_Log2FC",
]

SCORE_COLS = ["S_Score_MS2", "S_Score_MS3"]
WINDOW_ORDER = ["MS3_0s", "MS3_5s", "MS3_30s", "MS2_5s", "MS2_30s"]

SITE_CALL_COL = "sky_split_call_combined_worst"


# -------------------------
# Helpers
# -------------------------
def safe_str_col(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str)


def to_numeric_col(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def sheet_sites(xlfile: pd.ExcelFile, sheet_name: str) -> set[str]:
    if sheet_name not in xlfile.sheet_names:
        return set()
    s = xlfile.parse(sheet_name)
    if "Site" in s.columns:
        return set(s["Site"].dropna().astype(str))
    return set()


def normalize_ms_level(x: str) -> str:
    s = (x or "").strip().lower()
    if s in ("ms2", "2", "ms2.0"):
        return "ms2"
    if s in ("ms3", "3", "ms3.0"):
        return "ms3"
    return s


def normalize_call(x: str) -> str:
    s = (x or "").strip()
    if not s:
        return ""
    s_low = s.lower()
    if s_low.startswith("single"):
        return "Single"
    if s_low.startswith("partial"):
        return "Partial"
    if s_low.startswith("complete"):
        return "Complete"
    return s_low.title()


def normalize_site(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().upper()
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("_", " ").replace("/", " ").replace("\\", " ")
    s = re.sub(r"\bCYS\b", "C", s)
    s = re.sub(r"[^A-Z0-9 \-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" - ", "-").replace("- ", "-").replace(" -", "-")
    return s


def infer_window_from_filename(file_name: str, ms_level: str) -> str:
    fn = (file_name or "").lower()
    ms = (ms_level or "").lower()

    if ("ms3" in fn) or (ms == "ms3"):
        if "0s" in fn:
            return "MS3_0s"
        if "5s" in fn:
            return "MS3_5s"
        if "30s" in fn:
            return "MS3_30s"

    if ("ms2" in fn) or (ms == "ms2"):
        if "5s" in fn:
            return "MS2_5s"
        if "30s" in fn:
            return "MS2_30s"

    m = re.search(r"(\b0s\b|\b5s\b|\b30s\b)", fn)
    if m:
        win = m.group(1)
        if ms == "ms3":
            return f"MS3_{win}"
        if ms == "ms2":
            return f"MS2_{win}"
    return ""


def _as_list(x: Any) -> list:
    """
    Normalize trace arrays coming from parquet into plain Python lists.

    Avoids patterns like `x or []` which crash when x is a numpy array:
    ValueError: truth value of an array is ambiguous.
    """
    if x is None:
        return []
    # numpy arrays / pandas arrays / Series commonly support .tolist()
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    if isinstance(x, (list, tuple)):
        return list(x)
    # scalar fallback
    return [x]


def _downsample_xy(x, y, max_points: int):
    """Uniformly downsample to <= max_points for faster plotting."""
    if max_points is None or max_points <= 0:
        return x, y

    # Ensure sliceability / consistent lengths
    x = _as_list(x)
    y = _as_list(y)

    n = min(len(x), len(y))
    if n <= max_points:
        return x[:n], y[:n]
    step = max(1, int(math.ceil(n / max_points)))
    return x[:n:step], y[:n:step]


# Try importing plotly once (fast path), otherwise matplotlib fallback
try:
    import plotly.graph_objects as go  # type: ignore

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


def plot_replicate_overlay(rep_df: pd.DataFrame, title: str, *, max_traces: int, max_points: int, chart_key: int):
    # Limit number of traces to avoid pathological cases
    if max_traces is not None and max_traces > 0 and len(rep_df) > max_traces:
        rep_df = rep_df.iloc[:max_traces].copy()

    if HAS_PLOTLY:
        fig = go.Figure()
        for _, r in rep_df.iterrows():
            # IMPORTANT: do not use `... or []` because numpy arrays can't be truth-tested
            rt = _as_list(r.get("rt"))
            y = _as_list(r.get("intensity"))
            rt2, y2 = _downsample_xy(rt, y, max_points=max_points)

            label = f"mz {r.get('ProductMz','')} z{r.get('PrecursorCharge','')}"
            fig.add_trace(go.Scatter(x=rt2, y=y2, mode="lines", name=label, opacity=0.6))

        fig.update_layout(
            title=title,
            xaxis_title="RT (min)",
            yaxis_title="Intensity",
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, width="stretch", key=chart_key)
        return

    import matplotlib.pyplot as plt  # fallback

    fig, ax = plt.subplots()
    for _, r in rep_df.iterrows():
        rt = _as_list(r.get("rt"))
        y = _as_list(r.get("intensity"))
        rt2, y2 = _downsample_xy(rt, y, max_points=max_points)
        ax.plot(rt2, y2, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("RT (min)")
    ax.set_ylabel("Intensity")
    st.pyplot(fig, clear_figure=True)


# -------------------------
# Cached loading (pickle-safe)
# -------------------------
@st.cache_data(show_spinner=False)
def load_traces(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


@st.cache_data(show_spinner=False)
def load_unique_sheet(xlsx_path: str, unique_sheet: str) -> pd.DataFrame:
    # returns a DataFrame (pickle-serializable)
    return pd.read_excel(xlsx_path, sheet_name=unique_sheet, engine="openpyxl")


@st.cache_data(show_spinner=False)
def load_hit_site_sets(xlsx_path: str) -> tuple[set[str], set[str], set[str]]:
    """
    Returns (hit_sites_ms2, hit_sites_ms3, hit_sites_shared)
    All are pickle-serializable.
    """

    def _sheet_sites(sheet_name: str) -> set[str]:
        try:
            s = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
        except Exception:
            return set()
        if "Site" not in s.columns:
            return set()
        return set(s["Site"].dropna().astype(str))

    hit_sites_ms2 = _sheet_sites("Hits_MS2")
    hit_sites_ms3 = _sheet_sites("Hits_MS3")
    hit_sites_shared = hit_sites_ms2.intersection(hit_sites_ms3)
    return hit_sites_ms2, hit_sites_ms3, hit_sites_shared


def render_viewer(
    *,
    xlsx_path: Path,
    traces_path: Path,
    unique_sheet: str = DEFAULT_UNIQUE_SHEET,
    title: Optional[str] = "LC/MS Chromatogram Review — Site Browser",
    embedded: bool = True,
) -> None:
    """
    Render the chromatogram viewer inside another Streamlit app.

    - No set_page_config here (wizard owns that).
    - No reliance on fixed filenames; caller provides paths.
    - Uses only pickle-serializable cached returns (no pd.ExcelFile).
    """

    xlsx_path = Path(xlsx_path)
    traces_path = Path(traces_path)

    if title:
        st.subheader(title) if embedded else st.title(title)

    # Validate inputs
    if not xlsx_path.exists():
        st.error(f"Missing workbook: `{xlsx_path}`")
        return
    if not traces_path.exists():
        st.error(f"Missing traces parquet: `{traces_path}` (did Step 4 run? is pyarrow installed?)")
        return

    # Load traces
    try:
        traces = load_traces(str(traces_path))
    except Exception as e:
        st.error(f"Failed to read parquet: `{traces_path}`\n\n{e}")
        return

    # Load Unique_Sites sheet (pickle-safe)
    try:
        unique = load_unique_sheet(str(xlsx_path), unique_sheet)
    except Exception as e:
        st.error(f"Failed to read '{unique_sheet}' from `{xlsx_path}`\n\n{e}")
        return

    # Load hit site sets (pickle-safe)
    hit_sites_ms2, hit_sites_ms3, hit_sites_shared = load_hit_site_sets(str(xlsx_path))

    # Normalize columns
    for df in (traces, unique):
        for c in [
            "trace_id",
            "ms_level",
            "Gene Name",
            "Site",
            "call",
            "FileName",
            "Uniprot ID",
            "PeptideVariant",
            "Best Peptide",
            SITE_CALL_COL,
            "AA_len",
            "Cys_count",
        ]:
            safe_str_col(df, c)

    for c in SCORE_COLS + LFC_COLS:
        to_numeric_col(unique, c)

    if "Site" not in traces.columns:
        st.error(f"Traces parquet is missing a 'Site' column. Columns: {list(traces.columns)}")
        return

    traces["ms_level"] = traces["ms_level"].apply(normalize_ms_level)
    traces["call"] = traces["call"].apply(normalize_call)
    traces["Site_norm"] = traces["Site"].apply(normalize_site)

    unique["Site_norm"] = unique["Site"].apply(normalize_site) if "Site" in unique.columns else ""
    if SITE_CALL_COL in unique.columns:
        unique[SITE_CALL_COL] = unique[SITE_CALL_COL].apply(normalize_call)

    # Precompute replicate group once (speed win)
    if "replicate_group" not in traces.columns:
        traces["replicate_group"] = traces.apply(
            lambda r: infer_window_from_filename(r.get("FileName", ""), r.get("ms_level", "")),
            axis=1,
        )

    # Build site table from Unique_Sites
    keep_cols = []
    base_cols = [
        "Site",
        "Site_norm",
        "Gene Name",
        "Uniprot ID",
        "Best Peptide",
        "AA_len",
        "Cys_count",
        SITE_CALL_COL,
    ] + SCORE_COLS + LFC_COLS
    for c in base_cols:
        if c in unique.columns:
            keep_cols.append(c)

    for c in ["sky_split_call_MS2", "sky_split_call_MS3"]:
        if c in unique.columns:
            keep_cols.append(c)

    if not keep_cols:
        st.error(f"No expected columns found in '{unique_sheet}'. Columns: {list(unique.columns)}")
        return

    sites = unique[keep_cols].drop_duplicates(subset=["Site_norm"], keep="first").copy()
    sites["Site"] = sites["Site"].astype(str)
    sites["Site_norm"] = sites["Site_norm"].astype(str)

    sites["is_hit_ms2"] = sites["Site"].isin(hit_sites_ms2)
    sites["is_hit_ms3"] = sites["Site"].isin(hit_sites_ms3)
    sites["is_hit_shared"] = sites["Site"].isin(hit_sites_shared)

    # -------------------------
    # Controls (embedded: use expander to avoid hijacking wizard sidebar)
    # -------------------------
    ctrl_container = st.sidebar if not embedded else st.expander("Viewer controls", expanded=False)
    with ctrl_container:
        st.subheader("Global filters")
        gene_q = st.text_input("Gene contains", "", key="viewer_gene_q")
        site_q = st.text_input("Site contains", "", key="viewer_site_q")

        st.subheader("Trace filters")
        ms_levels = sorted([x for x in traces["ms_level"].unique() if x])
        ms_sel = st.multiselect("MS level", ms_levels, default=ms_levels, key="viewer_ms_sel")

        st.subheader("Split call filter (SITE-LEVEL)")
        if SITE_CALL_COL in sites.columns:
            call_levels = sorted([c for c in sites[SITE_CALL_COL].dropna().astype(str).unique().tolist() if c])
            if not call_levels:
                call_levels = ["Single", "Partial", "Complete"]
        else:
            call_levels = ["Single", "Partial", "Complete"]
        call_sel = st.multiselect("Combined/Worst call", call_levels, default=call_levels, key="viewer_call_sel")

        st.subheader("Plot throttles (speed)")
        max_traces_per_overlay = st.slider("Max traces per replicate overlay", 1, 40, 12, key="viewer_max_traces")
        max_points_per_trace = st.slider(
            "Max points per trace (downsample)", 200, 5000, 1500, step=100, key="viewer_max_pts"
        )

        st.subheader("Sorting")
        sort_mode = st.selectbox(
            "Sort sites by",
            options=[
                "Shared first, then S_Score_MS3 (desc), then S_Score_MS2 (desc)",
                "S_Score_MS3 (desc)",
                "S_Score_MS2 (desc)",
                "Trace count (filtered) (desc)",
                "None",
            ],
            index=0,
            key="viewer_sort_mode",
        )

        st.subheader("View mode")
        mode = st.radio(
            "Mode",
            ["Paged expanders (10/page)", "Single-site picker (fast)"],
            index=0,
            key="viewer_mode",
        )

    # -------------------------
    # Traces filtered ONLY by ms_sel
    # -------------------------
    traces_filt = traces[traces["ms_level"].isin(ms_sel)].copy()

    # Fast index: Site_norm -> traces rows (filtered)
    site_to_traces = {k: g for k, g in traces_filt.groupby("Site_norm", sort=False)}

    site_trace_summary = (
        traces_filt.groupby("Site_norm")
        .agg(n_traces_filtered=("trace_id", "count"))
        .reset_index()
    )

    site_trace_total = (
        traces.groupby("Site_norm")
        .agg(n_traces_total=("trace_id", "count"))
        .reset_index()
    )

    sites2 = (
        sites.merge(site_trace_total, on="Site_norm", how="left")
        .merge(site_trace_summary, on="Site_norm", how="left")
    )
    sites2["n_traces_total"] = pd.to_numeric(sites2["n_traces_total"], errors="coerce").fillna(0).astype(int)
    sites2["n_traces_filtered"] = pd.to_numeric(sites2["n_traces_filtered"], errors="coerce").fillna(0).astype(int)

    # SITE-level call filter
    if SITE_CALL_COL in sites2.columns and call_sel:
        sites2 = sites2[sites2[SITE_CALL_COL].astype(str).isin(call_sel)].copy()

    # Global filters
    if gene_q and "Gene Name" in sites2.columns:
        sites2 = sites2[sites2["Gene Name"].astype(str).str.contains(gene_q, case=False, na=False)]
    if site_q:
        sites2 = sites2[sites2["Site"].astype(str).str.contains(site_q, case=False, na=False)]

    # Remove sites with no traces after MS filter
    sites2 = sites2[sites2["n_traces_filtered"] > 0].copy()

    # Sorting
    if sort_mode == "Shared first, then S_Score_MS3 (desc), then S_Score_MS2 (desc)":
        by = ["is_hit_shared"]
        asc = [False]
        if "S_Score_MS3" in sites2.columns:
            by.append("S_Score_MS3")
            asc.append(False)
        if "S_Score_MS2" in sites2.columns:
            by.append("S_Score_MS2")
            asc.append(False)
        sites2 = sites2.sort_values(by=by, ascending=asc, na_position="last")
    elif sort_mode == "S_Score_MS3 (desc)" and "S_Score_MS3" in sites2.columns:
        sites2 = sites2.sort_values(by="S_Score_MS3", ascending=False, na_position="last")
    elif sort_mode == "S_Score_MS2 (desc)" and "S_Score_MS2" in sites2.columns:
        sites2 = sites2.sort_values(by="S_Score_MS2", ascending=False, na_position="last")
    elif sort_mode == "Trace count (filtered) (desc)":
        sites2 = sites2.sort_values(by="n_traces_filtered", ascending=False, na_position="last")

    # -------------------------
    # Rendering
    # -------------------------
    tab_ms3, tab_ms2, tab_shared = st.tabs(["MS3 hits", "MS2 hits", "Shared (MS2 ∩ MS3)"])

    def render_one_site(srow: pd.Series, panel_key: str, row_idx: int):
        site_norm = str(srow.get("Site_norm", "")).strip()
        site_disp = str(srow.get("Site", "")).strip()
        gene = str(srow.get("Gene Name", "")).strip()
        best_pep = str(srow.get("Best Peptide", "")).strip()
        site_call = str(srow.get(SITE_CALL_COL, "")).strip()

        tags = []
        if bool(srow.get("is_hit_shared", False)):
            tags.append("SHARED")
        if bool(srow.get("is_hit_ms3", False)):
            tags.append("MS3")
        if bool(srow.get("is_hit_ms2", False)):
            tags.append("MS2")

        label = f"{gene} | {site_disp}"
        if site_call:
            label += f" | {site_call}"
        if tags:
            label += " | " + ",".join(tags)
        label += f" | traces={int(srow.get('n_traces_filtered', 0))}"
        label += f"  [{panel_key}:{row_idx}]"

        with st.expander(label, expanded=False):
            if best_pep:
                st.markdown(f"**Best Peptide:** `{best_pep}`")

            aa_len = srow.get("AA_len", "")
            cys_count = srow.get("Cys_count", "")
            if str(aa_len).strip() or str(cys_count).strip():
                st.markdown(f"**AA_len:** `{aa_len}` &nbsp;&nbsp; **Cys_count:** `{cys_count}`")

            if site_call:
                st.markdown(f"**Combined/Worst split call:** `{site_call}`")

            c1, c2 = st.columns([1, 1])
            with c1:
                st.write("**Scores**")
                st.write({"S_Score_MS3": srow.get("S_Score_MS3", None), "S_Score_MS2": srow.get("S_Score_MS2", None)})
            with c2:
                st.write("**Log2FC**")
                lfc_present = [c for c in LFC_COLS if c in srow.index]
                lfc_table = pd.DataFrame({"window": lfc_present, "Log2FC": [srow.get(c) for c in lfc_present]})
                st.dataframe(lfc_table, width="stretch", hide_index=True)

            t = site_to_traces.get(site_norm)
            if t is None or len(t) == 0:
                st.warning("No traces remain for this site after current MS-level filter.")
                return

            t2 = t[t["replicate_group"].isin(WINDOW_ORDER)].copy()
            if len(t2) == 0:
                st.warning("Traces exist, but none could be assigned to the 5 replicate groups.")
                st.write("Example FileName values:", list(t["FileName"].dropna().unique())[:10])
                return

            present_groups = [g for g in WINDOW_ORDER if g in set(t2["replicate_group"])]
            st.subheader("Replicate chromatograms (overlay precursor traces)")

            cols = st.columns(2)
            ci = 0
            for g in present_groups:
                rep_df = t2[t2["replicate_group"] == g].copy()
                if "ProductMz" in rep_df.columns:
                    rep_df = rep_df.sort_values(by=["ProductMz"], ascending=True)
                with cols[ci % 2]:
                    plot_replicate_overlay(
                        rep_df,
                        title=f"{gene} {site_disp} — {g} ({len(rep_df)} precursor traces)",
                        max_traces=max_traces_per_overlay,
                        max_points=max_points_per_trace,
                        chart_key=f"plot_{panel_key}_{row_idx}_{g}_{ci}",
                    )
                ci += 1

    def _init_page_state(panel_key: str):
        k = f"viewer_page_{panel_key}"
        if k not in st.session_state:
            st.session_state[k] = 1

    def _page_nav(panel_key: str, total_pages: int):
        _init_page_state(panel_key)
        k = f"viewer_page_{panel_key}"

        cprev, cjump, cinfo, cnext = st.columns([1, 2, 3, 1])

        with cprev:
            if st.button("⬅️", key=f"viewer_prev_{panel_key}", disabled=(st.session_state[k] <= 1)):
                st.session_state[k] = max(1, st.session_state[k] - 1)

        with cjump:
            # Use a separate widget key so it doesn't fight session_state[k]
            jump_key = f"viewer_jump_{panel_key}"
            default_val = int(st.session_state[k])

            # number_input returns int when step is int and value is int
            target = st.number_input(
                "Go to page",
                min_value=1,
                max_value=max(1, int(total_pages)),
                value=min(max(1, default_val), int(total_pages)),
                step=1,
                key=jump_key,
                label_visibility="collapsed",
            )
            if st.button("Go", key=f"viewer_go_{panel_key}"):
                st.session_state[k] = int(target)

        with cinfo:
            st.markdown(f"**Page {st.session_state[k]} / {total_pages}**")

        with cnext:
            if st.button("➡️", key=f"viewer_next_{panel_key}", disabled=(st.session_state[k] >= total_pages)):
                st.session_state[k] = min(total_pages, st.session_state[k] + 1)


    def render_panel(panel_df: pd.DataFrame, panel_key: str, title2: str):
        st.subheader(title2)
        st.write(f"Sites shown: **{len(panel_df)}** (filtered by MS + site-level call selection)")

        show_cols = ["Site", "Uniprot ID", "Best Peptide", "AA_len", "Cys_count"]
        for c in LFC_COLS:
            if c in panel_df.columns:
                show_cols.append(c)

        show_cols += [
            "is_hit_ms2",
            "is_hit_ms3",
            SITE_CALL_COL,
            "S_Score_MS3",
            "S_Score_MS2",
            "n_traces_total",
            "n_traces_filtered",
        ]
        for c in ["sky_split_call_MS2", "sky_split_call_MS3"]:
            if c in panel_df.columns:
                show_cols.append(c)

        show_cols = [c for c in show_cols if c in panel_df.columns]

        hit_df = panel_df[show_cols].copy()
        hit_df.insert(0, "#", range(1, len(hit_df) + 1))
        st.dataframe(hit_df, width="stretch", height=260, hide_index=True)

        st.divider()

        if mode == "Single-site picker (fast)":
            opts = [
                f"{r.get('Site','')} | {r.get(SITE_CALL_COL,'')} | {r.get('Best Peptide','')} | traces={r.get('n_traces_filtered',0)}"
                for _, r in panel_df.iterrows()
            ]
            if not opts:
                st.info("No sites to display.")
                return
            pick = st.selectbox("Site", opts, index=0, key=f"viewer_sitepick_{panel_key}")
            idx = opts.index(pick)
            render_one_site(panel_df.iloc[idx], panel_key=panel_key, row_idx=idx)
            return

        per_page = 10
        n = len(panel_df)
        if n == 0:
            st.info("No sites to display.")
            return

        total_pages = max(1, math.ceil(n / per_page))
        _page_nav(panel_key, total_pages)

        page = st.session_state[f"viewer_page_{panel_key}"]
        start_idx = (page - 1) * per_page
        end_idx = min(page * per_page, n)
        page_df = panel_df.iloc[start_idx:end_idx]

        st.caption(f"Showing sites {start_idx + 1}–{end_idx} of {n}")
        st.subheader("Expand a site to view replicate chromatograms")
        for i, (_, row) in enumerate(page_df.iterrows()):
            render_one_site(row, panel_key=panel_key, row_idx=start_idx + i)

    with tab_ms3:
        render_panel(sites2[sites2["is_hit_ms3"]].copy(), "ms3", "MS3 hits (inclusive)")
    with tab_ms2:
        render_panel(sites2[sites2["is_hit_ms2"]].copy(), "ms2", "MS2 hits (inclusive)")
    with tab_shared:
        render_panel(sites2[sites2["is_hit_shared"]].copy(), "shared", "Shared hits (MS2 ∩ MS3)")
