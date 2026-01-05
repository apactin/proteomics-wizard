from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Any

import io
from datetime import datetime

import pandas as pd
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
            _venn_ms2_ms3(ms2_single, ms3_single, "Hit overlap (Single sites only, by Site)", download_name=f"venn_single_{run_tag}.png")
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
    # Plot: Split call distribution (simple, original meaning) + download
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

            # counts per gene
            g = (
                df_g.groupby("Gene Name", dropna=False)
                .agg(
                    n_unique=("Site", "count"),
                    n_ms2=("is_ms2_hit", "sum"),
                    n_ms3=("is_ms3_hit", "sum"),
                    n_shared=("is_shared", "sum"),
                    n_shared_single=("is_shared", lambda s: int(((df_g.loc[s.index, "is_shared"]) & (df_g.loc[s.index, "is_single"])).sum())),
                )
                .reset_index()
            )
            # Remove blank gene names (optional)
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

                # Presence count
                df_w["n_windows_present"] = df_w[lfc_cols].notna().sum(axis=1)

                # Sign agreement: fraction of non-NaN windows matching majority sign
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
                    ax.hist(df_w["n_windows_present"].dropna().values, bins=range(0, int(df_w["n_windows_present"].max() or 0) + 2))
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

                # Optional: show a small table of worst agreement
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

        # Prepare arrays per split type
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
                # Correlations
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
                # Robust thresholds via quantiles
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

                cols = [c for c in ["Gene Name", "Site", "_split_type", "S_Score_MS2", "S_Score_MS3", "Uniprot ID", "Best Peptide"] if c in df_d.columns]

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
