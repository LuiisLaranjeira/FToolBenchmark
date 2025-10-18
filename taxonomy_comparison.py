#!/usr/bin/env python3
"""
Taxonomy tool comparison.

Summary
-------
Evaluate each tool under each condition using a shared taxon universe that guarantees false negatives are counted.
Prioritize PR–AUC; also report ROC–AUC (interpret ROC–AUC cautiously when negatives are scarce).

Outputs (--outdir)
------------------
- tool_macro_metrics.csv      # macro-averaged across conditions (PRIMARY)
- tool_micro_metrics.csv      # micro-pooled across conditions (SECONDARY)
- group_core_metrics.csv      # per (tool, depth, read, deam_key, condition)
- curves/roc_<tool>.png, curves/pr_<tool>.png
- factor_plots_auc_pr.png, factor_plots_auc_roc.png, factor_plots_f1.png

CLI
---
python3 taxonomy_comparison.py \
  --root-dir . \
  --ground-truth ground_truth.txt \
  --outdir evaluation_results \
  --fixed-threshold 0.0 \
  --export-curves
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ----------------------- Logging -----------------------
LOG_FMT = "%(asctime)s | %(levelname)s | %(message)s"
logger = logging.getLogger("taxonomy_eval_uc")
logger.setLevel(logging.INFO)

# ----------------------- Condition name regex -----------------------
import re as _re

COND_RE = _re.compile(
    r"^depth(?P<depth>\d+)_read(?P<read>\d+)_deam(?P<deam>(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)$"
)

# near top of file
_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', 'H', 'd', '8']


def _marker_for(tool: str) -> str:
    return _MARKERS[hash(tool) % len(_MARKERS)]


def _bounded_ylim(metric: str):
    return (0.0, 1.05) if metric in {"auc_prc", "auc_roc", "f1", "precision", "recall"} else None


# ----------------------- Small utils -----------------------
def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def normalize_taxid_column(df: pd.DataFrame) -> pd.DataFrame:
    for cand in ("taxid", "TaxID", "tax_id", "taxId", "TAXID", "taxID"):
        if cand in df.columns and cand != "taxID":
            df = df.rename(columns={cand: "taxID"})
            break
    return df


def load_id_list(path: Path, label: str) -> List[str]:
    ids: List[str] = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if s and s.isdigit():
                ids.append(s)
    logger.info("Loaded %d %s", len(ids), label)
    return ids


# ----------------------- I/O: tool tables -----------------------
def load_count_tables(root_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Expect subdirs: <tool>_output/count_table.tsv with columns:
    taxID + one column per condition (matching COND_RE).
    """
    tool_tables: Dict[str, pd.DataFrame] = {}
    for item in root_dir.iterdir():
        if not (item.is_dir() and item.name.lower().endswith("_output")):
            continue
        tool = item.name[: -len("_output")]
        count_file = item / "count_table.tsv"
        if not count_file.exists():
            logger.warning("Skipping %s: missing %s", item, count_file.name)
            continue

        try:
            df = pd.read_csv(count_file, sep="\t")
        except Exception as e:
            logger.error("Error loading %s: %s", count_file, e)
            continue

        df = normalize_taxid_column(df)
        if "taxID" not in df.columns:
            logger.warning("Skipping %s: no 'taxID' col", count_file)
            continue

        cond_cols = [c for c in df.columns if c != "taxID" and COND_RE.fullmatch(c)]
        if not cond_cols:
            logger.warning("Skipping tool %s: no condition columns match %s", tool, COND_RE.pattern)
            continue

        keep = ["taxID"] + cond_cols
        df = df[keep].copy()
        df["taxID"] = df["taxID"].astype("string")
        tool_tables[tool] = df
        logger.info("Loaded tool %s: %d taxa, %d conditions", tool, len(df), len(cond_cols))
    return tool_tables


# ----------------------- Reshape & conditions -----------------------
def melt_to_long(tool_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide -> long rows: ['tool','condition','taxID','count']."""
    frames: List[pd.DataFrame] = []
    for tool, df in tool_tables.items():
        cond_cols = [c for c in df.columns if c != "taxID" and COND_RE.fullmatch(c)]
        long = df.melt(id_vars=["taxID"], value_vars=cond_cols,
                       var_name="condition", value_name="count")
        long["tool"] = tool
        frames.append(long)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out

    counts = pd.to_numeric(out["count"], errors="coerce")
    good = counts.notna() & np.isfinite(counts)
    dropped = int((~good).sum())
    if dropped:
        logger.warning("Dropping %d non-finite counts", dropped)
    out = out.loc[good].copy()
    out["count"] = counts.loc[good]
    if (out["count"] < 0).any():
        raise ValueError("Negative counts encountered.")
    return out


def parse_condition(c: str) -> Tuple[int, int, str, float]:
    m = COND_RE.fullmatch(c)
    if not m:
        raise ValueError(f"Bad condition name: {c}")
    depth = int(m.group("depth"))
    read = int(m.group("read"))
    deam_key = str(m.group("deam"))
    deam = float(deam_key)
    return depth, read, deam_key, deam


def list_tools_and_conditions(long_df: pd.DataFrame) -> Tuple[List[str], List[str], pd.DataFrame]:
    tools = sorted(long_df["tool"].unique().tolist())
    conditions = sorted(long_df["condition"].unique().tolist())
    meta_rows = []
    for cond in conditions:
        d, r, dk, dv = parse_condition(cond)
        meta_rows.append({"condition": cond, "depth": d, "read": r, "deam_key": dk, "deam": dv})
    cond_meta = pd.DataFrame(meta_rows)
    return tools, conditions, cond_meta


# ----------------------- Per-condition universe U_c -----------------------
def build_universe_by_condition(long_df: pd.DataFrame, GT_set: Set[str]) -> Dict[str, List[str]]:
    """
    Build the candidate taxon list for each condition.

    For a given condition, the candidate list includes:
      • all ground-truth taxa; and
      • any taxa reported by any tool under that condition.

    Returns
    -------
    dict[str, list[str]]
        Maps condition -> list of taxIDs (sorted numerically).
    """
    universes: Dict[str, List[str]] = {}
    # For each condition, collect everything reported by any tool, plus GT
    for cond, g in long_df.groupby("condition", as_index=False):
        reported = set(g["taxID"].astype(str).unique())
        Uc = sorted(GT_set | reported)
        universes[cond] = Uc
    return universes


def build_vectors_for_group(
        tool_cond_df: pd.DataFrame,
        Uc_sorted: List[str],
        GT_set: Set[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create aligned label and score vectors for one (tool, condition).

    Parameters
    ----------
    tool_cond_df : DataFrame
        Must have columns ['taxID', 'count'] for a single tool and condition.
    Uc_sorted : list[str]
        The candidate taxon list for this condition (order matters).
    GT_set : set[str]
        Ground-truth taxIDs.

    Returns
    -------
    y_true : np.ndarray[int]
        1 where the taxon is in the ground truth, 0 otherwise, aligned to Uc_sorted.
    y_scores : np.ndarray[float]
        The tool’s score for each taxon in Uc_sorted; 0.0 where the tool did not report it.
    """
    scores = {str(t): float(v) for t, v in zip(tool_cond_df["taxID"], tool_cond_df["count"])}
    y_scores = np.array([scores.get(tid, 0.0) for tid in Uc_sorted], dtype=float)
    y_true = np.array([1 if tid in GT_set else 0 for tid in Uc_sorted], dtype=int)
    return y_true, y_scores


# ----------------------- Threshold & metrics -----------------------
def select_predictions(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        fixed_threshold: Optional[float] = None,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    If fixed_threshold is provided, use it (>=). Otherwise maximize F1 over PR breakpoints,
    including predict-none/all extremes.
    """
    n_pos = int(y_true.sum())
    n = int(len(y_true))
    if n == 0:
        raise ValueError("Empty vectors for metrics.")
    if n_pos == 0:
        y_pred = np.zeros_like(y_true)
        thr = float(np.inf) if fixed_threshold is None else fixed_threshold
        return thr, float(precision_score(y_true, y_pred, zero_division=0)), float(
            recall_score(y_true, y_pred, zero_division=0)), float(f1_score(y_true, y_pred, zero_division=0)), y_pred
    if n_pos == n:
        y_pred = np.ones_like(y_true)
        thr = float(-np.inf) if fixed_threshold is None else fixed_threshold
        return thr, float(precision_score(y_true, y_pred, zero_division=0)), float(
            recall_score(y_true, y_pred, zero_division=0)), float(f1_score(y_true, y_pred, zero_division=0)), y_pred

    if fixed_threshold is not None:
        y_pred = (y_scores >= fixed_threshold).astype(int)
        return fixed_threshold, float(precision_score(y_true, y_pred, zero_division=0)), float(
            recall_score(y_true, y_pred, zero_division=0)), float(f1_score(y_true, y_pred, zero_division=0)), y_pred

    prec, rec, thresholds = precision_recall_curve(y_true, y_scores)
    if thresholds.size == 0:
        preds = [np.zeros_like(y_true), np.ones_like(y_true)]
        f1s = [f1_score(y_true, p, zero_division=0) for p in preds]
        best = int(np.argmax(f1s))
        y_pred = preds[best]
        thr = float(np.inf if best == 0 else -np.inf)
        return thr, float(precision_score(y_true, y_pred, zero_division=0)), float(
            recall_score(y_true, y_pred, zero_division=0)), float(f1s[best]), y_pred

    cand_thresholds = list(thresholds) + [np.inf, -np.inf]
    cand_preds = [(y_scores >= t).astype(int) for t in thresholds] + [
        (y_scores >= np.inf).astype(int),
        (y_scores >= -np.inf).astype(int),
    ]
    f1_vals = np.array([f1_score(y_true, p, zero_division=0) for p in cand_preds], dtype=float)
    best_idx = int(np.argmax(f1_vals))
    best_thr = float(cand_thresholds[best_idx])
    y_pred = cand_preds[best_idx]
    return best_thr, float(precision_score(y_true, y_pred, zero_division=0)), float(
        recall_score(y_true, y_pred, zero_division=0)), float(f1_vals[best_idx]), y_pred


def compute_metrics(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        fixed_threshold: Optional[float],
) -> Dict[str, float]:
    """Return auc_prc, auc_roc, threshold_used, precision, recall, f1, tp, fp, fn, tn."""
    has_both = (len(np.unique(y_true)) == 2)
    auc_roc = float(roc_auc_score(y_true, y_scores)) if has_both else np.nan
    auc_prc = float(average_precision_score(y_true, y_scores)) if has_both else np.nan
    thr, p_at, r_at, f1_at, y_pred = select_predictions(y_true, y_scores, fixed_threshold)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    return {
        "auc_prc": auc_prc,
        "auc_roc": auc_roc,
        "threshold_used": float(thr),
        "precision": float(p_at),
        "recall": float(r_at),
        "f1": float(f1_at),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ----------------------- Evaluation -----------------------
def evaluate_per_group(
        long_df: pd.DataFrame,
        tools: List[str],
        conditions: List[str],
        cond_meta: pd.DataFrame,
        universes: Dict[str, List[str]],
        GT_set: Set[str],
        fixed_threshold: Optional[float],
) -> pd.DataFrame:
    """Metrics per (tool, condition) using its U_c."""
    meta_map = cond_meta.set_index("condition").to_dict(orient="index")
    rows: List[dict] = []
    by_tool = {t: long_df[long_df["tool"] == t] for t in tools}

    for tool in tools:
        df_t = by_tool[tool]
        for cond in conditions:
            Uc = universes[cond]
            g = df_t[df_t["condition"] == cond][["taxID", "count"]]
            y_true, y_scores = build_vectors_for_group(g, Uc, GT_set)
            metrics = compute_metrics(y_true, y_scores, fixed_threshold)
            n_pos = int(y_true.sum())
            n_neg = int(len(y_true) - n_pos)
            info = {
                "tool": tool,
                "condition": cond,
                "depth": meta_map[cond]["depth"],
                "read": meta_map[cond]["read"],
                "deam_key": meta_map[cond]["deam_key"],
                "deam": meta_map[cond]["deam"],
                "n_taxa_uc": len(Uc),
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
            info.update(metrics)
            rows.append(info)
    return pd.DataFrame(rows)


def macro_average(group_df: pd.DataFrame) -> pd.DataFrame:
    """Unweighted mean across conditions per tool (PRIMARY)."""
    keep = ["auc_prc", "auc_roc", "precision", "recall", "f1"]
    agg = (group_df.groupby("tool", as_index=False)[keep]
           .mean(numeric_only=True))
    agg.rename(columns={
        "auc_prc": "pr_auc_macro",
        "auc_roc": "roc_auc_macro",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
    }, inplace=True)
    return agg


def evaluate_micro(
        long_df: pd.DataFrame,
        tools: List[str],
        conditions: List[str],
        universes: Dict[str, List[str]],
        GT_set: Set[str],
        fixed_threshold: Optional[float],
) -> pd.DataFrame:
    """Micro-pooled across ALL conditions per tool (SECONDARY)."""
    rows: List[dict] = []
    by_tool = {t: long_df[long_df["tool"] == t] for t in tools}

    for tool in tools:
        df_t = by_tool[tool]
        y_true_all: List[np.ndarray] = []
        y_score_all: List[np.ndarray] = []
        for cond in conditions:
            Uc = universes[cond]
            g = df_t[df_t["condition"] == cond][["taxID", "count"]]
            y_true, y_scores = build_vectors_for_group(g, Uc, GT_set)
            y_true_all.append(y_true)
            y_score_all.append(y_scores)
        y_true_cat = np.concatenate(y_true_all, axis=0)
        y_score_cat = np.concatenate(y_score_all, axis=0)

        metrics = compute_metrics(y_true_cat, y_score_cat, fixed_threshold)
        info = {
            "tool": tool,
            "n_samples": int(len(y_true_cat)),
            "n_positives": int(y_true_cat.sum()),
            "n_negatives": int(len(y_true_cat) - int(y_true_cat.sum())),
            "prevalence": float(y_true_cat.mean()),
        }
        info.update(metrics)
        rows.append(info)
    return pd.DataFrame(rows)


# ----------------------- Plots -----------------------
def export_tool_curves_micro(
        long_df: pd.DataFrame,
        tools: List[str],
        conditions: List[str],
        universes: Dict[str, List[str]],
        GT_set: Set[str],
        outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    by_tool = {t: long_df[long_df["tool"] == t] for t in tools}

    for tool in tools:
        df_t = by_tool[tool]
        y_true_all: List[np.ndarray] = []
        y_score_all: List[np.ndarray] = []
        for cond in conditions:
            Uc = universes[cond]
            g = df_t[df_t["condition"] == cond][["taxID", "count"]]
            y_true, y_scores = build_vectors_for_group(g, Uc, GT_set)
            y_true_all.append(y_true)
            y_score_all.append(y_scores)
        y_true_cat = np.concatenate(y_true_all, axis=0)
        y_score_cat = np.concatenate(y_score_all, axis=0)

        if len(np.unique(y_true_cat)) < 2:
            logger.warning("Skipping curves for %s: only one class present", tool)
            continue

        # ROC
        fpr, tpr, _ = roc_curve(y_true_cat, y_score_cat)
        auc_roc = roc_auc_score(y_true_cat, y_score_cat)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC — {tool} (AUC={auc_roc:.3f})")
        plt.tight_layout()
        plt.savefig(outdir / f"roc_{tool}.png", dpi=200, bbox_inches="tight")
        plt.close()

        # PR
        prec_arr, rec_arr, _ = precision_recall_curve(y_true_cat, y_score_cat)
        auc_pr = average_precision_score(y_true_cat, y_score_cat)
        plt.figure(figsize=(6, 5))
        plt.plot(rec_arr, prec_arr, lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR — {tool} (AP={auc_pr:.3f})")
        plt.tight_layout()
        plt.savefig(outdir / f"pr_{tool}.png", dpi=200, bbox_inches="tight")
        plt.close()


def factor_plots(group_df: pd.DataFrame, outdir: Path) -> None:
    """Three 2x2-style figures: AUC–PRC, AUC–ROC, F1 vs (depth, read, deam)."""
    outdir.mkdir(parents=True, exist_ok=True)

    def _plot(metric: str, path: Path) -> None:
        df = group_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric]).copy()
        if df.empty:
            logger.warning("No values for %s; skipping", metric)
            return

        tools = sorted(df["tool"].unique().tolist())
        plt.figure(figsize=(13, 8))

        # 1) vs depth
        ax = plt.subplot(2, 2, 1)
        for t in tools:
            m = df[df["tool"] == t].groupby("depth", as_index=True)[metric].mean().sort_index()
            if not m.empty:
                #ax.plot(m.index.values, m.values, marker="o", label=t)
                ax.plot(m.index.values, m.values, marker=_marker_for(t), label=t)
                ylim = _bounded_ylim(metric)
                if ylim: ax.set_ylim(*ylim)

        ax.set_xlabel("Depth")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs Depth")
        ax.legend()

        # 2) vs read
        ax = plt.subplot(2, 2, 2)
        for t in tools:
            m = df[df["tool"] == t].groupby("read", as_index=True)[metric].mean().sort_index()
            if not m.empty:
                #ax.plot(m.index.values, m.values, marker="o", label=t)
                ax.plot(m.index.values, m.values, marker=_marker_for(t), label=t)
                ylim = _bounded_ylim(metric)
                if ylim: ax.set_ylim(*ylim)

        ax.set_xlabel("Read length (bp)")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs Read length")
        ax.legend()

        # 3) vs deamination
        ax = plt.subplot(2, 1, 2)
        for t in tools:
            m = df[df["tool"] == t].groupby("deam_key", as_index=True)[metric].mean()
            if not m.empty:
                xs = np.array([float(k) for k in m.index])
                order = np.argsort(xs)
                #ax.plot(xs[order], m.values[order], marker="o", label=t)
                ax.plot(xs[order], m.values[order], marker=_marker_for(t), label=t)
                ylim = _bounded_ylim(metric)
                if ylim: ax.set_ylim(*ylim)
        ax.set_xlabel("Deamination rate")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs Deamination")
        ax.legend()

        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", path.name)

    _plot("auc_prc", outdir / "factor_plots_auc_pr.png")
    _plot("auc_roc", outdir / "factor_plots_auc_roc.png")
    _plot("f1", outdir / "factor_plots_f1.png")


# ----------------------- Main -----------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate taxonomy tools.")
    p.add_argument("--root-dir", type=Path, default=Path("."), help="Directory with *_output/count_table.tsv")
    p.add_argument("--ground-truth", type=Path, default=Path("ground_truth.txt"), help="Path to ground truth file")
    p.add_argument("--outdir", type=Path, default=Path("evaluation_results"), help="Output directory")
    p.add_argument("--fixed-threshold", type=float, default=None,
                   help="If set, use this score threshold (>=) for F1; else, maximize F1 over PR breakpoints.")
    p.add_argument("--export-curves", action="store_true", help="Export micro-pooled ROC/PR curves per tool")
    args = p.parse_args(argv)

    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(LOG_FMT))
        logger.addHandler(h)

    tool_tables = load_count_tables(args.root_dir)
    if not tool_tables:
        logger.error("No tools with valid count tables found in %s", args.root_dir)
        return

    GT_list = load_id_list(args.ground_truth, "ground-truth taxa")
    if not GT_list:
        logger.error("Ground truth is empty.")
        return
    GT_set = set(GT_list)

    long_df = melt_to_long(tool_tables)
    if long_df.empty:
        logger.error("No long-format data (after melting).")
        return

    tools, conditions, cond_meta = list_tools_and_conditions(long_df)
    universes = build_universe_by_condition(long_df, GT_set)

    # Per-condition metrics
    group_df = evaluate_per_group(
        long_df=long_df,
        tools=tools,
        conditions=conditions,
        cond_meta=cond_meta,
        universes=universes,
        GT_set=GT_set,
        fixed_threshold=args.fixed_threshold,
    )
    atomic_write_csv(args.outdir / "group_core_metrics.csv", group_df)

    # Macro (PRIMARY)
    macro_df = macro_average(group_df)
    atomic_write_csv(args.outdir / "tool_macro_metrics.csv", macro_df)

    # Micro (SECONDARY)
    micro_df = evaluate_micro(
        long_df=long_df,
        tools=tools,
        conditions=conditions,
        universes=universes,
        GT_set=GT_set,
        fixed_threshold=args.fixed_threshold,
    )
    atomic_write_csv(args.outdir / "tool_micro_metrics.csv", micro_df)

    # Plots
    factor_plots(group_df, args.outdir)
    if args.export_curves:
        export_tool_curves_micro(
            long_df=long_df,
            tools=tools,
            conditions=conditions,
            universes=universes,
            GT_set=GT_set,
            outdir=args.outdir / "curves",
        )

    ranked = macro_df.sort_values(["pr_auc_macro", "roc_auc_macro", "f1_macro"],
                                  ascending=[False, False, False])
    logger.info("Top tools (MACRO AP → ROC-AUC → F1):\n%s",
                ranked[["tool", "pr_auc_macro", "roc_auc_macro", "f1_macro"]]
                .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
