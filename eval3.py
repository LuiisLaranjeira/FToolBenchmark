#!/usr/bin/env python3
"""
Taxonomy tool comparison (lite): core metrics + per-tool micro-curves + factor plots.

Outputs:
  - tool_core_metrics.csv: one row per tool (micro-pooled)
  - group_core_metrics.csv: one row per (tool, depth, read, deam_key)
  - curves/roc_<tool>.png, curves/pr_<tool>.png (if --export-curves)
  - factor_plots_auc_roc.png, factor_plots_f1.png: each has 3 subplots (vs depth/read/deam)

CLI:
  --root-dir, --ground-truth, --outdir, --export-curves
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

# ------------- Logging -------------
LOG_FMT = "%(asctime)s | %(levelname)s | %(message)s"
logger = logging.getLogger("taxonomy_eval_lite")
logger.setLevel(logging.INFO)

# ------------- Regex for condition columns -------------
import re as _re
CONDITION_REGEX = (
    r"^depth(?P<depth>\d+)_read(?P<read>\d+)"
    r"_deam(?P<deam>(?:\d+(?:\.\d+)?)(?:[eE][+-]?\d+)?)$"
)
_COND_RE = _re.compile(CONDITION_REGEX)


# ------------- Utilities -------------
def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _normalize_taxid_column(df: pd.DataFrame) -> pd.DataFrame:
    for cand in ("taxid", "TaxID", "tax_id", "taxId", "TAXID", "taxID"):
        if cand in df.columns:
            if cand != "taxID":
                df = df.rename(columns={cand: "taxID"})
            break
    return df


# ------------- I/O -------------
def load_count_tables(root_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all `count_table.tsv` under `*_output` subdirs.
    Keep only columns matching CONDITION_REGEX (+ taxID).
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

        df = _normalize_taxid_column(df)
        if "taxID" not in df.columns:
            logger.warning("Skipping %s: no 'taxID' column", count_file)
            continue

        cond_cols = [c for c in df.columns if c != "taxID" and _COND_RE.fullmatch(c)]
        if not cond_cols:
            logger.warning("Skipping tool %s: no columns match %s", tool, CONDITION_REGEX)
            continue

        kept = ["taxID"] + cond_cols
        df = df[kept].copy()
        df["taxID"] = df["taxID"].astype("string")
        df["tool"] = tool
        tool_tables[tool] = df
        logger.info("Loaded %d rows, %d conditions for tool %s", len(df), len(cond_cols), tool)

    return tool_tables


def load_ground_truth(path: Path) -> Set[str]:
    truth: Set[str] = set()
    try:
        with path.open("r") as f:
            for line in f:
                s = line.strip()
                if s and s.isdigit():
                    truth.add(s)
        logger.info("Loaded %d ground-truth taxa", len(truth))
    except Exception as e:
        logger.error("Error loading ground truth from %s: %s", path, e)
    return truth


# ------------- Reshape -------------
def melt_to_long(tool_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not tool_tables:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for tool, df in tool_tables.items():
        cond_cols = [c for c in df.columns if c not in {"taxID", "tool"} and _COND_RE.fullmatch(c)]
        if not cond_cols:
            logger.warning("Skipping tool %s in melt: no matching columns", tool)
            continue
        long = df.melt(id_vars=["taxID"], value_vars=cond_cols, var_name="condition", value_name="count")
        long["tool"] = tool
        frames.append(long)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)

    # Validate counts: numeric, finite, non-negative
    counts = pd.to_numeric(out["count"], errors="coerce")
    good = counts.notna() & np.isfinite(counts)
    dropped = int((~good).sum())
    if dropped:
        logger.warning("Dropping %d rows with non-finite counts", dropped)
    out = out.loc[good].copy()
    out["count"] = counts.loc[good]
    if (out["count"] < 0).any():
        raise ValueError("Negative counts found; counts must be non-negative.")

    return out


def parse_conditions(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract depth/read/deam from 'condition'.
    Keeps a canonical deam_key (string) and a float deam for sorting.
    """
    if long_df.empty:
        return long_df

    extracted = long_df["condition"].str.extract(CONDITION_REGEX)
    mask_ok = extracted.notna().all(axis=1)
    if (d := (len(long_df) - int(mask_ok.sum()))):
        logger.warning("Dropping %d rows with non-matching condition names", d)
    df = long_df.loc[mask_ok].copy()

    depth_num = pd.to_numeric(extracted.loc[mask_ok, "depth"], errors="coerce")
    read_num  = pd.to_numeric(extracted.loc[mask_ok, "read"],  errors="coerce")
    deam_str  = extracted.loc[mask_ok, "deam"].astype(str)
    deam_num  = pd.to_numeric(deam_str, errors="coerce")

    bad_num = depth_num.isna() | read_num.isna() | deam_num.isna()
    if int(bad_num.sum()):
        logger.warning("Dropping %d rows with non-numeric depth/read/deam", int(bad_num.sum()))
    df = df.loc[~bad_num].copy()
    df["depth"] = depth_num.loc[~bad_num].astype(int)
    df["read"]  = read_num.loc[~bad_num].astype(int)
    df["deam_key"] = deam_str.loc[~bad_num].astype(str)
    df["deam"] = deam_num.loc[~bad_num].astype(float)
    return df


# ------------- Threshold selection (aligned with PR curve) -------------
def select_threshold_max_f1(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Choose threshold that maximizes F1 over PR breakpoints, including extremes.
    Uses comparator consistent with PR construction (>=), and considers predict-none/all.
    Returns: threshold, precision, recall, f1, y_pred
    """
    n_pos = int(y_true.sum())
    n = int(len(y_true))
    if n_pos == 0:
        # No positives in truth → best is to predict none
        y_pred = np.zeros_like(y_true)
        thr = float(np.inf)  # document: +inf = predict-none
        p_at = float(precision_score(y_true, y_pred, zero_division=0))
        r_at = float(recall_score(y_true, y_pred, zero_division=0))
        f1_at = float(f1_score(y_true, y_pred, zero_division=0))
        return thr, p_at, r_at, f1_at, y_pred
    if n_pos == n:
        # All positives → best is predict-all
        y_pred = np.ones_like(y_true)
        thr = float(-np.inf)  # document: -inf = predict-all
        p_at = float(precision_score(y_true, y_pred, zero_division=0))
        r_at = float(recall_score(y_true, y_pred, zero_division=0))
        f1_at = float(f1_score(y_true, y_pred, zero_division=0))
        return thr, p_at, r_at, f1_at, y_pred

    prec, rec, thresholds = precision_recall_curve(y_true, y_scores)

    if thresholds.size == 0:
        # All scores identical: compare both constant policies
        preds = [np.zeros_like(y_true), np.ones_like(y_true)]
        f1s = [f1_score(y_true, p, zero_division=0) for p in preds]
        best = int(np.argmax(f1s))
        y_pred = preds[best]
        thr = float(np.inf if best == 0 else -np.inf)  # document as special
        return thr, float(precision_score(y_true, y_pred, zero_division=0)), float(recall_score(y_true, y_pred, zero_division=0)), float(f1s[best]), y_pred

    # Evaluate F1 directly at each threshold (>=), plus extremes
    cand_thresholds = list(thresholds) + [np.inf, -np.inf]
    cand_preds = [(y_scores >= t).astype(int) for t in thresholds] + [
        (y_scores >= np.inf).astype(int),      # all 0
        (y_scores >= -np.inf).astype(int),     # all 1
    ]
    f1_vals = np.array([f1_score(y_true, p, zero_division=0) for p in cand_preds], dtype=float)
    best_idx = int(np.argmax(f1_vals))
    best_thr = float(cand_thresholds[best_idx])
    y_pred = cand_preds[best_idx]
    p_at = float(precision_score(y_true, y_pred, zero_division=0))
    r_at = float(recall_score(y_true, y_pred, zero_division=0))
    f1_at = float(f1_vals[best_idx])
    return best_thr, p_at, r_at, f1_at, y_pred


# ------------- Evaluation -------------
def evaluate_per_tool(long_df: pd.DataFrame, ground_truth: Set[str]) -> pd.DataFrame:
    """
    Pool all rows per tool (micro-averaging) and compute core metrics.
    Returns a DataFrame: one row per tool.
    """
    if long_df.empty:
        return pd.DataFrame()

    rows: List[dict] = []
    for tool, df_t in long_df.groupby("tool", as_index=False):
        y_true = df_t["taxID"].isin(ground_truth).astype(int).to_numpy()
        y_scores = df_t["count"].astype(float).to_numpy()

        has_both = (len(np.unique(y_true)) == 2)
        if not has_both:
            logger.warning("Tool %s has only one class present after pooling; ROC/PR undefined", tool)

        auc_roc = float(roc_auc_score(y_true, y_scores)) if has_both else np.nan
        auc_prc = float(average_precision_score(y_true, y_scores)) if has_both else np.nan

        thr, p_at, r_at, f1_at, y_pred = select_threshold_max_f1(y_true, y_scores)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        rows.append({
            "tool": tool,
            "n_samples": int(len(y_true)),
            "n_positives": int(y_true.sum()),
            "n_negatives": int(len(y_true) - int(y_true.sum())),
            "prevalence": float(y_true.mean()),
            "auc_roc": auc_roc,
            "auc_prc": auc_prc,
            "optimal_threshold": float(thr),
            "precision": float(p_at),
            "recall": float(r_at),
            "f1": float(f1_at),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return pd.DataFrame(rows)


def evaluate_per_group(long_df: pd.DataFrame, ground_truth: Set[str]) -> pd.DataFrame:
    """
    Compute core metrics per (tool, depth, read, deam_key).
    Rows where only one class is present get NaN ROC/PR; F1 still computed via threshold search.
    """
    if long_df.empty:
        return pd.DataFrame()

    # Need parsed factors
    df = parse_conditions(long_df)
    rows: List[dict] = []

    for (tool, depth, read, deam_key), g in df.groupby(["tool", "depth", "read", "deam_key"], as_index=False):
        y_true = g["taxID"].isin(ground_truth).astype(int).to_numpy()
        y_scores = g["count"].astype(float).to_numpy()

        if len(y_true) == 0:
            continue

        has_both = (len(np.unique(y_true)) == 2)
        auc_roc = float(roc_auc_score(y_true, y_scores)) if has_both else np.nan
        auc_prc  = float(average_precision_score(y_true, y_scores)) if has_both else np.nan

        thr, p_at, r_at, f1_at, _ = select_threshold_max_f1(y_true, y_scores)

        rows.append({
            "tool": tool, "depth": int(depth), "read": int(read),
            "deam_key": str(deam_key), "deam": float(g["deam"].iloc[0]),
            "n_samples": int(len(y_true)),
            "auc_roc": auc_roc, "auc_prc": auc_prc,
            "precision": float(p_at), "recall": float(r_at), "f1": float(f1_at),
        })

    return pd.DataFrame(rows)


# ------------- Curves & Factor Plots -------------
def export_tool_curves(long_df: pd.DataFrame, ground_truth: Set[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for tool, df_t in long_df.groupby("tool", as_index=False):
        y_true = df_t["taxID"].isin(ground_truth).astype(int).to_numpy()
        y_scores = df_t["count"].astype(float).to_numpy()
        if len(np.unique(y_true)) < 2:
            logger.warning("Skipping curves for %s: only one class present", tool)
            continue

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_roc = roc_auc_score(y_true, y_scores)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC — {tool} (AUC={auc_roc:.3f})")
        plt.tight_layout()
        plt.savefig(output_dir / f"roc_{tool}.png", dpi=200, bbox_inches="tight")
        plt.close()

        # PR
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_scores)
        auc_pr = average_precision_score(y_true, y_scores)
        plt.figure(figsize=(6, 5))
        plt.plot(rec_arr, prec_arr, lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR — {tool} (AP={auc_pr:.3f})")
        plt.tight_layout()
        plt.savefig(output_dir / f"pr_{tool}.png", dpi=200, bbox_inches="tight")
        plt.close()


def _plot_metric_vs_factor(results_df: pd.DataFrame, metric: str, outpath: Path) -> None:
    """
    Create a 3-panel figure for `metric`: vs depth, vs read, vs deam (by tool).
    Uses deam_key for grouping but sorts x by numeric deam for consistent order.
    """
    if results_df.empty:
        logger.warning("No results to plot for %s", metric)
        return

    # Work on a copy; drop NaNs for plotting means
    df = results_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[metric])

    if df.empty:
        logger.warning("All values NaN for %s; skipping plot", metric)
        return

    tools = list(df["tool"].unique())
    plt.figure(figsize=(13, 8))

    # 1) vs depth
    ax = plt.subplot(2, 2, 1)
    for t in tools:
        td = df[df["tool"] == t]
        means = td.groupby("depth", as_index=True)[metric].mean().sort_index()
        if not means.empty:
            ax.plot(means.index.values, means.values, marker="o", label=t)
    ax.set_xlabel("Depth")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs Depth")
    ax.legend()

    # 2) vs read
    ax = plt.subplot(2, 2, 2)
    for t in tools:
        td = df[df["tool"] == t]
        means = td.groupby("read", as_index=True)[metric].mean().sort_index()
        if not means.empty:
            ax.plot(means.index.values, means.values, marker="o", label=t)
    ax.set_xlabel("Read length")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs Read length")
    ax.legend()

    # 3) vs deam
    ax = plt.subplot(2, 1, 2)
    for t in tools:
        td = df[df["tool"] == t]
        means = td.groupby("deam_key", as_index=True)[metric].mean()
        if not means.empty:
            xs = np.array([float(k) for k in means.index])
            order = np.argsort(xs)
            ax.plot(xs[order], means.values[order], marker="o", label=t)
    ax.set_xlabel("Deamination rate")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs Deamination")
    ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s plots to %s", metric, outpath)


def create_factor_plots(group_results: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    _plot_metric_vs_factor(group_results, "auc_roc", outdir / "factor_plots_auc_roc.png")
    _plot_metric_vs_factor(group_results, "auc_prc", outdir / "factor_plots_auc_pr.png")
    _plot_metric_vs_factor(group_results, "f1", outdir / "factor_plots_f1.png")


# ------------- Main -------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Compare taxonomy tools: core metrics + micro-curves + factor plots (lite).")
    p.add_argument("--root-dir", type=Path, default=Path("."), help="Directory with *_output folders")
    p.add_argument("--ground-truth", type=Path, default=Path("ground_truth.txt"), help="Path to ground truth file")
    p.add_argument("--outdir", type=Path, default=Path("evaluation_results"), help="Output directory")

    args = p.parse_args(argv)

    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(LOG_FMT))
        logger.addHandler(h)

    logger.info("Starting lite evaluation")

    tool_tables = load_count_tables(args.root_dir)
    if not tool_tables:
        logger.error("No tools with valid count tables found in %s", args.root_dir)
        return

    ground_truth = load_ground_truth(args.ground_truth)
    if not ground_truth:
        logger.error("No ground truth data found at %s", args.ground_truth)
        return

    long_df = melt_to_long(tool_tables)
    if long_df.empty:
        logger.error("No data after melting tables")
        return

    # 1) Per-tool micro metrics
    tool_results = evaluate_per_tool(long_df, ground_truth)
    if tool_results.empty:
        logger.error("No per-tool results computed")
        return
    _atomic_write_csv(args.outdir / "tool_core_metrics.csv", tool_results)

    # 2) Per-(tool,depth,read,deam) group metrics for plots
    group_results = evaluate_per_group(long_df, ground_truth)
    if group_results.empty:
        logger.warning("No per-group results computed; factor plots will be skipped")
    else:
        _atomic_write_csv(args.outdir / "group_core_metrics.csv", group_results)
        create_factor_plots(group_results, args.outdir)
        export_tool_curves(long_df, ground_truth, args.outdir / "curves")

    # Console ranking (by PR-AUC then ROC-AUC)
    ranked = tool_results.sort_values(["auc_prc", "auc_roc"], ascending=[False, False])
    logger.info("Top tools by AP then ROC-AUC:\n%s", ranked[["tool", "auc_prc", "auc_roc", "f1"]].round(3).to_string(index=False))
    logger.info("All outputs in %s", args.outdir)


if __name__ == "__main__":
    main()
