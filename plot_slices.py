#!/usr/bin/env python3
"""
Slice plots for taxonomy tool benchmarks:
Fix two factors and vary the third, plotting one line per tool.

Input:
  - group_core_metrics.csv (from the evaluator), columns include:
      tool, depth (int), read (int), deam (float), deam_key (str),
      au_prc, auc_roc, precision, recall, f1

Usage examples:
  # 1) Fix depth=20, read=40; vary deamination
  python3 plot_slices.py \
    --input evaluation_results/group_core_metrics.csv \
    --outdir slice_plots \
    --vary deam --depth 20 --read 40

  # 2) Fix read=40, deam=0.0; vary depth
  python3 plot_slices.py \
    --input evaluation_results/group_core_metrics.csv \
    --outdir slice_plots \
    --vary depth --read 40 --deam 0.0

  # 3) Fix depth=20, deam=0.0; vary read
  python3 plot_slices.py \
    --input evaluation_results/group_core_metrics.csv \
    --outdir slice_plots \
    --vary read --depth 20 --deam 0.0

Options:
  --metrics au_prc auc_roc f1   # choose which metrics to plot (default: au_prc auc_roc f1)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VALID_VARY = {"deam", "depth", "read"}
DEFAULT_METRICS = ["au_prc", "auc_roc", "f1"]

# One marker per tool (cycles if there are many tools)
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '>', '<', '8', 'p']
LINESTYLES = ['-', '--', '-.', ':']  # fallback variety if tools > markers


def _build_style_map(all_tools):
    """
    Deterministic mapping: each tool gets a marker (geometry) and possibly a dash style.
    Keeps colors at matplotlib defaults; only shapes/styles change.
    """
    tools = sorted(all_tools)
    style_map = {}
    for i, t in enumerate(tools):
        marker = MARKERS[i % len(MARKERS)]
        linestyle = LINESTYLES[(i // len(MARKERS)) % len(LINESTYLES)]
        style_map[t] = dict(
            marker=marker,
            linestyle=linestyle,
            markersize=6,
            markerfacecolor='none',
            markeredgewidth=1.0,
            linewidth=1.6,
        )
    return style_map


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")


def _metric_title(metric: str) -> str:
    if metric.lower() == "au_prc":
        return "AUPRC"
    if metric.lower() == "auc_roc":
        return "AUC–ROC"
    if metric.lower() == "f1":
        return "F1"
    # fallback
    return metric.upper()


def _metric_ylim(metric: str):
    # Most of these are bounded in [0,1]; keep y-axis tidy
    if metric.lower() in {"au_prc", "auc_roc", "f1", "precision", "recall"}:
        return (0.0, 1.05)  # minimal headroom; avoids misleading scale
    return None


def plot_slice(df: pd.DataFrame, vary: str, metrics: List[str], outdir: Path,
               depth: int | None, read: int | None, deam: float | None) -> None:
    assert vary in VALID_VARY

    # Filter by the two fixed factors
    filt = pd.Series(True, index=df.index)
    title_suffix = []

    if vary != "depth":
        if depth is None:
            raise ValueError("When varying != depth, you must set --depth")
        filt &= (df["depth"] == int(depth))
        title_suffix.append(f"depth={depth}")

    if vary != "read":
        if read is None:
            raise ValueError("When varying != read, you must set --read")
        filt &= (df["read"] == int(read))
        title_suffix.append(f"read={read}")

    if vary != "deam":
        if deam is None:
            raise ValueError("When varying != deam, you must set --deam")
        # allow numeric tolerance
        filt &= np.isclose(df["deam"].astype(float), float(deam))
        title_suffix.append(f"deam={deam}")

    sub = df.loc[filt].copy()
    if sub.empty:
        raise ValueError("No rows match the requested slice (check depth/read/deam values).")

    style_map = _build_style_map(df['tool'].unique())

    # Clean and sort x-axis values
    if vary == "deam":
        xlabel = "Deamination rate"
    elif vary == "depth":
        xlabel = "Depth (×)"
    else:
        xlabel = "Read length (bp)"

    # For each metric, plot 1 panel with 1 line per tool
    outdir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        # drop rows with NaN metric for clean plotting
        subm = sub.dropna(subset=[metric]).copy()
        if subm.empty:
            # no data to plot for this metric; skip quietly
            continue

        plt.figure(figsize=(6.8, 4.8))
        for tool, g in subm.groupby("tool", as_index=False):
            # build y in x order; missing x get left out
            if vary == "deam":
                g = g.sort_values("deam")
                x = g["deam"].to_numpy()
            elif vary == "depth":
                g = g.sort_values("depth")
                x = g["depth"].to_numpy()
            else:
                g = g.sort_values("read")
                x = g["read"].to_numpy()

            y = g[metric].to_numpy()
            if len(x) == 0:
                continue
            style = style_map.get(tool, {})
            plt.plot(x, y, label=tool, **style)
            # Previous Implementation with all using the same style
            #plt.plot(x, y, marker="o", label=tool)

        plt.xlabel(xlabel)
        plt.ylabel(_metric_title(metric))
        ts = ", ".join(title_suffix)
        plt.title(f"{_metric_title(metric)} vs {xlabel} ({ts})")
        ylim = _metric_ylim(metric)
        if ylim:
            plt.ylim(*ylim)
        plt.legend()
        plt.tight_layout()

        # filename
        fixed_tag = "_".join(ts.replace("=", "")  # depth20, read40, deam0.0 like
                             .replace(",", "")
                             .split())
        fvar = {"deam": "deam", "depth": "depth", "read": "read"}[vary]
        outname = f"slice_{metric}_vary_{fvar}_{fixed_tag}.png"
        plt.savefig(outdir / outname, dpi=300, bbox_inches="tight")
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Slice plots: fix two factors and vary the third.")
    ap.add_argument("--input", type=Path, required=True,
                    help="Path to evaluator output: group_core_metrics.csv")
    ap.add_argument("--outdir", type=Path, default=Path("slice_plots"),
                    help="Directory to write slice figures")
    ap.add_argument("--vary", choices=list(VALID_VARY), required=True,
                    help="Which factor to vary: deam | depth | read")
    ap.add_argument("--depth", type=int, default=None, help="Fixed depth (×)")
    ap.add_argument("--read", type=int, default=None, help="Fixed read length (bp)")
    ap.add_argument("--deam", type=float, default=None, help="Fixed deamination rate")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                    help=f"Metrics to plot (default: {' '.join(DEFAULT_METRICS)})")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    _ensure_columns(df, ["tool", "depth", "read", "deam", "au_prc", "auc_roc", "f1"])

    plot_slice(
        df=df,
        vary=args.vary,
        metrics=args.metrics,
        outdir=args.outdir,
        depth=args.depth,
        read=args.read,
        deam=args.deam,
    )


if __name__ == "__main__":
    main()
