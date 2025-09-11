import glob
import os
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def load_ground_truth(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip().isdigit())

def load_predictions(file_path, threshold=0.0):
    taxids = set()
    with open(file_path) as f:
        for line in f:
            try:
                abundance, taxid = line.strip().split()
                if float(abundance) >= threshold:
                    taxids.add(taxid)
            except ValueError:
                continue
    return taxids

def evaluate_prediction(predicted, ground_truth):
    predicted = set(predicted)
    truth = set(ground_truth)

    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_all_tools(results_dir, ground_truth_path, threshold=0.0):
    ground_truth = load_ground_truth(ground_truth_path)
    metrics = {}

    for file in os.listdir(results_dir):
        if file.endswith("ReadspTaxon.txt"):
            tool = file.replace("_ReadspTaxon.txt", "")
            predictions = load_predictions(os.path.join(results_dir, file), threshold)
            scores = evaluate_prediction(predictions, ground_truth)
            metrics[tool] = scores

    return pd.DataFrame(metrics).T

def plot_metrics(df, title="Classifier Evaluation"):
    df.plot(kind="bar", ylim=(0, 1), figsize=(10, 5))
    plt.title(title)
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def load_counts_tables(root_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load count tables from subfolders ending with '_OUTPUT' in the root directory.

    Args:
        root_dir (str): Root directory containing tool output subfolders.

    Returns:
        Dict[str, pd.DataFrame]: Mapping of tool name (derived from folder) to DataFrame.
    """
    tool_tables = {}
    for subdir in os.listdir(root_dir):
        full_path = os.path.join(root_dir, subdir)
        if os.path.isdir(full_path) and subdir.lower().endswith("_output"):
            file_path = os.path.join(full_path, "count_table.tsv")
            if os.path.exists(file_path):
                tool_name = subdir[:-7]  # remove "_OUTPUT"
                df = pd.read_csv(file_path, sep="\t")
                df["tool"] = tool_name
                tool_tables[tool_name] = df
            else:
                print(f"[!] Missing count_table.tsv in {subdir}")
    return tool_tables

def merge_tables_to_long_format(tool_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all tool-specific tables into a single long-format DataFrame.

    Args:
        tool_tables (Dict[str, pd.DataFrame]): Dictionary of count tables.

    Returns:
        pd.DataFrame: Long-format merged DataFrame with columns: taxID, tool, condition, count
    """
    merged = pd.concat(tool_tables.values())
    long_df = merged.melt(id_vars=["taxID", "tool"], var_name="condition", value_name="count")
    return long_df

def plot_top_taxa_heatmap(df: pd.DataFrame, top_n: int = 10):
    """
    Plot heatmap of top N taxa across conditions.

    Args:
        df (pd.DataFrame): Wide-format count table (taxID as index).
        top_n (int): Number of top taxa to include.
    """
    df = df.set_index("taxID")
    top_taxids = df.sum(axis=1).nlargest(top_n).index
    df_top = df.loc[top_taxids]

    plt.figure(figsize=(16, 8))
    sns.heatmap(df_top, cmap="viridis", linewidths=0.1)
    plt.title(f"Top {top_n} Taxa Across Conditions")
    plt.xlabel("Condition")
    plt.ylabel("TaxID")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def compute_metrics_from_long_df(df, ground_truth_set):
    results = []

    # Group by tool and condition
    for (tool, condition), group in df.groupby(["tool", "condition"]):
        predicted_taxids = set(group[group["count"] > 0]["taxID"].astype(str))

        tp = len(predicted_taxids & ground_truth_set)
        fp = len(predicted_taxids - ground_truth_set)
        fn = len(ground_truth_set - predicted_taxids)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        results.append({
            "tool": tool,
            "condition": condition,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    return pd.DataFrame(results)

def plot_metric(df_metrics, metric="f1"):
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df_metrics, x="condition", y=metric, hue="tool")
    plt.xticks(rotation=90)
    plt.title(f"{metric.upper()} score per tool per condition")
    plt.tight_layout()
    plt.show()

def parse_condition_fields(df):
    # Split condition into depth, read, deam
    parsed = df["condition"].str.extract(r'depth(?P<depth>\d+)_read(?P<read>\d+)_deam(?P<deam>[\d.]+)')
    for col in ["depth", "read"]:
        parsed[col] = parsed[col].astype(int)
    parsed["deam"] = parsed["deam"].astype(float)
    return pd.concat([df, parsed], axis=1)

def compute_metrics_with_auc_ap(df, ground_truth_set):
    df["label"] = df["taxID"].astype(str).isin(ground_truth_set).astype(int)

    results = []
    grouped = df.groupby(["tool", "depth", "read", "deam"])

    for (tool, depth, read, deam), group in grouped:
        try:
            labels = group["label"]
            scores = group["count"]

            auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)

            results.append({
                "tool": tool,
                "depth": depth,
                "read": read,
                "deam": deam,
                "auc_roc": auc,
                "ap_score": ap
            })
        except ValueError as e:
            # Typically happens if only one class (e.g. all 0s or 1s)
            print(f"Skipping ({tool}, d={depth}, r={read}, deam={deam}): {e}")
            continue

    return pd.DataFrame(results)

def plot_metric_by_group(df_metrics, metric="auc_roc", group_by="tool"):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_metrics, x="deam", y=metric, hue=group_by, marker="o")
    plt.title(f"{metric.upper()} vs deamination level by {group_by}")
    plt.xlabel("Deamination level")
    plt.ylabel(metric.upper())
    plt.tight_layout()
    plt.show()

def plot_metric_by_read(metric_df, metric="auc_roc"):
    g = sns.FacetGrid(metric_df, col="tool", hue="read", col_wrap=3, height=4, sharey=False)
    g.map_dataframe(sns.lineplot, x="deam", y=metric, marker="o")
    g.add_legend()
    g.set_axis_labels("Deamination Level", metric.upper())
    g.set_titles("{col_name}")
    g.fig.suptitle(f"{metric.upper()} vs Deamination by Read Length", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_metric_by_depth(metric_df, metric="auc_roc"):
    g = sns.FacetGrid(metric_df, col="tool", hue="depth", col_wrap=3, height=4, sharey=False)
    g.map_dataframe(sns.lineplot, x="deam", y=metric, marker="o")
    g.add_legend()
    g.set_axis_labels("Deamination Level", metric.upper())
    g.set_titles("{col_name}")
    g.fig.suptitle(f"{metric.upper()} vs Deamination by Depth", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_metric_deam_breakdown(metric_df, metric="auc_roc", hue="read"):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=metric_df, x="deam", y=metric, hue=hue, style="tool", markers=True)
    plt.title(f"{metric.upper()} vs Deamination (Grouped by {hue})")
    plt.xlabel("Deamination Level")
    plt.ylabel(metric.upper())
    plt.tight_layout()
    plt.show()

# def compute_metrics_from_long_df(df: pd.DataFrame, ground_truth_set: set) -> pd.DataFrame:
#     """
#     Compute precision, recall, and F1 score for each combination of tool, depth, read, deam.
#
#     Args:
#         df: long-format DataFrame with columns [taxID, tool, depth, read, deam, count]
#         ground_truth_set: Set of true taxIDs
#
#     Returns:
#         DataFrame with columns: tool, depth, read, deam, precision, recall, f1
#     """
#     results = []
#
#     grouped = df.groupby(["tool", "depth", "read", "deam"])
#     for (tool, depth, read, deam), group in grouped:
#         predicted_taxids = set(group[group["count"] > 0]["taxID"])  # non-zero predictions
#         truth = ground_truth_set
#
#         tp = len(predicted_taxids & truth)
#         fp = len(predicted_taxids - truth)
#         fn = len(truth - predicted_taxids)
#
#         precision = tp / (tp + fp) if (tp + fp) else 0.0
#         recall = tp / (tp + fn) if (tp + fn) else 0.0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
#
#         results.append({
#             "tool": tool,
#             "depth": depth,
#             "read": read,
#             "deam": deam,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1
#         })
#
#     return pd.DataFrame(results)

def compute_pr_curve_data(df: pd.DataFrame, ground_truth: set) -> pd.DataFrame:
    """
    Compute precision-recall curve data for each tool+condition.

    Returns:
        DataFrame with columns: tool, depth, read, deam, precision, recall, threshold
    """
    curve_data = []

    grouped = df.groupby(["tool", "depth", "read", "deam"])
    for (tool, depth, read, deam), group in grouped:
        y_true = group["taxID"].isin(ground_truth).astype(int)
        y_scores = group["count"].astype(float)

        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            for p, r, t in zip(precision, recall, list(thresholds) + [None]):
                curve_data.append({
                    "tool": tool,
                    "depth": depth,
                    "read": read,
                    "deam": deam,
                    "precision": p,
                    "recall": r,
                    "threshold": t
                })
        except ValueError:
            # Handle cases where y_true is all 0 or all 1
            continue

    return pd.DataFrame(curve_data)

def plot_pr_curves(curve_df: pd.DataFrame, tool: str = None, deam: float = None):
    """
    Plot PR curves, optionally filtering by tool or deamination level.
    """
    df = curve_df.copy()
    if tool:
        df = df[df["tool"] == tool]
    if deam is not None:
        df = df[df["deam"] == deam]

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="recall", y="precision", hue="depth", style="read", markers=True)
    title = "Precision-Recall Curve"
    if tool: title += f" – Tool: {tool}"
    if deam is not None: title += f" – Deam: {deam}"
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def enrich_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a DataFrame with a 'condition' column of the form
    'depth{depth}_read{read}_deam{deam}' and return a new DataFrame
    with three new columns:
      - depth (int)
      - read  (int)
      - deam  (float)

    Rows whose 'condition' doesn’t match the pattern will have NaNs.
    """
    # regex with named capture groups for depth/read/deam
    pattern = r'depth(?P<depth>\d+)_read(?P<read>\d+)_deam(?P<deam>[\d.]+)'

    # extract into a new DataFrame of strings
    parts = df['condition'].str.extract(pattern)

    # convert to correct dtypes
    parts = parts.astype({
        'depth': 'Int64',  # pandas nullable integer
        'read': 'Int64',
        'deam': 'float'
    })

    # join back to the original df
    return df.join(parts)

def plot_tool_performance_grid(
        df: pd.DataFrame,
        tools: List[str] = None,
        metrics: List[str] = ("auc_roc", "ap_score", "f1"),
        params: List[str] = ("depth", "read", "deam")
):
    """
    Plots a 3×3 grid of lineplots: rows = metrics, cols = parameters.
    Each subplot shows all tools’ performance vs that parameter.

    Args:
      df: DataFrame with columns ['tool', <params...>, <metrics...>].
      tools: if provided, a subset of tools to plot; otherwise all in df.
      metrics: which metric-columns to plot (in row order).
      params:   which parameter-columns to plot (in col order).
    """
    # optionally filter tools
    if tools is not None:
        df = df[df["tool"].isin(tools)]

    n_rows, n_cols = len(metrics), len(params)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharey=False
    )

    # Map to prettier titles
    metric_titles = {
        "auc_roc": "AUCROC",
        "ap_score": "AUPRC",
        "f1": "F1"
    }

    for i, metric in enumerate(metrics):
        for j, param in enumerate(params):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            sns.lineplot(
                data=df,
                x=param,
                y=metric,
                hue="tool",
                marker="o",
                ax=ax
            )
            # Only leftmost plots get y‑labels
            if j == 0:
                ax.set_ylabel(metric_titles.get(metric, metric).upper())
            else:
                ax.set_ylabel("")
            # Only bottom row gets x‑labels
            if i == n_rows - 1:
                ax.set_xlabel(param.capitalize())
            else:
                ax.set_xlabel("")
            # Only top row gets titles
            if i == 0:
                ax.set_title(param.capitalize())
            # Legend only on the top‑right plot to avoid crowding
            if (i, j) == (0, n_cols - 1):
                ax.legend(title="Tool")
            else:
                ax.get_legend().remove()

    fig.suptitle(
        "Tool Performance Comparison Across Parameters and Metrics",
        fontsize=16,
        weight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # results_dir = "results"  # folder containing *_ReadspTaxon.txt files
    # ground_truth_path = "ground_truth.txt"
    #
    # df = evaluate_all_tools(results_dir, ground_truth_path, threshold=0.1)
    # print(df.round(3))
    # plot_metrics(df)

    # Load and preview the merged long-format table for user inspection
    tool_tables = load_counts_tables(".")
    long_df = merge_tables_to_long_format(tool_tables)

    ground_truth = load_ground_truth("ground_truth.txt")
    metrics_df = compute_metrics_from_long_df(long_df, ground_truth_set=ground_truth)
    # plot_metric(metrics_df, "f1")  # You can also try "precision", "recall"
    # plot_metric(metrics_df, "precision")  # You can also try "precision", "recall"
    # plot_metric(metrics_df, "recall")  # You can also try "precision", "recall"

    # # 1) loaded + merged your count tables
    # long_df = merge_tables_to_long_format(tool_tables)
    #
    # # 2) enriched with depth/read/deam
    # long_df = enrich_conditions(long_df)
    #
    # # 3) computed per‑tool metrics (including AUC/AP)
    # metrics_df = compute_metrics_with_auc_ap(long_df, ground_truth)
    #
    # plot_tool_performance_grid(metrics_df)

    long_df = merge_tables_to_long_format(tool_tables)
    long_df = enrich_conditions(long_df)
    metrics_df = compute_metrics_with_auc_ap(long_df, ground_truth)
    plot_tool_performance_grid(metrics_df)

    # Plot results
    # plot_metric_by_group(metrics_df, metric="auc_roc", group_by="tool")
    # plot_metric_by_group(metrics_df, metric="ap_score", group_by="tool")
    #
    # plot_metric_by_read(metrics_df, metric="auc_roc")
    # plot_metric_by_depth(metrics_df, metric="ap_score")
    # plot_metric_deam_breakdown(metrics_df, metric="auc_roc", hue="depth")

    #long_df_parsed = compute_metrics_from_long_df(metrics_df, ground_truth_set=ground_truth)
    #
    # plot_metric(metrics_df, "f1")
    # plot_metric(metrics_df, "precision")
    # plot_metric(metrics_df, "recall")
    #
    # curve_df = compute_pr_curve_data(long_df_parsed, ground_truth)
    #
    # # Plot for one tool
    # plot_pr_curves(curve_df, tool="Kraken2")
    #
    # # Plot for specific deam level across tools
    # plot_pr_curves(curve_df, deam=0.3)

