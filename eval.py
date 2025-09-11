import os
from typing import Dict, List, Set

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)


def load_ground_truth(path: str) -> Set[str]:
    """
    Load ground truth taxIDs from a text file (one per line).
    Only lines consisting of digits are kept.
    """
    with open(path) as f:
        return {
            line.strip()
            for line in f
            if line.strip().isdigit()
        }


def load_counts_tables(root_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Look for subdirectories ending with '_OUTPUT' in root_dir,
    read each 'count_table.tsv' into a DataFrame, and tag it with its tool name.
    """
    tool_tables = {}
    for subdir in os.listdir(root_dir):
        path = os.path.join(root_dir, subdir)
        if os.path.isdir(path) and subdir.lower().endswith("_output"):
            fp = os.path.join(path, "count_table.tsv")
            if os.path.exists(fp):
                tool = subdir[:-7]  # strip "_OUTPUT"
                df = pd.read_csv(fp, sep="\t", dtype={"taxID": str})
                df["tool"] = tool
                tool_tables[tool] = df
            else:
                print(f"[!] Missing count_table.tsv in {subdir}")
    return tool_tables


def merge_tables_to_long_format(tool_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate all tool DataFrames and melt into long format
    with columns: taxID, tool, condition, count.
    """
    df = pd.concat(tool_tables.values(), ignore_index=True)
    return df.melt(
        id_vars=["taxID", "tool"],
        var_name="condition",
        value_name="count"
    )


def enrich_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split 'condition' strings like 'depth10_read40_deam0.3' into
    three new columns: depth (int), read (int), deam (float).
    """
    pattern = r"depth(?P<depth>\d+)_read(?P<read>\d+)_deam(?P<deam>[\d.]+)"
    parts = df["condition"].str.extract(pattern).astype({
        "depth": "Int64",
        "read":  "Int64",
        "deam":  "float"
    })
    return df.join(parts)


def compute_metrics_with_auc_ap(
    df: pd.DataFrame,
    ground_truth: Set[str]
) -> pd.DataFrame:
    """
    For each combination of tool, depth, read, deam:
      - drops rows lacking count or parsed condition values
      - computes ROC AUC
      - computes average precision (AUPRC)
      - computes F1 at threshold = 0.5 on the raw counts
    Returns a DataFrame with columns:
      tool, depth, read, deam, auc_roc, ap_score, f1
    """
    # Work on a copy to avoid side‑effects
    df = df.copy()

    # 1) Drop any rows where count is NaN or the parsed features are missing
    df = df.dropna(subset=["count", "depth", "read", "deam"])

    # 2) Create binary labels
    df["label"] = df["taxID"].isin(ground_truth).astype(int)

    rows = []
    grouped = df.groupby(["tool", "depth", "read", "deam"])
    for (tool, depth, read, deam), g in grouped:
        y_true = g["label"]
        y_score = g["count"].astype(float)

        # If only one class present, skip
        if len(y_true.unique()) < 2:
            continue

        # Compute metrics
        auc = roc_auc_score(y_true, y_score)
        ap  = average_precision_score(y_true, y_score)

        # F1 at threshold=0.5
        preds = (y_score >= 0.5).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append({
            "tool":     tool,
            "depth":    int(depth),
            "read":     int(read),
            "deam":     float(deam),
            "auc_roc":  auc,
            "ap_score": ap,
            "f1":       f1
        })

    return pd.DataFrame(rows)


def plot_tool_performance_grid(
    df: pd.DataFrame,
    tools: List[str] = None,
    metrics: List[str] = ("auc_roc", "ap_score", "f1"),
    params:  List[str] = ("depth", "read", "deam")
):
    """
    Draw a grid of lineplots:
      - rows = metrics
      - cols = parameters
      - hue = tool
    """
    if tools is not None:
        df = df[df["tool"].isin(tools)]

    n_rows, n_cols = len(metrics), len(params)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False
    )

    titles = {"auc_roc":"AUCROC", "ap_score":"AUPRC", "f1":"F1"}

    for i, metric in enumerate(metrics):
        for j, param in enumerate(params):
            ax = axes[i][j]
            sns.lineplot(
                data=df,
                x=param,
                y=metric,
                hue="tool",
                marker="o",
                errorbar=None,
                ax=ax
            )
            if i == 0:
                ax.set_title(param.capitalize())
            if j == 0:
                ax.set_ylabel(titles.get(metric, metric))
            else:
                ax.set_ylabel("")
            if i == n_rows - 1:
                ax.set_xlabel(param.capitalize())
            else:
                ax.set_xlabel("")
            # only show legend in top-right
            if (i, j) == (0, n_cols - 1):
                ax.legend(title="Tool")
            else:
                ax.legend_.remove()

    fig.suptitle(
        "Tool Performance Across Depth, Read Length, and Deamination",
        fontsize=16, weight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def show_metrics_table(
    df: pd.DataFrame,
    title: str = "Tool Performance Metrics"
) -> None:
    """
    Print the metrics DataFrame in a readable table format.

    Args:
        df: DataFrame containing columns like ['tool', 'depth', 'read', 'deam',
            'auc_roc', 'ap_score', 'precision', 'recall', 'f1'].
        title: Optional title to display above the table.
    """
    if df.empty:
        print("No metrics to display; the DataFrame is empty.")
        return

    # Print the title
    print(f"\n{title}\n")

    # Use tabular print
    try:
        # If running in a notebook, display will render HTML tables
        from IPython.display import display
        display(df)
    except ImportError:
        # Fallback to console-friendly string
        print(df.to_string(index=False))

def complete_metrics_df(
    df: pd.DataFrame,
    tools:     list[str]  = None,
    depths:    list[int]  = None,
    reads:     list[int]  = None,
    deams:     list[float]= None,
    fill_zero: bool       = True
) -> pd.DataFrame:
    """
    Ensure every combination of tool × depth × read × deam appears.
    Missing metric values are filled with 0 (or left as NaN if fill_zero=False).
    """
    if tools  is None: tools  = df['tool'].unique().tolist()
    if depths is None: depths = df['depth'].unique().tolist()
    if reads  is None: reads  = df['read'].unique().tolist()
    if deams  is None: deams  = df['deam'].unique().tolist()

    idx = pd.MultiIndex.from_product(
        [tools, depths, reads, deams],
        names=['tool','depth','read','deam']
    )
    full = df.set_index(['tool','depth','read','deam']).reindex(idx).reset_index()

    metric_cols = [c for c in full.columns if c not in ('tool','depth','read','deam')]
    if fill_zero:
        full[metric_cols] = full[metric_cols].fillna(0.0)
    return full

def compute_metrics_with_threshold(
    df: pd.DataFrame,
    ground_truth: Set[str],
    positive_threshold: float = 50.0
) -> pd.DataFrame:
    """
    Compute AUC, AUPRC, and precision/recall/F1 at a given count threshold.

    Args:
      df: DataFrame in long form with columns ['taxID','tool','condition','count','depth','read','deam'].
      ground_truth: set of true taxIDs.
      positive_threshold: minimum count to call a taxID 'predicted positive'.

    Returns:
      DataFrame with columns
      ['tool','depth','read','deam','auc_roc','ap_score','precision','recall','f1'].
    """
    # 1) Clean out any rows with missing data
    df = df.dropna(subset=["count", "depth", "read", "deam"]).copy()
    df["label"] = df["taxID"].isin(ground_truth).astype(int)

    records = []
    grouped = df.groupby(["tool", "depth", "read", "deam"])
    for (tool, depth, read, deam), g in grouped:
        y_true  = g["label"]
        y_score = g["count"].astype(float)

        # Skip if only one class is present
        if len(y_true.unique()) < 2:
            continue

        # ROC AUC & Average Precision use raw scores
        auc = roc_auc_score(y_true, y_score)
        ap  = average_precision_score(y_true, y_score)

        # Apply the user‑specified count threshold
        preds = (y_score >= positive_threshold).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        records.append({
            "tool":      tool,
            "depth":     int(depth),
            "read":      int(read),
            "deam":      float(deam),
            "auc_roc":   auc,
            "ap_score":  ap,
            "precision": precision,
            "recall":    recall,
            "f1":        f1
        })

    return pd.DataFrame.from_records(records)


def compute_metrics_corrected(
        df: pd.DataFrame,
        ground_truth: Set[str],
        positive_threshold: float = 1.0,  # More reasonable for count data
        min_samples_per_group: int = 10
) -> pd.DataFrame:
    """
    Compute metrics ensuring all ground truth taxa are evaluated.

    Args:
        df: DataFrame with columns ['taxID', 'tool', 'depth', 'read', 'deam', 'count']
        ground_truth: Set of true positive taxIDs
        positive_threshold: Minimum count to call positive
        min_samples_per_group: Minimum samples needed per group for stable metrics
    """
    df = df.dropna(subset=["count", "depth", "read", "deam"]).copy()

    # Ensure all ground truth taxa are represented in each condition
    complete_combinations = []
    for (tool, depth, read, deam), group in df.groupby(['tool', 'depth', 'read', 'deam']):
        present_taxa = set(group['taxID'])
        missing_taxa = ground_truth - present_taxa

        # Add missing ground truth taxa with count=0
        for taxID in missing_taxa:
            complete_combinations.append({
                'taxID': taxID, 'tool': tool, 'depth': int(depth),
                'read': int(read), 'deam': float(deam), 'count': 0.0
            })

    if complete_combinations:
        missing_df = pd.DataFrame(complete_combinations)
        df = pd.concat([df, missing_df], ignore_index=True)

    # Create binary labels
    df["label"] = df["taxID"].isin(ground_truth).astype(int)

    # Validate ground truth coverage
    present_taxa = set(df['taxID'].unique())
    missing_gt = ground_truth - present_taxa
    if missing_gt:
        print(f"Warning: {len(missing_gt)} ground truth taxa not found in any tool output")

    records = []
    grouped = df.groupby(["tool", "depth", "read", "deam"])

    for (tool, depth, read, deam), group in grouped:
        y_true = group["label"]
        y_score = group["count"].astype(float)

        # Validate group has sufficient data and class balance
        if len(group) < min_samples_per_group:
            print(f"Warning: Skipping {tool} depth={depth} read={read} deam={deam} - only {len(group)} samples")
            continue

        if len(y_true.unique()) < 2:
            print(f"Warning: Skipping {tool} depth={depth} read={read} deam={deam} - only one class present")
            continue

        # Check if we have reasonable class balance (at least 2 of each class)
        class_counts = y_true.value_counts()
        if class_counts.min() < 2:
            print(
                f"Warning: {tool} depth={depth} read={read} deam={deam} has very imbalanced classes: {class_counts.to_dict()}")

        # Compute ranking-based metrics (use raw scores)
        try:
            auc = roc_auc_score(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
        except ValueError as e:
            print(f"Error computing AUC/AP for {tool}: {e}")
            continue

        # Apply threshold for classification metrics
        preds = (y_score >= positive_threshold).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())

        # Compute metrics with proper zero-handling
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Additional useful metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        records.append({
            "tool": tool,
            "depth": int(depth),
            "read": int(read),
            "deam": float(deam),
            "auc_roc": auc,
            "ap_score": ap,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "accuracy": accuracy,
            "n_samples": len(group),
            "n_positives": int(y_true.sum()),
            "n_negatives": int((1 - y_true).sum())
        })

    return pd.DataFrame(records)


def validate_evaluation_setup(df: pd.DataFrame, ground_truth: Set[str]) -> Dict[str, any]:
    """
    Validate the evaluation setup and return diagnostic information.
    """
    diagnostics = {}

    # Check ground truth coverage
    all_taxa = set(df['taxID'].unique())
    present_gt = ground_truth & all_taxa
    missing_gt = ground_truth - all_taxa

    diagnostics['ground_truth_stats'] = {
        'total_ground_truth': len(ground_truth),
        'present_in_data': len(present_gt),
        'missing_from_data': len(missing_gt),
        'coverage_rate': len(present_gt) / len(ground_truth) if ground_truth else 0
    }

    # Check tool coverage
    tool_coverage = {}
    for tool in df['tool'].unique():
        tool_data = df[df['tool'] == tool]
        tool_taxa = set(tool_data['taxID'].unique())
        tool_coverage[tool] = {
            'total_taxa': len(tool_taxa),
            'ground_truth_detected': len(ground_truth & tool_taxa),
            'ground_truth_missed': len(ground_truth - tool_taxa)
        }

    diagnostics['tool_coverage'] = tool_coverage

    # Check count distributions
    count_stats = df.groupby('tool')['count'].agg(['min', 'max', 'mean', 'std']).to_dict()
    diagnostics['count_distributions'] = count_stats

    return diagnostics

if __name__ == "__main__":
   # 1) Load and merge counts
    tool_tables = load_counts_tables(".")
    long_df = merge_tables_to_long_format(tool_tables)

    # 2) Extract features
    long_df = enrich_conditions(long_df)

    # 3) Load ground truth and compute metrics
    gt = load_ground_truth("ground_truth.txt")
    metrics_df = compute_metrics_corrected(long_df, gt)


    plot_tool_performance_grid(metrics_df)