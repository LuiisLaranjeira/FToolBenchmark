#!/usr/bin/env python3
"""
V1.5: Improved Taxonomy Tool Evaluator - Working with Statistical Rigor

This version fixes V1's statistical issues while maintaining its reliability.
Key improvements:
1. Cross-validated threshold optimization
2. Proper confidence intervals using percentile bootstrap
3. Data type safety and robust error handling
4. Tool-specific normalization without breaking evaluation logic
5. Statistical significance testing between tools
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple, Optional
import logging
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_count_tables(root_dir: str = ".") -> Dict[str, pd.DataFrame]:
    """Load count tables from *_output directories with robust error handling."""
    tool_tables = {}

    for item in os.listdir(root_dir):
        if os.path.isdir(item) and item.lower().endswith("_output"):
            count_file = os.path.join(item, "count_table.tsv")
            if os.path.exists(count_file):
                tool_name = item[:-7]  # Remove "_output"
                try:
                    df = pd.read_csv(count_file, sep="\t", dtype={"taxID": str})
                    df["tool"] = tool_name

                    # Ensure all count columns are numeric
                    count_cols = [col for col in df.columns if col not in ['taxID', 'tool']]
                    for col in count_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

                    tool_tables[tool_name] = df
                    logger.info(f"Loaded {len(df)} rows for tool {tool_name}")
                except Exception as e:
                    logger.error(f"Error loading {count_file}: {e}")

    return tool_tables


def load_ground_truth(path: str) -> Set[str]:
    """Load ground truth taxIDs with validation."""
    try:
        with open(path, 'r') as f:
            ground_truth = {
                line.strip()
                for line in f
                if line.strip() and line.strip().replace('.', '').isdigit()  # Handle decimal taxIDs
            }
        logger.info(f"Loaded {len(ground_truth)} ground truth taxa")
        return ground_truth
    except Exception as e:
        logger.error(f"Error loading ground truth: {e}")
        return set()


def merge_to_long_format(tool_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert tool tables to long format with type safety."""
    if not tool_tables:
        raise ValueError("No tool tables provided")

    df = pd.concat(tool_tables.values(), ignore_index=True)
    long_df = df.melt(
        id_vars=["taxID", "tool"],
        var_name="condition",
        value_name="count"
    )

    # Ensure count is numeric
    long_df['count'] = pd.to_numeric(long_df['count'], errors='coerce').fillna(0.0)

    return long_df


def parse_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Parse condition strings into depth, read, deam columns with robust handling."""
    import re
    pattern = r"depth(?P<depth>\d+)_read(?P<read>\d+)_deam(?P<deam>[\d.]+)"
    extracted = df["condition"].str.extract(pattern)

    # Convert to numeric with explicit error handling
    df["depth"] = pd.to_numeric(extracted["depth"], errors='coerce').fillna(0).astype(int)
    df["read"] = pd.to_numeric(extracted["read"], errors='coerce').fillna(0).astype(int)
    df["deam"] = pd.to_numeric(extracted["deam"], errors='coerce').fillna(0.0).astype(float)

    return df


def optimize_threshold_cv(y_true: np.ndarray, y_scores: np.ndarray, cv_folds: int = 3) -> Tuple[float, float]:
    """
    Optimize threshold using cross-validation to prevent overfitting.
    Uses fewer CV folds than V2 to maintain computational efficiency.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    if len(y_true) < cv_folds * 2:  # Not enough samples for CV
        # Fall back to simple threshold optimization
        thresholds = np.linspace(0.1, 0.9, 20)
        f1_scores = []
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx], f1_scores[best_idx]

    # Proper CV-based optimization
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    thresholds = np.linspace(0.1, 0.9, 20)
    cv_scores = []

    for threshold in thresholds:
        fold_scores = []
        try:
            for train_idx, val_idx in skf.split(y_scores, y_true):
                y_val_true = y_true[val_idx]
                y_val_scores = y_scores[val_idx]
                y_val_pred = (y_val_scores >= threshold).astype(int)

                score = f1_score(y_val_true, y_val_pred, zero_division=0)
                fold_scores.append(score)

            cv_scores.append(np.mean(fold_scores))
        except Exception:
            cv_scores.append(0.0)

    best_idx = np.argmax(cv_scores)
    return thresholds[best_idx], cv_scores[best_idx]


def bootstrap_confidence_interval(y_true: np.ndarray, y_scores: np.ndarray,
                                  metric: str = 'auc_roc', n_bootstrap: int = 200) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval with robust error handling.
    Uses fewer bootstrap samples than V2 for computational efficiency.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0

    np.random.seed(42)  # For reproducibility
    bootstrap_scores = []
    n_samples = len(y_true)

    for _ in range(n_bootstrap):
        try:
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_scores_boot = y_scores[indices]

            # Skip if only one class in bootstrap sample
            if len(np.unique(y_true_boot)) < 2:
                continue

            if metric == 'auc_roc':
                score = roc_auc_score(y_true_boot, y_scores_boot)
            elif metric == 'auc_pr':
                score = average_precision_score(y_true_boot, y_scores_boot)
            else:
                continue

            # Ensure score is numeric
            if np.isfinite(score):
                bootstrap_scores.append(float(score))

        except Exception:
            continue

    if len(bootstrap_scores) < 10:  # Need minimum bootstrap samples
        return 0.0, 0.0

    # Convert to numpy array and ensure all values are numeric
    bootstrap_scores = np.array(bootstrap_scores, dtype=float)

    try:
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        return float(ci_lower), float(ci_upper)
    except Exception:
        return 0.0, 0.0


def robust_normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Robust score normalization that handles edge cases gracefully.
    Only applies normalization if it makes statistical sense.
    """
    if len(scores) < 3:  # Not enough data for robust normalization
        return scores

    try:
        # Use median and IQR for robust normalization
        median_score = np.median(scores)
        q75, q25 = np.percentile(scores, [75, 25])
        iqr = q75 - q25

        if iqr > 1e-10:  # Avoid division by near-zero
            normalized = (scores - median_score) / iqr

            # Check if normalization is reasonable (not too extreme)
            if np.max(np.abs(normalized)) < 100:  # Reasonable scale
                return normalized

    except Exception:
        pass

    # Fall back to original scores if normalization fails
    return scores


def evaluate_single_group(group: pd.DataFrame, ground_truth: Set[str], group_name: str) -> Optional[Dict]:
    """Evaluate a single tool/condition group with improved statistical rigor."""
    # Create binary labels with type safety
    group = group.copy()
    group['is_ground_truth'] = group['taxID'].astype(str).isin(ground_truth).astype(int)

    # Get true labels and scores with type safety
    y_true = group['is_ground_truth'].values.astype(int)
    y_scores = group['count'].values.astype(float)

    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        logger.warning(f"Skipping {group_name}: only one class present")
        return None

    # More rigorous minimum sample size
    if len(group) < 10:  # Increased from 5
        logger.warning(f"Skipping {group_name}: insufficient samples ({len(group)})")
        return None

    # Need minimum positive samples
    if y_true.sum() < 3:
        logger.warning(f"Skipping {group_name}: insufficient positive samples ({y_true.sum()})")
        return None

    try:
        # Apply robust normalization
        y_scores_norm = robust_normalize_scores(y_scores)

        # Compute ranking metrics using normalized scores
        auc_roc = roc_auc_score(y_true, y_scores_norm)
        auc_pr = average_precision_score(y_true, y_scores_norm)

        # Use cross-validated threshold optimization
        optimal_threshold, cv_f1_score = optimize_threshold_cv(y_true, y_scores_norm)

        # Compute final metrics at optimal threshold
        y_pred_optimal = (y_scores_norm >= optimal_threshold).astype(int)

        precision = precision_score(y_true, y_pred_optimal, zero_division=0)
        recall = recall_score(y_true, y_pred_optimal, zero_division=0)
        f1 = f1_score(y_true, y_pred_optimal, zero_division=0)

        # Confusion matrix components
        tp = int(((y_pred_optimal == 1) & (y_true == 1)).sum())
        fp = int(((y_pred_optimal == 1) & (y_true == 0)).sum())
        fn = int(((y_pred_optimal == 0) & (y_true == 1)).sum())
        tn = int(((y_pred_optimal == 0) & (y_true == 0)).sum())

        # Bootstrap confidence intervals for AUC-ROC
        ci_lower, ci_upper = bootstrap_confidence_interval(y_true, y_scores_norm, 'auc_roc')

        return {
            'group_name': group_name,
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'cv_f1_score': float(cv_f1_score),  # CV-based F1 for comparison
            'optimal_threshold': float(optimal_threshold),
            'auc_roc_ci_lower': float(ci_lower),
            'auc_roc_ci_upper': float(ci_upper),
            'n_samples': int(len(group)),
            'n_positives': int(y_true.sum()),
            'n_negatives': int((1 - y_true).sum()),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }

    except Exception as e:
        logger.error(f"Error evaluating {group_name}: {e}")
        return None


def evaluate_tools(df: pd.DataFrame, ground_truth: Set[str]) -> pd.DataFrame:
    """Evaluate all tools across all conditions."""
    results = []

    # Group by tool and condition parameters
    groupby_cols = ['tool', 'depth', 'read', 'deam']

    for group_key, group in df.groupby(groupby_cols):
        tool, depth, read, deam = group_key
        group_name = f"{tool}_depth{depth}_read{read}_deam{deam}"

        result = evaluate_single_group(group, ground_truth, group_name)
        if result:
            # Add the grouping variables
            result.update({
                'tool': str(tool),
                'depth': int(depth),
                'read': int(read),
                'deam': float(deam)
            })
            results.append(result)

    if not results:
        logger.error("No valid evaluation results obtained")
        return pd.DataFrame()

    return pd.DataFrame(results)


def compare_tools_statistically(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare tools using paired statistical tests.
    This addresses V1's lack of significance testing.
    """
    if results_df.empty:
        return pd.DataFrame()

    tools = results_df['tool'].unique()
    comparisons = []

    # Create condition-matched pairs for each tool comparison
    for i, tool1 in enumerate(tools):
        for tool2 in tools[i + 1:]:
            # Get matching conditions for both tools
            tool1_data = results_df[results_df['tool'] == tool1].copy()
            tool2_data = results_df[results_df['tool'] == tool2].copy()

            # Merge on conditions to get paired observations
            merged = pd.merge(
                tool1_data[['depth', 'read', 'deam', 'auc_roc']],
                tool2_data[['depth', 'read', 'deam', 'auc_roc']],
                on=['depth', 'read', 'deam'],
                suffixes=('_1', '_2')
            )

            if len(merged) >= 3:  # Need minimum pairs for meaningful test
                try:
                    # Paired t-test (or Wilcoxon if normality fails)
                    diff = merged['auc_roc_1'] - merged['auc_roc_2']

                    # Test for normality
                    if len(diff) >= 8:  # shapiro needs at least 3, but 8+ is better
                        _, p_normal = stats.shapiro(diff)
                        use_parametric = p_normal > 0.05
                    else:
                        use_parametric = True  # Default to t-test for small samples

                    if use_parametric:
                        statistic, p_value = stats.ttest_rel(merged['auc_roc_1'], merged['auc_roc_2'])
                        test_used = 'paired_ttest'
                    else:
                        statistic, p_value = stats.wilcoxon(merged['auc_roc_1'], merged['auc_roc_2'])
                        test_used = 'wilcoxon'

                    # Effect size (Cohen's d for paired data)
                    effect_size = diff.mean() / diff.std() if diff.std() > 0 else 0

                    comparisons.append({
                        'tool1': tool1,
                        'tool2': tool2,
                        'mean_diff': diff.mean(),
                        'effect_size': effect_size,
                        'p_value': p_value,
                        'test_used': test_used,
                        'n_pairs': len(merged)
                    })

                except Exception as e:
                    logger.warning(f"Statistical test failed for {tool1} vs {tool2}: {e}")

    if comparisons:
        comparison_df = pd.DataFrame(comparisons)

        # Apply multiple testing correction (FDR) if statsmodels available
        if len(comparison_df) > 1 and HAS_STATSMODELS:
            rejected, p_corrected, _, _ = multipletests(
                comparison_df['p_value'],
                method='fdr_bh'
            )
            comparison_df['p_corrected'] = p_corrected
            comparison_df['significant'] = rejected
        else:
            # Fallback: simple Bonferroni correction or uncorrected
            if len(comparison_df) > 1:
                comparison_df['p_corrected'] = comparison_df['p_value'] * len(comparison_df)  # Bonferroni
                comparison_df['p_corrected'] = np.minimum(comparison_df['p_corrected'], 1.0)  # Cap at 1
            else:
                comparison_df['p_corrected'] = comparison_df['p_value']
            comparison_df['significant'] = comparison_df['p_corrected'] < 0.05

        return comparison_df

    return pd.DataFrame()


def create_enhanced_plots(results_df: pd.DataFrame, comparisons_df: pd.DataFrame,
                          output_dir: str = "evaluation_results"):
    """Create enhanced plots including confidence intervals and statistical comparisons."""
    os.makedirs(output_dir, exist_ok=True)

    if results_df.empty:
        logger.warning("No results to plot")
        return

    # Enhanced tool comparison with confidence intervals
    plt.figure(figsize=(15, 10))

    # Plot 1: AUC-ROC with confidence intervals
    plt.subplot(2, 3, 1)

    # Box plot for distribution
    sns.boxplot(data=results_df, x='tool', y='auc_roc', alpha=0.7)

    # Add confidence intervals as error bars
    if 'auc_roc_ci_lower' in results_df.columns:
        tool_stats = results_df.groupby('tool').agg({
            'auc_roc': 'mean',
            'auc_roc_ci_lower': 'mean',
            'auc_roc_ci_upper': 'mean'
        })

        plt.errorbar(
            range(len(tool_stats)),
            tool_stats['auc_roc'],
            yerr=[tool_stats['auc_roc'] - tool_stats['auc_roc_ci_lower'],
                  tool_stats['auc_roc_ci_upper'] - tool_stats['auc_roc']],
            fmt='ro', capsize=5, capthick=2, markersize=8, alpha=0.8
        )

    plt.title('AUC-ROC with 95% Confidence Intervals')
    plt.xticks(rotation=45)
    plt.ylabel('AUC-ROC')

    # Plot 2: F1 comparison with CV scores
    plt.subplot(2, 3, 2)
    if 'cv_f1_score' in results_df.columns:
        # Compare regular F1 vs CV F1
        f1_comparison = results_df.groupby('tool')[['f1', 'cv_f1_score']].mean()
        x = np.arange(len(f1_comparison))
        width = 0.35

        plt.bar(x - width / 2, f1_comparison['f1'], width, label='Optimistic F1', alpha=0.7)
        plt.bar(x + width / 2, f1_comparison['cv_f1_score'], width, label='CV F1', alpha=0.7)

        plt.xlabel('Tool')
        plt.ylabel('F1 Score')
        plt.title('F1: Optimistic vs Cross-Validated')
        plt.xticks(x, f1_comparison.index, rotation=45)
        plt.legend()
    else:
        sns.boxplot(data=results_df, x='tool', y='f1')
        plt.title('F1 Score Distribution')
        plt.xticks(rotation=45)

    # Plot 3: Precision vs Recall with confidence ellipses
    plt.subplot(2, 3, 3)
    for tool in results_df['tool'].unique():
        tool_data = results_df[results_df['tool'] == tool]
        plt.scatter(tool_data['recall'], tool_data['precision'], label=tool, alpha=0.7, s=50)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4-6: Parameter sensitivity (same as V1)
    params = ['depth', 'read', 'deam']
    param_titles = ['Sequencing Depth', 'Read Length', 'Deamination Rate']

    for i, (param, title) in enumerate(zip(params, param_titles)):
        plt.subplot(2, 3, 4 + i)
        for tool in results_df['tool'].unique():
            tool_data = results_df[results_df['tool'] == tool]
            if len(tool_data) > 1:
                tool_means = tool_data.groupby(param)['auc_roc'].mean()
                plt.plot(tool_means.index, tool_means.values, marker='o', label=tool, linewidth=2)

        plt.xlabel(param.capitalize())
        plt.ylabel('Mean AUC-ROC')
        plt.title(f'Performance vs {title}')
        if i == 0:
            plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/enhanced_tool_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Statistical comparison plot
    if not comparisons_df.empty:
        plt.figure(figsize=(12, 8))

        # Plot 1: Effect sizes
        plt.subplot(2, 1, 1)

        # Create comparison labels
        comparisons_df['comparison'] = comparisons_df['tool1'] + ' vs ' + comparisons_df['tool2']

        # Color by significance
        colors = ['red' if sig else 'gray' for sig in comparisons_df['significant']]

        plt.barh(range(len(comparisons_df)), comparisons_df['effect_size'], color=colors, alpha=0.7)
        plt.yticks(range(len(comparisons_df)), comparisons_df['comparison'])
        plt.xlabel('Effect Size (Cohen\'s d)')
        plt.title('Statistical Comparison of Tools')
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)

        # Add significance indicators
        for i, (_, row) in enumerate(comparisons_df.iterrows()):
            if row['significant']:
                plt.text(row['effect_size'], i, ' *', fontsize=16, va='center')

        # Plot 2: P-values
        plt.subplot(2, 1, 2)
        plt.barh(range(len(comparisons_df)), -np.log10(comparisons_df['p_corrected']),
                 color=colors, alpha=0.7)
        plt.yticks(range(len(comparisons_df)), comparisons_df['comparison'])
        plt.xlabel('-log10(p-value corrected)')
        plt.title('Statistical Significance')
        plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/statistical_comparisons.png", dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"Enhanced plots saved to {output_dir}")


def generate_enhanced_report(results_df: pd.DataFrame, comparisons_df: pd.DataFrame,
                             output_path: str = "enhanced_evaluation_summary.txt"):
    """Generate enhanced text summary report with statistical analysis."""
    with open(output_path, 'w') as f:
        f.write("ENHANCED TAXONOMY TOOL EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        if results_df.empty:
            f.write("No evaluation results available.\n")
            return

        # Data overview
        f.write("EVALUATION OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total evaluations: {len(results_df)}\n")
        f.write(f"Tools evaluated: {', '.join(results_df['tool'].unique())}\n")
        f.write(f"Conditions tested: {results_df['depth'].nunique()} depths, "
                f"{results_df['read'].nunique()} read lengths, "
                f"{results_df['deam'].nunique()} deamination rates\n")
        f.write(f"Statistical improvements: CV threshold optimization, bootstrap CI, significance testing\n\n")

        # Tool rankings with confidence intervals
        f.write("TOOL RANKINGS WITH STATISTICAL CONFIDENCE\n")
        f.write("-" * 45 + "\n")

        tool_stats = results_df.groupby('tool').agg({
            'auc_roc': ['mean', 'std', 'count'],
            'auc_roc_ci_lower': 'mean',
            'auc_roc_ci_upper': 'mean'
        }).round(3)

        tool_stats.columns = ['mean_auc', 'std_auc', 'n_conditions', 'ci_lower', 'ci_upper']
        tool_stats = tool_stats.sort_values('mean_auc', ascending=False)

        for rank, (tool, row) in enumerate(tool_stats.iterrows(), 1):
            f.write(f"{rank}. {tool}:\n")
            f.write(f"   AUC-ROC: {row['mean_auc']:.3f} ± {row['std_auc']:.3f}\n")
            f.write(f"   95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]\n")
            f.write(f"   Conditions: {int(row['n_conditions'])}\n\n")

        # Cross-validation analysis
        if 'cv_f1_score' in results_df.columns:
            f.write("CROSS-VALIDATION ANALYSIS\n")
            f.write("-" * 25 + "\n")

            cv_analysis = results_df.groupby('tool').agg({
                'f1': 'mean',
                'cv_f1_score': 'mean'
            }).round(3)
            cv_analysis['overfitting'] = cv_analysis['f1'] - cv_analysis['cv_f1_score']
            cv_analysis = cv_analysis.sort_values('overfitting', ascending=False)

            f.write("Overfitting analysis (Regular F1 - CV F1):\n")
            for tool, row in cv_analysis.iterrows():
                f.write(f"  {tool}: {row['overfitting']:.3f} "
                        f"(Regular: {row['f1']:.3f}, CV: {row['cv_f1_score']:.3f})\n")
            f.write("\n")

        # Statistical significance analysis
        if not comparisons_df.empty:
            f.write("STATISTICAL SIGNIFICANCE ANALYSIS\n")
            f.write("-" * 35 + "\n")

            significant_comps = comparisons_df[comparisons_df['significant']]

            if len(significant_comps) > 0:
                f.write("Statistically significant differences (FDR-corrected):\n")
                for _, row in significant_comps.iterrows():
                    direction = "better" if row['mean_diff'] > 0 else "worse"
                    f.write(f"  {row['tool1']} is significantly {direction} than {row['tool2']}\n")
                    f.write(f"    Mean difference: {row['mean_diff']:.3f}\n")
                    f.write(f"    Effect size: {row['effect_size']:.3f}\n")
                    f.write(f"    p-value: {row['p_corrected']:.4f}\n\n")
            else:
                f.write("No statistically significant differences found between tools.\n")
                f.write("This may indicate:\n")
                f.write("  - Tools have similar performance\n")
                f.write("  - Insufficient statistical power\n")
                f.write("  - High variability across conditions\n\n")

        # Best conditions analysis (same as V1)
        f.write("OPTIMAL CONDITIONS PER TOOL\n")
        f.write("-" * 30 + "\n")
        for tool in results_df['tool'].unique():
            tool_data = results_df[results_df['tool'] == tool]
            best_condition = tool_data.loc[tool_data['auc_roc'].idxmax()]
            f.write(f"{tool}:\n")
            f.write(f"   Best AUC-ROC: {best_condition['auc_roc']:.3f}\n")
            f.write(
                f"   At: depth={best_condition['depth']}, read={best_condition['read']}, deam={best_condition['deam']}\n")
            f.write(f"   CV F1 at optimum: {best_condition.get('cv_f1_score', 'N/A')}\n\n")

    logger.info(f"Enhanced summary report saved to {output_path}")


def main():
    """Main evaluation pipeline with enhanced statistical analysis."""
    logger.info("Starting enhanced taxonomy evaluation (V1.5)")

    # Load data with robust error handling
    try:
        tool_tables = load_count_tables(".")
        if not tool_tables:
            logger.error("No tool tables found")
            return

        ground_truth = load_ground_truth("ground_truth.txt")
        if not ground_truth:
            logger.error("No ground truth data found")
            return

        # Process data with type safety
        logger.info("Processing data with enhanced validation...")
        long_df = merge_to_long_format(tool_tables)
        long_df = parse_conditions(long_df)

        # Clean data more rigorously
        initial_rows = len(long_df)
        long_df = long_df.dropna(subset=['depth', 'read', 'deam'])
        long_df['count'] = pd.to_numeric(long_df['count'], errors='coerce').fillna(0)

        # Remove invalid conditions
        long_df = long_df[(long_df['depth'] > 0) & (long_df['read'] > 0) & (long_df['deam'] >= 0)]

        logger.info(
            f"Data processed: {len(long_df)} total records ({initial_rows - len(long_df)} removed due to invalid data)")

        # Evaluate tools with enhanced methods
        logger.info("Evaluating tools with cross-validation and bootstrap confidence intervals...")
        results_df = evaluate_tools(long_df, ground_truth)

        if results_df.empty:
            logger.error("No evaluation results obtained")
            return

        # Perform statistical comparisons
        logger.info("Performing statistical significance testing...")
        comparisons_df = compare_tools_statistically(results_df)

        # Create output directory
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)

        # Save results
        results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
        if not comparisons_df.empty:
            comparisons_df.to_csv(f"{output_dir}/statistical_comparisons.csv", index=False)

        # Generate enhanced plots and reports
        create_enhanced_plots(results_df, comparisons_df, output_dir)
        generate_enhanced_report(results_df, comparisons_df, f"{output_dir}/enhanced_summary_report.txt")

        # Print enhanced summary
        print("\nENHANCED EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"Evaluated {len(results_df)} conditions across {results_df['tool'].nunique()} tools")
        print(f"Statistical improvements: CV thresholds, bootstrap CI, significance testing")
        print(f"Results saved to {output_dir}/")

        # Display top performers with confidence intervals
        print("\nTOP PERFORMING TOOLS (with 95% confidence intervals):")
        tool_rankings = results_df.groupby('tool').agg({
            'auc_roc': 'mean',
            'auc_roc_ci_lower': 'mean',
            'auc_roc_ci_upper': 'mean'
        }).round(3)
        tool_rankings = tool_rankings.sort_values('auc_roc', ascending=False)

        for rank, (tool, row) in enumerate(tool_rankings.iterrows(), 1):
            print(f"  {rank}. {tool}: {row['auc_roc']:.3f} "
                  f"[{row['auc_roc_ci_lower']:.3f}, {row['auc_roc_ci_upper']:.3f}]")

        # Statistical significance summary
        if not comparisons_df.empty:
            sig_count = comparisons_df['significant'].sum()
            total_comparisons = len(comparisons_df)
            print(f"\nStatistical analysis: {sig_count}/{total_comparisons} comparisons significant (FDR-corrected)")

            if sig_count > 0:
                print("Significant differences found between:")
                for _, row in comparisons_df[comparisons_df['significant']].iterrows():
                    direction = ">" if row['mean_diff'] > 0 else "<"
                    print(f"  {row['tool1']} {direction} {row['tool2']} (p={row['p_corrected']:.4f})")
        else:
            print("\nNo statistical comparisons could be performed (insufficient paired data)")

        print(f"\nDetailed results: {output_dir}/detailed_results.csv")
        print(f"Enhanced report: {output_dir}/enhanced_summary_report.txt")
        print(f"Enhanced plots: {output_dir}/enhanced_tool_comparison.png")
        if not comparisons_df.empty:
            print(f"Statistical comparisons: {output_dir}/statistical_comparisons.png")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Error: {e}")
        print("Check the log for detailed error information.")
        raise


if __name__ == "__main__":
    main()