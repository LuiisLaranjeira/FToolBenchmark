# FToolBenchmark

A benchmarking framework for evaluating metagenomic taxonomy classification tools under various sequencing conditions.

## Overview

FToolBenchmark evaluates taxonomy classification tools (Centrifuge, Kraken2, CLARK, Kaiju, FALCON) across different experimental conditions:
- **Sequencing depth** (coverage)
- **Read length** (bp)
- **Deamination rate** (DNA degradation, relevant for ancient DNA)

The framework computes performance metrics including AUC-PRC, AUC-ROC, F1 score, precision, and recall.

## Requirements

```bash
pip install pandas numpy matplotlib scikit-learn biopython
```

## Workflow

### 1. Generate Ground Truth (Optional)

Extract taxonomic IDs from reference genome files:

```bash
python taxonomy_processing.py \
  --fna_folder /path/to/reference/genomes \
  --gt_output ground_truth.txt \
  --email your.email@example.com \
  --gt_only
```

### 2. Process Tool Outputs

Convert classification tool outputs into a standardized format:

```bash
python taxonomy_processing.py \
  --input /path/to/tool/reports \
  --output_dir Tool_output \
  --email your.email@example.com \
  --api_key YOUR_NCBI_API_KEY
```

**Supported tools:**
- Centrifuge
- Kraken2
- CLARK
- Kaiju
- FALCON

The script automatically detects the tool type and generates:
- `count_table.tsv` - Counts per taxon across all conditions
- Per-sample files with taxonomic names and ranks

### 3. Evaluate and Compare Tools

Run the comprehensive evaluation:

```bash
python taxonomy_comparison.py \
  --root-dir . \
  --ground-truth ground_truth.txt \
  --outdir evaluation_results \
  --fixed-threshold 0.0 \
  --export-curves
```

**Outputs:**
- `tool_macro_metrics.csv` - Average performance across conditions (PRIMARY)
- `tool_micro_metrics.csv` - Pooled performance (SECONDARY)
- `group_core_metrics.csv` - Per-condition metrics
- `factor_plots_*.png` - Performance vs. experimental factors
- `curves/` - ROC and PR curves per tool

### 4. Create Slice Plots

Generate focused plots varying one parameter at a time:

```bash
# Vary deamination rate (fix depth=20, read=40)
python plot_slices.py \
  --input evaluation_results/group_core_metrics.csv \
  --outdir slice_plots \
  --vary deam --depth 20 --read 40

# Vary sequencing depth (fix read=40, deam=0.0)
python plot_slices.py \
  --input evaluation_results/group_core_metrics.csv \
  --outdir slice_plots \
  --vary depth --read 40 --deam 0.0

# Vary read length (fix depth=20, deam=0.0)
python plot_slices.py \
  --input evaluation_results/group_core_metrics.csv \
  --outdir slice_plots \
  --vary read --depth 20 --deam 0.0
```

## File Naming Convention

Tool outputs should follow this pattern:
```
depth{DEPTH}_read{READ}_deam{DEAM}
```

Example: `depth20_read40_deam0.0`

## Directory Structure

```
.
├── Tool1_output/
│   └── count_table.tsv
├── Tool2_output/
│   └── count_table.tsv
├── ground_truth.txt
├── evaluation_results/
│   ├── tool_macro_metrics.csv
│   ├── group_core_metrics.csv
│   └── factor_plots_*.png
└── slice_plots/
    └── slice_*.png
```

## Evaluation Metrics

- **AUC-PRC** (Primary): Area under Precision-Recall curve
- **AUC-ROC**: Area under ROC curve
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

## Tips

- Use an NCBI API key for faster processing (3 requests/sec vs 0.33/sec)
- The framework handles missing data and tool-specific output formats automatically
- Macro-averaged metrics are recommended for overall tool comparison
- Consider deamination effects when benchmarking tools for ancient DNA studies
