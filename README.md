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
- 

## Utility Scripts

### Database Setup

**Download Reference Genomes**
```bash
# Download viral genomes from NCBI
./download_references_ncbi.sh viruses


# Download bacterial genomes
./download_references_ncbi.sh bacteria

# Alternative
./download_references.sh viruses
```

**Create FALCON2 Reference Database**
```bash
# Concatenate all reference sequences into a single FASTA file
./generate_reference_sequences.sh /path/to/reference_fastas input-sequences.fna

# Alternative: Use process_gz_files.sh for compressed files (It will concatenate all .gz files)
./process_gz_files.sh /path/to/reference_fastas input-sequences.fna

# Alternative: Manual concatenation from decompressed files
cat /path/to/reference_fastas/*.fna > input-sequences.fna
```

**Create Kraken2 Database**

*Option 1: Manual (recommended for complex setups)*
```bash
# Step 1: Convert FASTA headers to Kraken2 format (using assembly metadata)
./fasta_to_kraken2.sh viruses/viruses_sorted.tsv viruses/reference_fastas/

# Step 2: Download NCBI taxonomy
k2 download-taxonomy --db kraken2_db

# Step 3: Add files to library (batch process)
./kraken_add_to_library.sh kraken2_db kraken2_ready

# Alternative: Add single file (if needed)
k2 add-to-library --file kraken2_ready/GCF_000837145.fna --db kraken2_db

# Step 4: Build the database
kraken2-build --build --db kraken2_db --threads 8
```

**Note**: Do NOT use `kraken2-build --add-to-library` (has known issues). Use `k2 add-to-library` instead.

*Option 2: Simple automated approach*
```bash
# All-in-one script (downloads taxonomy and builds database)
./create_kraken2_db.sh /path/to/reference_fastas /path/to/kraken2_db
```


**Create Centrifuge Database**

```bash
# Step 1: Create seqid2taxid conversion table
./fasta_to_seq2taxid.sh viruses/viruses_sorted.tsv viruses/reference_fastas/

# Step 2: Concatenate all sequences into single file
# Option A: From compressed files
./process_gz_files.sh viruses/reference_fastas/ input-sequences.fna

# Option B: From decompressed files
cat library/*/*.fna > input-sequences.fna

# The "taxonomy/nodes.dmp" and  "taxonomy/nodes.dmp" is expected to be already downloaded for the kraken database, but can be downloaded via
centrifuge-download -o taxonomy taxonomy

# Step 3: Build Centrifuge index
centrifuge-build -p 8 \
  --conversion-table seqid2taxid.map \
  --taxonomy-tree k_viruses/taxonomy/nodes.dmp \
  --name-table k_viruses/taxonomy/names.dmp \
  input-sequences.fna c_viruses
```

**Create CLARK Database**

```bash
# Step 1: Create database directory structure
mkdir clark_viruses
mkdir clark_viruses/Custom

# Step 2: Decompress reference FASTA files (CLARK doesn't accept compressed files)
./decompress_fna_files.sh viruses/reference_fastas/ viruses/decompressed_reference_fasta

# Step 3: Copy sequences to Custom folder
cp viruses/decompressed_reference_fasta/* clark_viruses/Custom/
# OR use rsync
sudo rsync -vahP viruses/decompressed_reference_fasta/ CLARKV1.3.0.0/CLARK_db/Custom/

# Step 4: Define targets
./set_targets.sh clark_viruses custom

# Note: Build essentials required (if not already installed)
# sudo apt update && sudo apt install build-essential
```

**Create Kaiju Database**

```bash
# Step 1: Create BWT index
kaiju-mkbwt -a DNA -n 8 -o kaiju_db input-sequences.fna

# Step 2: Create FM-index
kaiju-mkfmi kaiju_db
```

### Accession to TaxID Mapping

```bash
# From a file
./accession_to_taxid_v3.sh -f accessions.txt -e your@email.com -o results.tsv

# Parse FASTA files in a directory
./accession_to_taxid_v3.sh -d /path/to/fastas -e your@email.com -o taxids.tsv

# From comma-separated list
./accession_to_taxid_v3.sh -a "NC_001925.1,NC_000913.3" -e your@email.com
```


### Data Simulation

**Simulate Ancient DNA Reads with Gargammel**

Before simulation, prepare your reference data:
- **Endogenous DNA** (`endo/`): Target organisms (e.g., concatenated viral genomes)
- **Bacterial contamination** (`bact/`): E.coli (GCF_000005845.2) or other bacteria
- **Human contamination** (`cont/`): Human mitochondrial DNA (NC_012920.1)

Download references:
- Human mitochondria: [NC_012920.1](https://www.ncbi.nlm.nih.gov/search/all/?term=NC_012920.1)
- E.coli: [GCF_000005845.2](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/)

```bash
# Run multiple parameter combinations
./run_multiple_simulations.sh

# Customizable parameters in script:
# - depths: (1 2 5 10 20 40 60)
# - read_sizes: (20 30 40 50 75 100 150)
# - deaminations: (0 0.1 0.2 0.3)
```

**Directory structure for Gargammel:**
```
data/
├── endo/           # Target DNA (viruses, etc.)
├── bact/           # Bacterial contamination
└── cont/           # Human contamination
```

### Read Processing

**Trim FASTQ Files**
```bash
./trim_fastq_files.sh input_dir output_dir
```


### Run Classification Tools

**Centrifuge**
```bash
./centrifuge_run.sh output/centrifuge
```

**Kraken2**
```bash
./kraken2_run.sh output/KRAKEN
```

**CLARK**
```bash
# Classify samples
./clark_classify.sh input/fastq output/CLARK

# Generate reports
./clark_run.sh output/CLARK
```

**Kaiju**
```bash
# Run classification
./kaiju_run.sh output/KAIJU

# Convert to table format
./kaiju_output_2_table.sh
```

**FALCON2**
```bash
./falcon2_run.sh output/falcon
```


### File Utilities

**Generate Reference Sequences**
```bash
# Concatenate all .fna files into single reference
./generate_reference_sequences.sh source_dir output_file.fna
```

**Decompress Files**
```bash
# Decompress .fna.gz files
./decompress_fna_files.sh source_dir dest_dir

# Decompress .fq.gz files
./decompress_fqgz_files.sh source_dir dest_dir

# Process and concatenate multiple .gz files
./process_gz_files.sh input_dir output_file.fna
```

**Check FASTA Headers**
```bash
# Validate headers in compressed or uncompressed files
./check_fasta_headers.sh sequences.fna.gz
```

**Compare Directories**
```bash
# Find missing files between two directories (by stem)
./compare_stems.sh dir1 dir2
```

## Complete Workflow Example

### 1. Setup and Data Preparation
```bash
# Download reference genomes
./download_references_ncbi.sh viruses
cd viruses/

# Build Kraken2 database with proper headers
../accession_to_taxid_v3.sh -d reference_fastas/ -e your@email.com -o taxids.tsv
../fasta_to_kraken2_alt.sh taxids.tsv reference_fastas/ -o kraken2_ready
../create_kraken2_db.sh kraken2_ready/ ../kraken2_db/
```

### 2. Simulate Reads
```bash
# Edit parameters in run_multiple_simulations.sh as needed
./run_multiple_simulations.sh

# Optional: Trim reads
./trim_fastq_files.sh simulated_reads/ trimmed_reads/
```

### 3. Run Classification Tool
```bash
# Kraken2
./kraken2_run.sh simulated_reads/
```

## Script Dependencies

- **Python scripts**: pandas, numpy, matplotlib, scikit-learn, biopython
- **Shell scripts**: NCBI Entrez Direct (E-utilities), wget, standard Unix tools
- **Classification tools**: Centrifuge, Kraken2, CLARK, Kaiju, FALCON2
- **Simulation tools**: Gargammel (for ancient DNA simulation)
- **Read processing**: AdapterRemoval (for quality trimming)

### Installing Entrez Direct
```bash
sh -c "$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"
```

### Installing Gargammel
```bash
# See: https://github.com/grenaud/gargammel
```

## Tips & Best Practices

- **Use NCBI API key**: Significantly faster accession->taxid mapping (3 req/sec vs 0.33 req/sec)
- **Database building**: The `fasta_to_kraken2_alt.sh` approach is recommended as it handles diverse file formats
- **Simulation parameters**: Adjust depth, read length, and deamination rates in `run_multiple_simulations.sh` for your use case
- **Memory requirements**: Kraken2 and Centrifuge databases can be large (several GB); ensure adequate RAM
- **Parallel processing**: Most classification tools support multi-threading (already set to 8 threads in scripts)
- **Ancient DNA**: Deamination simulation is critical for benchmarking tools on ancient samples

## License

See LICENSE file for details.