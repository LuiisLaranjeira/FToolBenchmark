#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# fasta_to_kraken2.sh
#
# Usage:
#   ./fasta_to_kraken2.sh <accession_taxid.tsv> <path_to_reference_fastas>
#
# Steps:
#   1) Read the "Accession -> TaxID" mapping from <accession_taxid.tsv> 
#      (output from accession_to_taxid script: accession<TAB>taxid)
#   2) For each *.fna/.fna.gz/.fa/.fa.gz in <path_to_reference_fastas>, 
#      extract accession from filename and find matching TaxID.
#   3) Uncompress on the fly (if needed) and rewrite each header line:
#        >kraken:taxid|<TaxID> original_header
#   4) Write the converted FASTA to "kraken2_ready/" folder.
#
# Requirements:
#   - Bash 4+ for associative arrays
#   - gunzip/zcat for compressed files
###############################################################################

usage() {
    cat << EOF
Usage: $0 <accession_taxid.tsv> <path_to_reference_fastas> [options]

Convert FASTA files to Kraken2 format using accession->taxid mapping.

ARGUMENTS:
    accession_taxid.tsv     Tab-separated file: accession<TAB>taxid
    path_to_reference_fastas Directory containing FASTA files

OPTIONS:
    -o DIR                  Output directory (default: kraken2_ready)
    -h                      Show this help

EXAMPLES:
    # Basic usage
    $0 taxids.tsv data/bact/
    
    # Custom output directory  
    $0 taxids.tsv data/viruses/ -o kraken2_viruses/
    
INPUT FORMAT (accession_taxid.tsv):
    NC_000883.2     10798
    NC_000898.1     32604
    NC_001611.1     10255

FASTA FILE EXAMPLE:
    File: GCF_000859885.1_ViralProj15197_genomic.fna
    Header: >NC_001546.1 Arabis mosaic virus small satellite RNA, complete genome
    Accession extracted: NC_001546.1

SUPPORTED FASTA FORMATS:
    - *.fna, *.fa, *.fasta (uncompressed)
    - *.fna.gz, *.fa.gz, *.fasta.gz (gzip compressed)  
    - *.fna.bz2, *.fa.bz2, *.fasta.bz2 (bzip2 compressed)

NOTE: 
    The script extracts the accession from the first FASTA header in each file
    and applies the corresponding TaxID to ALL sequences in that file.
EOF
}

# Default values
OUTPUT_DIR="kraken2_ready"

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        -o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1" >&2
            usage
            exit 1
            ;;
        *)
            # Positional arguments
            if [[ -z "${TSV_FILE:-}" ]]; then
                TSV_FILE="$1"
            elif [[ -z "${FASTAS_DIR:-}" ]]; then
                FASTAS_DIR="$1"
            else
                echo "Error: Too many arguments" >&2
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [[ -z "${TSV_FILE:-}" || -z "${FASTAS_DIR:-}" ]]; then
    echo "Error: Missing required arguments" >&2
    usage
    exit 1
fi

if [[ ! -f "$TSV_FILE" ]]; then
    echo "Error: File '$TSV_FILE' not found" >&2
    exit 1
fi

if [[ ! -d "$FASTAS_DIR" ]]; then
    echo "Error: Directory '$FASTAS_DIR' not found" >&2
    exit 1
fi

echo "Accession->TaxID file: $TSV_FILE"
echo "FASTAs directory: $FASTAS_DIR"
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

###############################################################################
# 1) Build a map: Accession -> TaxID
###############################################################################
declare -A TAXIDMAP

# Read accession<TAB>taxid format (output from accession_to_taxid script)
while IFS=$'\t' read -r acc txid; do
    # Skip empty lines or malformed lines
    [[ -z "$acc" || -z "$txid" ]] && continue
    
    # Skip lines that might be headers or comments
    [[ "$acc" =~ ^[A-Z]{1,2}_?[0-9] ]] || continue
    
    TAXIDMAP["$acc"]="$txid"
done < "$TSV_FILE"

echo "Loaded ${#TAXIDMAP[@]} accession->TaxID mappings."

if [[ ${#TAXIDMAP[@]} -eq 0 ]]; then
    echo "Error: No valid accession->taxid mappings found in $TSV_FILE" >&2
    echo "Expected format: accession<TAB>taxid" >&2
    exit 1
fi

###############################################################################
# 2) Function to read files (handles compression)
###############################################################################
read_fasta() {
    local file="$1"
    case "$file" in
        *.gz)  zcat "$file" ;;
        *.bz2) bzcat "$file" ;;
        *.xz)  xzcat "$file" ;;
        *.Z)   zcat "$file" ;;
        *)     cat "$file" ;;
    esac
}

###############################################################################
# 3) Function to extract accession from FASTA header
###############################################################################
extract_accession_from_fasta() {
    local fasta_file="$1"
    
    # Read first header line and extract accession
    local first_header
    first_header=$(read_fasta "$fasta_file" | head -1 | grep '^>')
    
    if [[ -z "$first_header" ]]; then
        return 1
    fi
    
    # Remove the ">" and extract accession pattern
    local header_content="${first_header#>}"
    local acc
    acc=$(echo "$header_content" | grep -oE "[A-Z]{1,2}_?[0-9]{6,9}\.?[0-9]*" | head -1)
    
    echo "$acc"
}

###############################################################################
# 4) Process all FASTA files in the directory
###############################################################################
processed=0
skipped=0

# Create a temporary file to store file list (avoids subshell issues)
temp_filelist=$(mktemp)
find "$FASTAS_DIR" -type f \( \
    -name "*.fasta" -o -name "*.fa" -o -name "*.fna" -o \
    -name "*.fasta.gz" -o -name "*.fa.gz" -o -name "*.fna.gz" -o \
    -name "*.fasta.bz2" -o -name "*.fa.bz2" -o -name "*.fna.bz2" -o \
    -name "*.fasta.xz" -o -name "*.fa.xz" -o -name "*.fna.xz" \
    \) > "$temp_filelist"

# Process each file
while read -r fasta_file; do
    [[ -z "$fasta_file" ]] && continue
    
    filename=$(basename "$fasta_file")
    
    # Extract accession from FASTA header (not filename)
    accession=$(extract_accession_from_fasta "$fasta_file")
    
    if [[ -z "$accession" ]]; then
        echo "WARNING: Could not extract accession from FASTA headers in $filename => skipping."
        skipped=$((skipped + 1))
        continue
    fi
    
    # Look up taxid
    taxid="${TAXIDMAP[$accession]:-}"
    
    if [[ -z "$taxid" ]]; then
        echo "WARNING: No TaxID found for accession $accession (file: $filename) => skipping."
        skipped=$((skipped + 1))
        continue
    fi
    
    # Determine output filename (remove compression extensions)
    output_name="$filename"
    output_name="${output_name%.gz}"
    output_name="${output_name%.bz2}"
    output_name="${output_name%.xz}"
    output_name="${output_name%.Z}"
    
    out_fasta="${OUTPUT_DIR}/${output_name}"
    
    echo "Processing $filename => $output_name [Found accession: $accession, TaxID: $taxid]"
    
    # Convert FASTA headers to Kraken2 format
    if read_fasta "$fasta_file" | awk -v TID="$taxid" '
        BEGIN { OFS="" }
        /^>/ {
            header = substr($0,2)  # remove leading ">"
            # Kraken2 format: >kraken:taxid|<TID> <header>
            print ">kraken:taxid|", TID, " ", header
            next
        }
        # Print sequence lines as-is
        { print $0 }
    ' > "$out_fasta"; then
        processed=$((processed + 1))
        echo "  ✅ Successfully created: $out_fasta"
    else
        echo "  ❌ Failed to create: $out_fasta"
        skipped=$((skipped + 1))
    fi
    
done < "$temp_filelist"

# Cleanup
rm -f "$temp_filelist"

echo ""
echo "Conversion complete!"
echo "Processed files: $processed"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Files are now ready for Kraken2 database building."