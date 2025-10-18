#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# fasta_to_seq2taxid.sh
#
# Usage:
#   ./fasta_to_seq2taxid.sh <organism_sorted.tsv> <path_to_reference_fastas> [email]
#
# Steps:
#   1) Read the "AssemblyAccession -> TaxID" mapping from <organism_sorted.tsv>.
#   2) For each *.fna/.fna.gz in <path_to_reference_fastas>, find the matching TaxID.
#   3) If no TaxID found locally, extract sequence accession and query NCBI online.
#   4) Extract sequence IDs from headers and map them to the TaxID.
#   5) Write the output to seqid2taxid.map.
#
# Requirements:
#   - Entrez Direct (for online TaxID lookup)
#   - Email address (for NCBI API, optional but recommended)
###############################################################################

usage() {
    cat << EOF
Usage: $0 <organism_sorted.tsv> <path_to_reference_fastas> [email]

Create seqid2taxid.map file for Kraken2 database building.

ARGUMENTS:
    organism_sorted.tsv         Tab-delimited file with AssemblyAccession->TaxID mapping
    path_to_reference_fastas    Directory containing FASTA files
    email                       Email address for NCBI API (optional but recommended)

OPTIONS:
    -k KEY                      NCBI API key (optional)
    -o FILE                     Output file (default: seqid2taxid.map)
    -h                          Show this help

EXAMPLES:
    $0 organism_sorted.tsv data/genomes/ user@example.com
    $0 organism_sorted.tsv data/genomes/ user@example.com -k YOUR_API_KEY
    $0 organism_sorted.tsv data/genomes/ user@example.com -o custom_seqid2taxid.map

INPUT FORMAT (organism_sorted.tsv):
    AssemblyAccession    OrgName    ftp    taxid    length
    GCF_000005825.2      E.coli     ...    511145   4641652

OUTPUT FORMAT (seqid2taxid.map):
    NC_000913.3    511145
    NC_001416.1    10710
EOF
}

# Default values
OUTPUT_FILE="seqid2taxid.map"
EMAIL=""
API_KEY=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
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
            elif [[ -z "${EMAIL:-}" ]]; then
                EMAIL="$1"
            else
                echo "Error: Too many arguments" >&2
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
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

echo "TSV file: $TSV_FILE"
echo "FASTAs dir: $FASTAS_DIR"
echo "Output file: $OUTPUT_FILE"

# Set up NCBI credentials if provided
if [[ -n "$EMAIL" ]]; then
    export NCBI_EMAIL="$EMAIL"
    echo "Using email: $EMAIL"
fi

if [[ -n "$API_KEY" ]]; then
    export NCBI_API_KEY="$API_KEY"
    echo "Using API key: ${API_KEY:0:8}..."
fi

# Check if EDirect is available for online lookup
EDIRECT_AVAILABLE=false
if command -v esummary >/dev/null 2>&1; then
    EDIRECT_AVAILABLE=true
    echo "Entrez Direct available for online TaxID lookup"
else
    echo "WARNING: Entrez Direct not found. Will skip files without local TaxID mapping."
    echo "Install with: sh -c \"\$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)\""
fi

# Initialize output file
> "$OUTPUT_FILE"

###############################################################################
# Functions
###############################################################################

# Function to read file (handles compression)
read_file() {
    local file="$1"
    case "$file" in
        *.gz)  zcat "$file" ;;
        *.bz2) bzcat "$file" ;;
        *.xz)  xzcat "$file" ;;
        *.Z)   zcat "$file" ;;
        *)     cat "$file" ;;
    esac
}

# Function to extract accession from FASTA header
extract_accession_from_fasta() {
    local fasta_file="$1"
    
    # Read first header line and extract accession
    local first_header
    first_header=$(read_file "$fasta_file" | head -1 | grep '^>')
    
    if [[ -z "$first_header" ]]; then
        return 1
    fi
    
    # Remove the ">" and extract accession pattern
    local header_content="${first_header#>}"
    local acc
    acc=$(echo "$header_content" | grep -oE "[A-Z]{1,2}_?[0-9]{6,9}\.?[0-9]*" | head -1)
    
    echo "$acc"
}

# Function to query NCBI for TaxID
query_taxid_online() {
    local accession="$1"
    
    if [[ "$EDIRECT_AVAILABLE" != "true" ]]; then
        return 1
    fi
    
    echo "  🔍 Querying NCBI for accession: $accession" >&2
    
    local taxid
    taxid=$(esummary -db nucleotide -id "$accession" 2>/dev/null | \
            xtract -pattern DocumentSummary -element TaxId 2>/dev/null | \
            head -1)
    
    if [[ -n "$taxid" && "$taxid" != "" ]]; then
        echo "$taxid"
        return 0
    else
        return 1
    fi
}

###############################################################################
# 1) Build a map: AssemblyAccession -> TaxID
###############################################################################
declare -A TAXIDMAP

# Expected columns in $TSV_FILE (tab-delimited):
#   1) AssemblyAccession  2) OrgName  3) ftp  4) taxid  5) length
# We'll parse just the first and fourth columns.
while IFS=$'\t' read -r acc org ftp txid len; do
    # Skip header lines if any or lines missing the needed fields
    [[ -z "$acc" || -z "$txid" || "$acc" == "AssemblyAccession" ]] && continue
    TAXIDMAP["$acc"]="$txid"
done < "$TSV_FILE"

echo "Loaded ${#TAXIDMAP[@]} accession->TaxID entries from local mapping."

###############################################################################
# 2) Process all FASTA files in the directory
###############################################################################
processed=0
skipped=0
online_queries=0

# Create a temporary file to store file list (avoids subshell issues)
temp_filelist=$(mktemp)
find "$FASTAS_DIR" -type f \( \
    -name "*.fasta" -o -name "*.fa" -o -name "*.fna" -o \
    -name "*.fasta.gz" -o -name "*.fa.gz" -o -name "*.fna.gz" -o \
    -name "*.fasta.bz2" -o -name "*.fa.bz2" -o -name "*.fna.bz2" -o \
    -name "*.fasta.xz" -o -name "*.fa.xz" -o -name "*.fna.xz" \
    \) > "$temp_filelist"

# Process each file
while read -r fna_file; do
    [[ -z "$fna_file" ]] && continue
    
    filename="$(basename "$fna_file")"
    
    # Validate file integrity for compressed files
    if [[ "$filename" == *.gz ]] && ! gunzip -t "$fna_file" 2>/dev/null; then
        echo "ERROR: $fna_file is corrupted. Skipping..."
        skipped=$((skipped + 1))
        continue
    fi
    
    taxid=""
    lookup_method="unknown"
    
    # Method 1: Try assembly accession extraction (original logic)
    if [[ "$filename" =~ _genomic\.fna ]]; then
        # Remove the .fna.gz suffix
        prefix="${filename%.fna.gz}"
        prefix="${prefix%.fna}"
        
        # Remove known patterns
        prefix="${prefix%%_cds_from_genomic*}" 
        prefix="${prefix%%_rna_from_genomic*}" 
        prefix="${prefix%%_protein*}"
        prefix="${prefix%%_genomic*}"
        prefix="${prefix%%_ASM*}"
        
        # Extract the core accession
        core="$(echo "$prefix" | grep -oE 'GCF_[0-9]+\.[0-9]+' || true)"
        if [[ -z "$core" ]]; then
            core="$(echo "$prefix" | grep -oE 'GCA_[0-9]+\.[0-9]+' || true)"
        fi
        
        if [[ -n "$core" ]]; then
            taxid="${TAXIDMAP[$core]:-}"
            if [[ -n "$taxid" ]]; then
                lookup_method="local_assembly"
            fi
        fi
    fi
    
    # Method 2: If no TaxID found, try sequence accession extraction + online lookup
    if [[ -z "$taxid" ]]; then
        sequence_accession=$(extract_accession_from_fasta "$fna_file")
        
        if [[ -n "$sequence_accession" ]]; then
            echo "📁 Processing $filename [Sequence accession: $sequence_accession]"
            
            # Try online lookup
            if taxid=$(query_taxid_online "$sequence_accession"); then
                lookup_method="online_sequence"
                online_queries=$((online_queries + 1))
                echo "  ✅ Found TaxID online: $taxid"
                # Add small delay to respect NCBI rate limits
                sleep 0.34
            else
                echo "  ❌ Could not find TaxID online for $sequence_accession"
            fi
        fi
    fi
    
    # Skip if no TaxID found
    if [[ -z "$taxid" ]]; then
        echo "WARNING: No TaxID found for $filename. Skipping..."
        skipped=$((skipped + 1))
        continue
    fi
    
    echo "🧬 Processing $filename [TaxID=$taxid, Method=$lookup_method]"
    
    ###########################################################################
    # 3) Extract sequence IDs and write to output
    ###########################################################################
    read_file "$fna_file" | awk -v TID="$taxid" '
        BEGIN { OFS="\t" }
        /^>/ {
            # Extract the sequence ID from the header
            header = substr($0,2)  # remove leading ">"
            seqid = header
            if (seqid ~ / /) {
                seqid = substr(seqid, 1, index(seqid, " ") - 1)  # Take up to the first space
            }
            print seqid, TID
        }
    ' >> "$OUTPUT_FILE"
    
    processed=$((processed + 1))
    
done < "$temp_filelist"

# Cleanup
rm -f "$temp_filelist"

echo ""
echo "==============================================="
echo "All conversions done!"
echo "Processed files: $processed"
echo "Skipped files: $skipped"
echo "Online queries: $online_queries"
echo "Output written to: $OUTPUT_FILE"
echo "==============================================="

# Show sample of output
if [[ -f "$OUTPUT_FILE" ]] && [[ -s "$OUTPUT_FILE" ]]; then
    echo ""
    echo "Sample output (first 5 lines):"
    head -5 "$OUTPUT_FILE"
    echo ""
    echo "Total sequence->taxid mappings: $(wc -l < "$OUTPUT_FILE")"
fi