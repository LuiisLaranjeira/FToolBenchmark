#!/bin/bash

# accession_to_taxid.sh
# Map accession numbers to taxonomic IDs using Entrez Direct

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Map GenBank/RefSeq accession numbers to taxonomic IDs.

OPTIONS:
    -f FILE     File containing accession numbers (one per line)
    -p FILE     Parse file to extract accession numbers
    -a LIST     Comma-separated list of accessions
    -e EMAIL    Email address (sets NCBI_EMAIL env var)
    -k KEY      API key (sets NCBI_API_KEY env var)
    -h          Show this help

EXAMPLES:
    $0 -f accessions.txt -e user@example.com
    $0 -p sequences.fasta.gz -e user@example.com
    $0 -p analysis_report.bz2 -e user@example.com
    $0 -a "NC_001925.1,NC_000913.3" -e user@example.com

OUTPUT: accession<TAB>taxid
EOF
}

# Parse arguments
while getopts "f:p:a:e:k:h" opt; do
    case $opt in
        f) ACCESSION_FILE="$OPTARG" ;;
        p) PARSE_FILE="$OPTARG" ;;
        a) ACCESSION_LIST="$OPTARG" ;;
        e) export NCBI_EMAIL="$OPTARG" ;;
        k) export NCBI_API_KEY="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done

# Check if EDirect is installed
if ! command -v esummary >/dev/null 2>&1; then
    echo "Error: Entrez Direct not found. Install with:" >&2
    echo "  sh -c \"\$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)\"" >&2
    exit 1
fi

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

# Get accessions
if [[ -n "$ACCESSION_FILE" ]]; then
    # Read accessions from file (one per line)
    accessions=$(read_file "$ACCESSION_FILE" | grep -v '^#' | grep -v '^[[:space:]]*$' | tr '\n' ',' | sed 's/,$//')
elif [[ -n "$PARSE_FILE" ]]; then
    # Parse file to extract accession numbers
    regex="[A-Z]{1,2}_?[0-9]{6,9}\.?[0-9]*"
    accessions=$(read_file "$PARSE_FILE" | grep -oE "$regex" | sort -u | tr '\n' ',' | sed 's/,$//')
elif [[ -n "$ACCESSION_LIST" ]]; then
    accessions="$ACCESSION_LIST"
else
    echo "Error: Provide accessions with -f, -p, or -a" >&2
    usage
    exit 1
fi

# Check if we found any accessions
if [[ -z "$accessions" ]]; then
    echo "Error: No accessions found" >&2
    exit 1
fi

# Query NCBI and extract accession and taxid
esummary -db nucleotide -id "$accessions" | \
xtract -pattern DocumentSummary -element AccessionVersion,TaxId