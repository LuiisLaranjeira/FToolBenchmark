#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# fasta_to_seq2taxid.sh
#
# Usage:
#   ./fasta_to_seq2taxid.sh <organism_sorted.tsv> <path_to_reference_fastas>
#
# Steps:
#   1) Read the "AssemblyAccession -> TaxID" mapping from <organism_sorted.tsv>.
#   2) For each *.fna.gz in <path_to_reference_fastas>, find the matching TaxID.
#   3) Extract sequence IDs from headers and map them to the TaxID.
#   4) Write the output to seqid2taxid.map.
#
###############################################################################

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <organism_sorted.tsv> <path_to_reference_fastas>"
  exit 1
fi

TSV_FILE="$1"
FASTAS_DIR="$2"

echo "TSV file: $TSV_FILE"
echo "FASTAs dir: $FASTAS_DIR"

# Define output file for seqid2taxid.map
OUTPUT_FILE="seqid2taxid.map"
> "$OUTPUT_FILE"  # Empty the file if it exists

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

echo "Loaded ${#TAXIDMAP[@]} accession->TaxID entries."

###############################################################################
# 2) Loop over each *.fna.gz in FASTAs_DIR and find the matching TaxID
###############################################################################
shopt -s nullglob
for fna_gz in "$FASTAS_DIR"/*_genomic.fna.gz; do
  # Example: GCF_000006625.1_ASM662v1_genomic.fna.gz
  filename="$(basename "$fna_gz")"

  # Validate the .gz file's integrity
  if ! gunzip -t "$fna_gz" 2>/dev/null; then
    echo "ERROR: $fna_gz is corrupted. Skipping..."
    continue
  fi

  # Remove the .fna.gz suffix
  prefix="${filename%.fna.gz}"
  
  # Remove known patterns in steps:
  #   _cds_from_genomic, _rna_from_genomic, _genomic
  #   also remove any trailing _ASMxxxx
  prefix="${prefix%%_cds_from_genomic*}" 
  prefix="${prefix%%_rna_from_genomic*}" 
  prefix="${prefix%%_protein*}"           # If you have protein files
  prefix="${prefix%%_genomic*}"
  prefix="${prefix%%_ASM*}"

  # Extract the core accession with grep -oE:
  # First try GCF, then try GCA if GCF not found
  core="$(echo "$prefix" | grep -oE 'GCF_[0-9]+\.[0-9]+' || true)"
  if [[ -z "$core" ]]; then
    core="$(echo "$prefix" | grep -oE 'GCA_[0-9]+\.[0-9]+' || true)"
  fi

  if [[ -z "$core" ]]; then
    echo "WARNING: Could not extract GCF/GCA from $filename => skipping."
    continue
  fi

  # Now core might be e.g. "GCF_000006625.1" 
  subacc="$core"

  # Try direct map
  taxid="${TAXIDMAP[$subacc]:-}"

  if [[ -z "$taxid" ]]; then
    echo "WARNING: No TaxID found for $fna_gz (prefix: $subacc). Skipping..."
    continue
  fi

  echo "Processing $filename  [TaxID=$taxid]"

  #############################################################################
  # 3) Uncompress and extract sequence IDs
  #############################################################################
  gunzip -c "$fna_gz" \
  | awk -v TID="$taxid" '
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
  ' \
  >> "$OUTPUT_FILE"
done

echo "All conversions done. Output written to: $OUTPUT_FILE"
