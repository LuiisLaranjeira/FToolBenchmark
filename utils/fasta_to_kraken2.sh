#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# fasta_to_kraken2.sh
#
# Usage:
#   ./fasta_to_kraken2.sh <organism_sorted.tsv> <path_to_reference_fastas>
#
# Steps:
#   1) Read the "AssemblyAccession -> TaxID" mapping from <organism_sorted.tsv>.
#   2) For each *.fna.gz in <path_to_reference_fastas>, find the matching TaxID.
#   3) Uncompress on the fly (gunzip -c) and rewrite each header line:
#        >kraken:taxid|<TaxID> original_header
#   4) Write the converted FASTA to a new folder "kraken2_ready/".
#
# Requirements:
#   - Bash 4+ for "declare -A" associative array
#   - gunzip (or zcat)
###############################################################################

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <organism_sorted.tsv> <path_to_reference_fastas>"
  exit 1
fi

TSV_FILE="$1"
FASTAS_DIR="$2"

echo "TSV file: $TSV_FILE"
echo "FASTAs dir: $FASTAS_DIR"

# Make an output folder for the rewritten FASTAs
OUTPUT_DIR="kraken2_ready"
mkdir -p "$OUTPUT_DIR"

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

  # We'll write a new file with same name in "kraken2_ready"
  out_fna="${OUTPUT_DIR}/${filename%.gz}"  # e.g. kraken2_ready/GCF_000006625.1_ASM662v1_genomic.fna

  echo "Converting $filename => $out_fna  [TaxID=$taxid]"

  #############################################################################
  # 3) Uncompress + rewrite headers
  #############################################################################
  gunzip -c "$fna_gz" \
  | awk -v TID="$taxid" '
    BEGIN { OFS="" }
    /^>/ {
      header = substr($0,2)  # remove leading ">"
      # Kraken2 wants: >kraken:taxid|<TID> <header>
      print ">kraken:taxid|", TID, " ", header
      next
    }
    # Otherwise, print sequence lines as-is
    { print $0 }
  ' \
  > "$out_fna"
done

echo "All conversions done. See folder: $OUTPUT_DIR"
