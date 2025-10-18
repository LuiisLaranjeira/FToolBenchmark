#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: ./download_references.sh [organism]
# Example: ./download_references.sh bacteria
# If no argument is given, defaults to 'viruses'.

## (Plan Step 1) ESearch
organism="${1:-viruses}"
##query="${organism}[ORGN] AND srcdb_refseq[prop]"
query="${organism}[Organism] AND 'complete genome'[filter] AND latest_refseq[filter] AND (latest[filter] NOT anomalous[filter])"
echo "Querying assembly DB for: $query"

mkdir -p "$organism"
cd "$organism"

esearch -db assembly -query "$query" > "${organism}_search.xml"

## (Plan Step 2) ESummary + xtract
cat "${organism}_search.xml" \
  | esummary -db assembly \
  > "${organism}_summary.xml"


cat "${organism}_summary.xml" \
| xtract \
    -pattern DocumentSummary \
    -element AssemblyAccession Organism FtpPath_RefSeq Taxid \
    -block 'Meta/Stats/Stat' \
    -if '@category' -equals 'total_length' \
    -element '.' \
| sed -E 's/\tStat +"?([0-9]+)"?/\t\1/g' \
| awk -F'\t' 'NF==5' \
> "${organism}.tsv"

## Sort by total_length ascending (column 5)
## Then sort uniquely by taxid (column 4), the `-u` picks the FIRST occurrence of each TaxID
sort -t $'\t' -k5,5n --stable "${organism}.tsv" \
| sort -t $'\t' -k4,4 -u --stable \
| sort -t $'\t' -k5,5n --stable \
> "${organism}_sorted.tsv"

## (Plan Step 3) Download FASTA
output_dir="reference_fastas"
mkdir -p "$output_dir"

echo "Starting downloads..."
while IFS=$'\t' read -r acc org ftp taxid size; do
  if [[ -n "$ftp" ]]; then
    # Each assembly directory typically has a file ending in "_genomic.fna.gz", exclude RNA and CDS
    wget -q -P "$output_dir" "${ftp}/*_genomic.fna.gz" \
    --reject '*_rna_from_*' \
    --reject '*_cds_from_*'
  fi
done < "${organism}_sorted.tsv"

echo "All downloads complete."
echo "Files saved in: ${organism}/${output_dir}"
