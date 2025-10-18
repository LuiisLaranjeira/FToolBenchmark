#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: ./create_kraken2_db.sh /path/to/reference_fastas /path/to/kraken2_db

# 1) Parse arguments
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <path_to_fastas> <kraken2_db_folder>"
  exit 1
fi

FASTAS_DIR="$1"
KRAKEN2_DB="$2"

echo "FASTAs directory: $FASTAS_DIR"
echo "Kraken2 DB folder: $KRAKEN2_DB"

# 2) Make sure the DB folder exists (or create it)
mkdir -p "$KRAKEN2_DB"

# 3) Download or update taxonomy if needed
#    (You may skip if you already have taxonomy)
#echo "Downloading/Updating taxonomy..."
#kraken2-build --download-taxonomy --db "$KRAKEN2_DB"

# 4) Loop over FASTA files and add them
echo "Adding FASTA files to library..."
for f in "$FASTAS_DIR"/*.fna*; do
  if [[ -f "$f" ]]; then
    echo "  Adding: $f"
    k2 add-to-library --file "$f" --db "$KRAKEN2_DB"
  fi
done

# 5) Build the Kraken2 database
#    Adjust --threads if desired, e.g. --threads 8
echo "Building Kraken2 database..."
kraken2-build --build --db "$KRAKEN2_DB" --threads 8


echo "Kraken2 database creation complete!"
echo "DB location: $KRAKEN2_DB"
