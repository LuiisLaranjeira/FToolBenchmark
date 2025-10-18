#!/usr/bin/env bash

# Usage: ./kraken2_run.sh [OUT_DIR]

# Get OUT_DIR from first argument or use default (Directory containing kraken2 output files)
OUT_DIR="${1:-output/KRAKEN}"

# Output directory for reports
REPORTS_DIR="reports_output/KRAKEN"

# Create the reports directory if it doesn't exist
mkdir -p "${REPORTS_DIR}"

# Loop over each kraken2 output file in OUT_DIR
for FILE in "${OUT_DIR}"/*.fq; do
  
  # Extract the filename without the directory and extension
  BASENAME=$(basename "${FILE}" .txt)
  
  # Construct the output report filename
  REPORT_FILE="${REPORTS_DIR}/${BASENAME}_report.txt"
  
  # Run
  /usr/bin/time -v -- kraken2 -db kraken2_db/ --threads 8 "${FILE}" > "${REPORT_FILE}"
    
  echo "Generated report: ${REPORT_FILE}"
done
