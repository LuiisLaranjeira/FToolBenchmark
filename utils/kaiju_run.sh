#!/usr/bin/env bash

# Usage: ./kaiju_run.sh [OUT_DIR]

# Get OUT_DIR from first argument or use default (Directory containing kaiju output files)
OUT_DIR="${1:-output/KAIJU}"

# Output directory for reports
REPORTS_DIR="reports_output/KAIJU"

# Create the reports directory if it doesn't exist
mkdir -p "${REPORTS_DIR}"

# Loop over each kaiju output file in OUT_DIR
for FILE in "${OUT_DIR}"/*.fq; do
  
  # Extract the filename without the directory and extension
  BASENAME=$(basename "${FILE}" .txt)
  
  # Construct the output report filename
  REPORT_FILE="${REPORTS_DIR}/${BASENAME}_report.txt"
  
  # Run
  /usr/bin/time -v -- kaiju -t kraken2_db/taxonomy/nodes.dmp -f kaiju_db.fmi -i "${FILE}" -z 8 -o "${REPORT_FILE}"
    
  echo "Generated report: ${REPORT_FILE}"
done
