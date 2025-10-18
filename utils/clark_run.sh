#!/usr/bin/env bash

# Usage: ./clark_run.sh [OUT_DIR]

# Get OUT_DIR from first argument or use default (Directory containing clark output files)
OUT_DIR="${1:-output/CLARK}"

# Output directory for reports
REPORTS_DIR="reports_output/CLARK"

# Create the reports directory if it doesn't exist
mkdir -p "${REPORTS_DIR}"

# Loop over each clark output file in OUT_DIR
for FILE in "${OUT_DIR}"/*.csv; do
  
  # Extract the filename without the directory and extension
  BASENAME=$(basename "${FILE}" .txt)
  
  # Construct the output report filename
  REPORT_FILE="${REPORTS_DIR}/${BASENAME}_report.txt"
  
  # Run clark2table
  /usr/bin/time -v -- ./estimate_abundance.sh \
    -D CLARK_db/ \
    -F "${FILE}" \
    > "${REPORT_FILE}"
    
  echo "Generated report: ${REPORT_FILE}"
done
