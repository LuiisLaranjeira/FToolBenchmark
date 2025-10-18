#!/usr/bin/env bash

# Usage: ./falcon2_run.sh [OUT_DIR]

# Get OUT_DIR from first argument or use default (Directory containing falcon output files)
OUT_DIR="${1:-output/falcon}"

# Output directory for reports
REPORTS_DIR="reports_output/FALCON2"

# Create the reports directory if it doesn't exist
mkdir -p "${REPORTS_DIR}"

# Loop over each falcon output file in OUT_DIR
for FILE in "${OUT_DIR}"/*.fq; do
  
  # Extract the filename without the directory and extension
  BASENAME=$(basename "${FILE}" .txt)
  
  # Construct the output report filename
  REPORT_FILE="${REPORTS_DIR}/${BASENAME}_report.txt"
  
  # Run
  /usr/bin/time -v -- ./FALCON2 meta -v -F -t 15 -l 47 -n 8 -x "${REPORT_FILE}" "${FILE}" input-sequences.fna
    
  echo "Generated report: ${REPORT_FILE}"
done
