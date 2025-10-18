#!/bin/bash

# Usage information
usage() {
    echo "Usage: $0 <database_name> <input_directory>"
    exit 1
}

# Check number of arguments
if [ $# -ne 2 ]; then
    usage
fi

DBNAME="$1"
INPUT_DIR="$2"

# Validate database name
if [ -z "$DBNAME" ]; then
    echo "Error: Database name is empty."
    usage
fi

# Validate directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory '$INPUT_DIR' not found."
    exit 1
fi

# Check if any matching files exist
if [ ! "$(ls -A "${INPUT_DIR}"/*.fna 2>/dev/null)" ]; then
    echo "Error: No .fna files found in /$INPUT_DIR"
    exit 1
fi

# Process files
for file in "${INPUT_DIR}"/*.fna
do
    echo "Processing $file..."
    if ! k2 add-to-library --file "$file" --db "$DBNAME"; then
        echo "Error processing $file"
        exit 1
    fi
done

echo "All files processed successfully"