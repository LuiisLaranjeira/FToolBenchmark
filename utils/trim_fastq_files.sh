#!/bin/bash

# Script to trim FASTQ files using AdapterRemoval
# Usage: ./trim_fastq_files.sh <input_directory> <output_directory>

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    echo "Example: $0 ./raw_reads ./trimmed_reads"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Count total files to process
total_files=$(find "$INPUT_DIR" -maxdepth 1 -name "*.fq" -o -name "*.fastq" | wc -l)

if [ "$total_files" -eq 0 ]; then
    echo "No .fq or .fastq files found in $INPUT_DIR"
    exit 1
fi

echo "Found $total_files FASTQ files to process"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "----------------------------------------"

# Counter for processed files
processed=0

# Process each .fq or .fastq file
for fastq_file in "$INPUT_DIR"/*.fq "$INPUT_DIR"/*.fastq; do
    # Skip if no files match (handles case where one extension doesn't exist)
    [ -e "$fastq_file" ] || continue
    
    # Get the basename without path
    filename=$(basename "$fastq_file")
    # Remove extension
    basename_no_ext="${filename%.fq}"
    basename_no_ext="${basename_no_ext%.fastq}"
    
    echo "Processing: $filename"
    
    # Run AdapterRemoval
	AdapterRemoval \
	  --file1 "$fastq_file" \
	  --basename "${OUTPUT_DIR}/${basename_no_ext}" \
	  --trimns --trimqualities --minquality 2 \
	  --minlength 20 \
	  --threads 8
    
    # Check if AdapterRemoval succeeded
    if [ $? -eq 0 ]; then
        ((processed++))
        echo "  ✓ Successfully processed $filename ($processed/$total_files)"
    else
        echo "  ✗ Error processing $filename"
    fi
    
    echo "----------------------------------------"
done

echo "Processing complete!"
echo "Processed $processed out of $total_files files"
echo "Results saved to: $OUTPUT_DIR"
