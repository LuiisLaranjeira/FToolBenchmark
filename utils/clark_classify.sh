#!/usr/bin/env bash
# Usage: ./clark_classify.sh [INPUT_DIR] [OUTPUT_DIR]
# Classify metagenome samples using CLARK with classify_metagenome.sh

# Get directories from arguments or use defaults
INPUT_DIR="${1:-input/fastq}"          # Directory containing input FASTQ files
OUTPUT_DIR="${2:-output/CLARK}"        # Directory for classification outputs

# Create the output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Check if input directory exists
if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "Error: Input directory '${INPUT_DIR}' does not exist"
    echo "Usage: $0 [INPUT_DIR] [OUTPUT_DIR]"
    exit 1
fi

# Check if classify_metagenome.sh exists and is executable
if [[ ! -x "./classify_metagenome.sh" ]]; then
    echo "Error: classify_metagenome.sh not found or not executable"
    exit 1
fi

# Process each FASTQ file in the input directory
for INPUT_FILE in "${INPUT_DIR}"/*.{fq,fastq,fq.gz,fastq.gz}; do
    # Skip if no files match the pattern
    [[ ! -e "${INPUT_FILE}" ]] && continue
    
    # Extract the filename without directory and extension(s)
    BASENAME=$(basename "${INPUT_FILE}")
    BASENAME="${BASENAME%.gz}"        # Remove .gz if present
    BASENAME="${BASENAME%.fq}"        # Remove .fq
    BASENAME="${BASENAME%.fastq}"     # Remove .fastq
    
    # Construct the output path for this sample
    SAMPLE_OUTPUT="${OUTPUT_DIR}/${BASENAME}"
    
    echo "Processing: ${INPUT_FILE}"
    echo "Output to: ${SAMPLE_OUTPUT}"
    
    # Run classify_metagenome.sh with the specified parameters
    if /usr/bin/time -v -- ./classify_metagenome.sh \
        -O "${INPUT_FILE}" \
        -R "${SAMPLE_OUTPUT}" \
        --light \
        -n 8; then
        echo "✓ Successfully processed: ${BASENAME}"
    else
        echo "✗ Failed to process: ${BASENAME}"
    fi
    
    echo "----------------------------------------"
done

echo "Classification complete. Results in: ${OUTPUT_DIR}"