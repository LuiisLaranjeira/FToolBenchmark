#!/bin/bash

# Define parameter arrays
depths=(1 2 5 10 20 40 60)
read_sizes=(20 30 40 50 75 100 150)
deaminations=(0 0.1 0.2 0.3)

# Define input directory
INPUT_DIR="data/"
OUTPUT_DIR="simulated_reads"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all combinations
for depth in "${depths[@]}"; do
  for read_size in "${read_sizes[@]}"; do
    for deamination in "${deaminations[@]}"; do
      # Define output prefix
      OUTPUT_PREFIX="${OUTPUT_DIR}/sim_depth${depth}_read${read_size}_deam${deamination}"
      
      # Define deamination option
      if (( $(echo "$deamination == 0" | bc -l) )); then
        DEAM_OPTION=""
      else
        # Use constant: v=0.05, l=1
        v=0.03
        l=0.4
        d=0.01
        s="$deamination"
        DEAM_OPTION="-damage ${v},${l},${d},${s}"
      fi
      
      # Run Gargammel
      gargammel --comp 0.6,0.02,0.38 \
               -o "$OUTPUT_PREFIX" \
               -c "$depth" \
               -rl "$read_size" \
               $DEAM_OPTION \
               -se \
               "$INPUT_DIR"
      
      echo "Completed: Depth=${depth}x, Read Size=${read_size}bp, Deamination=${deamination}"
    done
  done
done

echo "All simulations completed."
