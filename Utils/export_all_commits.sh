#!/bin/bash

# Output directory
OUTPUT_DIR="commits_export"
mkdir -p "$OUTPUT_DIR"

# Get all commits (newest first)
COMMITS=$(git log --reverse --pretty=format:"%H")

# Loop through each commit
for COMMIT in $COMMITS; do
    # Get metadata
    COMMIT_DATE=$(git show -s --format=%cd --date=short $COMMIT)
    COMMIT_MSG=$(git show -s --format=%s $COMMIT | tr -cd '[:alnum:]_-' | cut -c1-30)
    
    # Folder name: date_hash
    EXPORT_DIR="$OUTPUT_DIR/${COMMIT_DATE}_${COMMIT:0:7}_${COMMIT_MSG}"
    mkdir -p "$EXPORT_DIR"

    echo "Exporting commit: $COMMIT to $EXPORT_DIR"

    # Export as tar
    git archive --format=tar --output="$EXPORT_DIR/files.tar" "$COMMIT"
    tar -xf "$EXPORT_DIR/files.tar" -C "$EXPORT_DIR"
    rm "$EXPORT_DIR/files.tar"
done

echo "✅ Done! Exported to: $OUTPUT_DIR/"
