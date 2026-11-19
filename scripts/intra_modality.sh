#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Stage 1: Intra-modality H3 mapping and representation enhancement (raw data -> 4-modality embeddings)
RAW_DATA_DIR="/path/to/your_raw_data"
OUTPUT_DIR="/path/to/your_output"
CITY="your_city"

# 1. Map raw data from four modalities to a unified H3 grid
python -m IntraModality.poi.map_h3 "$CITY" \
    --source-dir "$RAW_DATA_DIR/poi" \
    --regions "$RAW_DATA_DIR/regions.csv" \
    --output-root "$OUTPUT_DIR/processed"

python -m IntraModality.mobility_flow.map_h3 "$CITY" \
    --source-csv "$RAW_DATA_DIR/flow.csv" \
    --regions "$RAW_DATA_DIR/regions.csv" \
    --output-dir "$OUTPUT_DIR/processed"

python -m IntraModality.map_images_to_h3 streetview \
    "$RAW_DATA_DIR/streetview.csv" \
    "$RAW_DATA_DIR/regions.csv" \
    "$OUTPUT_DIR/processed/$CITY"

python -m IntraModality.map_images_to_h3 satellite \
    "$RAW_DATA_DIR/satellite.tif" \
    "$RAW_DATA_DIR/regions.csv" \
    "$OUTPUT_DIR/processed/$CITY"

# 2. Generate embeddings using four modality-specific encoders
python -m IntraModality.poi.main \
    --city "$CITY" \
    --data-dir "$OUTPUT_DIR/processed" \
    --output-dir "$OUTPUT_DIR/embeddings"

python -m IntraModality.mobility_flow.main \
    --city "$CITY" \
    --data-dir "$OUTPUT_DIR/processed" \
    --output-dir "$OUTPUT_DIR/embeddings"

python -m IntraModality.streetview.main \
    --city "$CITY" \
    --data-dir "$OUTPUT_DIR/processed" \
    --output-dir "$OUTPUT_DIR/embeddings"

python -m IntraModality.satellite.main \
    --city "$CITY" \
    --data-dir "$OUTPUT_DIR/processed" \
    --output-dir "$OUTPUT_DIR/embeddings"
