#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Stage 2: Align 4-modality embeddings and train disentangled representations
EMBEDDINGS_DIR="data"
OUTPUT_DIR="/path/to/your_output"
MODEL_CONFIG="config/model.json"

# 1. Align and concatenate multimodal embeddings based on H3 indices
python -m IntraModality.concat \
    --data-dir "$EMBEDDINGS_DIR" \
    --output-dir "$OUTPUT_DIR/aligned"

# 2. Disentangle multimodal representations
python -m DisentanglePositive.main \
    --data-dir "$OUTPUT_DIR/aligned" \
    --config "$MODEL_CONFIG" \
    --output-dir "$OUTPUT_DIR/representations"
