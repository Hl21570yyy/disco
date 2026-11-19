#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Stage 3: Train and evaluate downstream tasks from disentangled representations
REPRESENTATIONS_DIR="checkpoint"
OUTPUT_DIR="/path/to/your_output"
LABELS_PATH="data/singapore_gdp.csv"
LABEL_COLUMN="gdp"
TASK="regression"
DOWNSTREAM_CONFIG="config/downstream.json"
SPLIT_PATH="data/sg_gdp_split.json"

EXTRA_ARGS=()
if [[ -n "$SPLIT_PATH" ]]; then
    EXTRA_ARGS+=(--split "$SPLIT_PATH")
fi
if [[ "$TASK" == "regression" ]]; then
    EXTRA_ARGS+=(--log-target)
fi

python -m DownstreamTask.main \
    --data-dir "$REPRESENTATIONS_DIR" \
    --labels "$LABELS_PATH" \
    --label-column "$LABEL_COLUMN" \
    --task "$TASK" \
    --config "$DOWNSTREAM_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
