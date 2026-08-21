#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CHECKPOINT="training/RUN/sensenovau1_5_8b_pt/1500"
BASE_MODEL="/models/SenseNova-U1.5-8B-MoT"

# Export EMA weights.
WEIGHTS_SRC="$CHECKPOINT/averaged_model"
OUTPUT="${CHECKPOINT}_ema_hf"

# To export live weights instead, comment out the two lines above and use:
# WEIGHTS_SRC="$CHECKPOINT"
# OUTPUT="${CHECKPOINT}_hf"

python training/tools/revert2hf.py \
    --src "$CHECKPOINT" \
    --weights-src "$WEIGHTS_SRC" \
    --tgt "$OUTPUT" \
    --extras-from "$BASE_MODEL"
