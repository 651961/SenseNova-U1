# bash export_hf.sh \
#     /datasets/codes_zsqiao/SenseNova-U1/training/RUN/sensenovau1_5_8b_pt/snapshot/1 \
#     /models/SenseNova-U1.5-8B-MoT-Preview-Layered-step2000 \
#     normal

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="${SCRIPT_DIR}/training"
DEFAULT_BASE_MODEL="/models/SenseNova-U1.5-8B-MoT-Preview-Layered"

usage() {
    printf '%s\n' \
        "Usage:" \
        "  bash export_hf.sh CHECKPOINT_DIR OUTPUT_DIR [ema|normal] [BASE_MODEL_DIR]" \
        "" \
        "Example (normal weights):" \
        "  bash export_hf.sh \\" \
        "    training/RUN/sensenovau1_8b_smoke_test/snapshot/1 \\" \
        "    /models/SenseNova-U1-8B-MoT-Infographic-ft-step1000-normal \\" \
        "    normal"
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if (( $# < 2 || $# > 4 )); then
    usage >&2
    exit 2
fi

CHECKPOINT_DIR="$1"
OUTPUT_DIR="$2"
WEIGHT_TYPE="${3:-ema}"
BASE_MODEL_DIR="${4:-${MODEL_NAME_OR_PATH:-$DEFAULT_BASE_MODEL}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

[[ -d "$CHECKPOINT_DIR" ]] || die "checkpoint directory does not exist: $CHECKPOINT_DIR"
[[ -f "$CHECKPOINT_DIR/model_config.pt" ]] || die "missing $CHECKPOINT_DIR/model_config.pt"
[[ -d "$BASE_MODEL_DIR" ]] || die "base HF model directory does not exist: $BASE_MODEL_DIR"
[[ -f "$BASE_MODEL_DIR/config.json" ]] || die "missing $BASE_MODEL_DIR/config.json"
[[ -f "$BASE_MODEL_DIR/tokenizer_config.json" ]] || die "missing $BASE_MODEL_DIR/tokenizer_config.json"

# Canonical paths keep symlinks in the temporary EMA staging directory valid.
CHECKPOINT_DIR="$(cd -- "$CHECKPOINT_DIR" && pwd -P)"
BASE_MODEL_DIR="$(cd -- "$BASE_MODEL_DIR" && pwd -P)"

if [[ -e "$OUTPUT_DIR" && ! -d "$OUTPUT_DIR" ]]; then
    die "output path exists and is not a directory: $OUTPUT_DIR"
fi
if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "output directory is not empty; use a new directory: $OUTPUT_DIR"
fi

shopt -s nullglob
TEMP_SOURCE=""

cleanup() {
    if [[ -n "$TEMP_SOURCE" && -d "$TEMP_SOURCE" ]]; then
        rm -rf -- "$TEMP_SOURCE"
    fi
}
trap cleanup EXIT

case "$WEIGHT_TYPE" in
    ema)
        EMA_DIR="$CHECKPOINT_DIR/averaged_model"
        [[ -d "$EMA_DIR" ]] || die "EMA directory does not exist: $EMA_DIR"

        EMA_SHARDS=("$EMA_DIR"/model_*.pt)
        (( ${#EMA_SHARDS[@]} > 0 )) || die "no EMA model shards found in $EMA_DIR"

        TEMP_SOURCE="$(mktemp -d)"
        cp -- "$CHECKPOINT_DIR/model_config.pt" "$TEMP_SOURCE/"
        for shard in "${EMA_SHARDS[@]}"; do
            ln -s -- "$shard" "$TEMP_SOURCE/$(basename -- "$shard")"
        done
        SOURCE_DIR="$TEMP_SOURCE"
        ;;
    normal)
        NORMAL_SHARDS=("$CHECKPOINT_DIR"/model_wp*_pp*.pt)
        (( ${#NORMAL_SHARDS[@]} > 0 )) || die "no normal model shards found in $CHECKPOINT_DIR"
        SOURCE_DIR="$CHECKPOINT_DIR"
        ;;
    *)
        die "weight type must be 'ema' or 'normal', got: $WEIGHT_TYPE"
        ;;
esac

export PYTHONPATH="${TRAINING_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

printf 'Checkpoint : %s\n' "$CHECKPOINT_DIR"
printf 'Weights    : %s\n' "$WEIGHT_TYPE"
printf 'Base model : %s\n' "$BASE_MODEL_DIR"
printf 'Output     : %s\n' "$OUTPUT_DIR"

"$PYTHON_BIN" "$TRAINING_DIR/tools/revert2hf.py" \
    --src "$SOURCE_DIR" \
    --tgt "$OUTPUT_DIR" \
    --typ neo++_mot \
    --extras-from "$BASE_MODEL_DIR"

[[ -f "$OUTPUT_DIR/model.safetensors.index.json" ]] || die "conversion finished without model.safetensors.index.json"
[[ -f "$OUTPUT_DIR/config.json" ]] || die "conversion finished without config.json"
[[ -f "$OUTPUT_DIR/tokenizer_config.json" ]] || die "conversion finished without tokenizer_config.json"
OUTPUT_SHARDS=("$OUTPUT_DIR"/model-*.safetensors)
(( ${#OUTPUT_SHARDS[@]} > 0 )) || die "conversion finished without safetensors shards"

printf '\nExport complete: %s\n' "$OUTPUT_DIR"
printf 'Use it for inference with: --model_path %s\n' "$OUTPUT_DIR"
