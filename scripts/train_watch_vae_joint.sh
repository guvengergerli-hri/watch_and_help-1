#!/usr/bin/env bash

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  python -m watch.vae_joint_train --help
  exit 0
fi

EPOCHS="${EPOCHS:-200}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
LATENT_SIZE="${LATENT_SIZE:-32}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-0.0003}"
KL_WEIGHT="${KL_WEIGHT:-0.01}"
KL_WARMUP_EPOCHS="${KL_WARMUP_EPOCHS:-20}"
KL_WARMUP_START_FACTOR="${KL_WARMUP_START_FACTOR:-0.0}"
FREE_BITS="${FREE_BITS:-0.0}"
USE_ACTIONS="${USE_ACTIONS:-1}"
RECONSTRUCT_ACTIONS="${RECONSTRUCT_ACTIONS:-1}"
ACTION_WEIGHT="${ACTION_WEIGHT:-0.1}"
PREDICATE_AUX_WEIGHT="${PREDICATE_AUX_WEIGHT:-0.5}"
LABELED_FRACTION="${LABELED_FRACTION:-1.0}"
LABEL_MASK_MODE="${LABEL_MASK_MODE:-fixed}"
LABEL_MASK_SEED="${LABEL_MASK_SEED:-}"
POS_WEIGHT_MIN="${POS_WEIGHT_MIN:-1.0}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-20.0}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
GPU_ID="${GPU_ID:-}"
DEVICE="${DEVICE:-}"

EXTRA_ARGS=()
if [[ "$USE_ACTIONS" == "1" ]]; then
  EXTRA_ARGS+=(--use-actions)
fi
if [[ "$RECONSTRUCT_ACTIONS" == "1" ]]; then
  EXTRA_ARGS+=(--reconstruct-actions)
else
  EXTRA_ARGS+=(--no-reconstruct-actions)
fi
if [[ "$USE_ACTIONS" != "1" && "$RECONSTRUCT_ACTIONS" == "1" ]]; then
  echo "RECONSTRUCT_ACTIONS=1 requires USE_ACTIONS=1"
  exit 1
fi

# If DEVICE is unspecified or generic "cuda", use GPU_ID to select an explicit CUDA index.
if [[ -n "$GPU_ID" && ( -z "$DEVICE" || "$DEVICE" == "cuda" ) ]]; then
  DEVICE="cuda:${GPU_ID}"
fi
if [[ -n "$DEVICE" ]]; then
  EXTRA_ARGS+=(--device "$DEVICE")
fi

EXTRA_ARGS+=(
  --action-weight "$ACTION_WEIGHT"
  --predicate-aux-weight "$PREDICATE_AUX_WEIGHT"
  --labeled-fraction "$LABELED_FRACTION"
  --label-mask-mode "$LABEL_MASK_MODE"
  --pos-weight-min "$POS_WEIGHT_MIN"
  --pos-weight-max "$POS_WEIGHT_MAX"
)
if [[ -n "$LABEL_MASK_SEED" ]]; then
  EXTRA_ARGS+=(--label-mask-seed "$LABEL_MASK_SEED")
fi

echo "Watch joint VAE device: ${DEVICE:-<trainer default>}"
if [[ -n "$GPU_ID" ]]; then
  echo "Watch joint VAE GPU_ID env: $GPU_ID"
fi
echo "Watch joint VAE labeled fraction: $LABELED_FRACTION (mode=$LABEL_MASK_MODE seed=${LABEL_MASK_SEED:-<seed-arg>})"
echo "Watch joint VAE num_workers: $NUM_WORKERS | log_interval: $LOG_INTERVAL"

python -m watch.vae_joint_train \
  --metadata dataset/watch_data/metadata.json \
  --train-json dataset/watch_data/gather_data_actiongraph_train.json \
  --val-json dataset/watch_data/gather_data_actiongraph_test.json \
  --train-split-key train_data \
  --val-split-key test_data \
  --output-dir checkpoints/watch_vae_joint \
  --tensorboard-logdir log_tb/watch_vae_joint \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --hidden-size "$HIDDEN_SIZE" \
  --latent-size "$LATENT_SIZE" \
  --transformer-nhead 2 \
  --stable-slots \
  --kl-weight "$KL_WEIGHT" \
  --kl-warmup-epochs "$KL_WARMUP_EPOCHS" \
  --kl-warmup-start-factor "$KL_WARMUP_START_FACTOR" \
  --free-bits "$FREE_BITS" \
  --class-weight 1.0 \
  --state-weight 1.0 \
  --coord-weight 1.0 \
  --mask-weight 0.2 \
  --logvar-min -10 \
  --logvar-max 10 \
  --log-interval "$LOG_INTERVAL" \
  --lr "$LR" \
  --save-teacher-encoder \
  --teacher-prefix teacher_encoder \
  --teacher-scope backbone \
  "${EXTRA_ARGS[@]}" \
  "$@"

# Example:
# KL_WEIGHT=0.01 KL_WARMUP_EPOCHS=20 FREE_BITS=0.01 USE_ACTIONS=1 RECONSTRUCT_ACTIONS=1 \
# ACTION_WEIGHT=0.1 PREDICATE_AUX_WEIGHT=0.5 LABELED_FRACTION=0.1 LABEL_MASK_MODE=fixed LABEL_MASK_SEED=123 \
# bash scripts/train_watch_vae_joint.sh
# GPU_ID=2 NUM_WORKERS=8 LOG_INTERVAL=50 bash scripts/train_watch_vae_joint.sh
