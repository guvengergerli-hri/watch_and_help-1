#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  python -m watch.vae_predicate_vectors_train --help
  exit 0
fi

# Allow passing checkpoint as first positional argument:
#   bash scripts/train_watch_vae_predicate_vectors.sh checkpoints/watch_vae/<run>/watch_vae_best.pt
if [[ -z "${VAE_CKPT:-}" && $# -gt 0 && "${1:-}" != -* ]]; then
  VAE_CKPT="$1"
  shift
fi

: "${VAE_CKPT:?Set VAE_CKPT to a stage-1 VAE checkpoint path (watch_vae_best.pt/watch_vae_last.pt)}"
GPU_ID="${GPU_ID:-}"
DEVICE="${DEVICE:-}"
EXTRA_ARGS=(--vae-checkpoint "$VAE_CKPT")

# VAE_CKPT=checkpoints/watch_vae/20260222_164327_hid128_lat64_lr0p0003_bs8_act0_ss1_kl0p01_cw1p0_sw1p0_ow1p0_mw0p2/best_model.pt DEVICE=cuda:2 bash scripts/train_watch_vae_predicate_vectors.sh 

# GPU/device selection (precedence: explicit DEVICE env > GPU_ID env > trainer default).
# Examples:
#   VAE_CKPT=checkpoints/watch_vae/<run>/watch_vae_best.pt GPU_ID=1 bash scripts/train_watch_vae_predicate_vectors.sh
#   VAE_CKPT=... DEVICE=cuda:2 bash scripts/train_watch_vae_predicate_vectors.sh
#   VAE_CKPT=... DEVICE=cpu bash scripts/train_watch_vae_predicate_vectors.sh
# If DEVICE is unspecified or generic "cuda", use GPU_ID to select an explicit CUDA index.
if [[ -n "$GPU_ID" && ( -z "$DEVICE" || "$DEVICE" == "cuda" ) ]]; then
  DEVICE="cuda:${GPU_ID}"
fi
if [[ -n "$DEVICE" ]]; then
  EXTRA_ARGS+=(--device "$DEVICE")
fi

echo "VAE predicate-vector stage-2 source checkpoint: $VAE_CKPT"
echo "VAE predicate-vector device: ${DEVICE:-<trainer default>}"
if [[ -n "$GPU_ID" ]]; then
  echo "VAE predicate-vector GPU_ID env: $GPU_ID"
fi

python -m watch.vae_predicate_vectors_train \
  --metadata dataset/watch_data/metadata.json \
  --train-json dataset/watch_data/gather_data_actiongraph_train.json \
  --val-json dataset/watch_data/gather_data_actiongraph_test.json \
  --train-split-key train_data \
  --val-split-key test_data \
  --output-dir checkpoints/watch_vae_predicate_vectors \
  --tensorboard-logdir log_tb/watch_vae_predicate_vectors \
  --batch-size 8 \
  --epochs 100 \
  --stable-slots \
  --labeled-fraction 1.0 \
  --label-mask-mode fixed \
  --label-mask-seed 123 \
  --lr 0.0003 \
  "${EXTRA_ARGS[@]}" \
  "$@"

# Example for partial labels:
# VAE_CKPT=... bash scripts/train_watch_vae_predicate_vectors.sh --labeled-fraction 0.5
