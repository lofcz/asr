#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 UNIQUE_RUN_NAME SESSION_SECONDS UPDATE_COUNT" >&2
    echo "Session buckets: 600, 1800, 3600. Runs one job across both GPUs." >&2
    exit 2
fi
case "$2" in 600|1800|3600) ;; *) echo 'Invalid session bucket' >&2; exit 2 ;; esac
[[ "$3" =~ ^[1-9][0-9]*$ ]] || { echo 'Update count must be positive' >&2; exit 2; }
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
venv_path="${ASR_VENV_PATH:-/data/work/asr-pilot/venv}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS=2
extra=()
if [[ "$2" == 3600 ]]; then extra+=(--offload --pageable-offload); fi
exec "$venv_path/bin/torchrun" --standalone --nproc-per-node=2 train_distributed.py \
    --name "$1" --mode qlora --bucket "$2" --steps "$3" \
    --rank 16 --lr 1e-5 --mlp-chunk 1024 --save "${extra[@]}"
