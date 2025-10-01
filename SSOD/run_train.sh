#!/usr/bin/env bash

# Flexible training launcher.
# You can override OUTPUT_DIR by:
#   1) passing it as the first arg:   ./run_train.sh /path/to/output
#   2) or via env var:                OUT_DIR=/path/to/output ./run_train.sh
# Defaults to ./output under unbiased-teacher.

set -euo pipefail

# Allow overriding via environment variables
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
# If NUM_GPUS is not provided, infer from CUDA_VISIBLE_DEVICES
if [[ -z "${NUM_GPUS:-}" ]]; then
  IFS=',' read -ra DEV_ARR <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS=${#DEV_ARR[@]}
fi

CONFIG=${CONFIG:-/path/to/unbiased-teacher/configs/brats21_supervision/faster_rnn_R_50_FPN_sup5_run1.yaml} #set your config file here

# Per-GPU batch; totals must be divisible by NUM_GPUS (UBTeacher enforces this)
PER_GPU_LABEL=${PER_GPU_LABEL:-8}
PER_GPU_UNLABEL=${PER_GPU_UNLABEL:-8}
TOTAL_LABEL=$(( PER_GPU_LABEL * NUM_GPUS ))
TOTAL_UNLABEL=$(( PER_GPU_UNLABEL * NUM_GPUS ))

# DataLoader workers per process (tune based on CPU cores)
WORKERS=${WORKERS:-4}

# Enable AMP (mixed precision) by default for speed unless explicitly disabled
AMP_ENABLED=${AMP_ENABLED:-True}
# Normalize to Python boolean literals expected by yacs (True/False)
case "${AMP_ENABLED}" in
  1|true|TRUE|yes|Yes) AMP_VAL=True ;;
  0|false|FALSE|no|No) AMP_VAL=False ;;
  True|False) AMP_VAL=${AMP_ENABLED} ;;
  *) AMP_VAL=True ;;
esac

# Resolve output directory from arg or env vars, fallback to ./output
# Honors either OUT_DIR or OUTPUT_DIR environment variables
OUT_DIR=${1:-${OUT_DIR:-${OUTPUT_DIR:-./output}}}

# Unique dist URL per run to avoid EADDRINUSE when launching multiple jobs
if [[ -z "${DIST_URL:-}" ]]; then
  # Find a free local TCP port
  FREE_PORT=$(python - <<'PY'
import socket
s=socket.socket()
s.bind(('',0))
print(s.getsockname()[1])
PY
)
  DIST_URL="tcp://127.0.0.1:${FREE_PORT}"
fi

python train_net.py \
      --num-gpus "${NUM_GPUS}" \
      --config "${CONFIG}" \
      --dist-url "${DIST_URL}" \
      OUTPUT_DIR "${OUT_DIR}" \
      DATALOADER.NUM_WORKERS "${WORKERS}" \
      SOLVER.IMG_PER_BATCH_LABEL "${TOTAL_LABEL}" \
      SOLVER.IMG_PER_BATCH_UNLABEL "${TOTAL_UNLABEL}" \
      SOLVER.AMP.ENABLED "${AMP_VAL}"


# python train_net.py \
#       --eval-only \
#       --num-gpus 1 \
#       --config configs/brats21_supervision/faster_rnn_R_50_FPN_sup5_run1.yaml \
#       OUTPUT_DIR ./output_eval \
#       SOLVER.IMG_PER_BATCH_LABEL 32 SOLVER.IMG_PER_BATCH_UNLABEL 32 MODEL.WEIGHTS /root/autodl-tmp/unbiased-teacher/output/model_0002999.pth

# python train_net.py \
#       --resume \
#       --num-gpus 1 \
#       --config configs/brats21_supervision/faster_rnn_R_50_FPN_sup5_run1.yaml \
#       OUTPUT_DIR ./output_resume \
#       SOLVER.IMG_PER_BATCH_LABEL 32 SOLVER.IMG_PER_BATCH_UNLABEL 32 MODEL.WEIGHTS /root/autodl-tmp/unbiased-teacher/output_10/model_0002499.pth
