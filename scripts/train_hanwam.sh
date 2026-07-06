#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-control/HanWAM/config/hanwam.yml}"
MODE="${MODE:-1}"
STAGE="${STAGE:-both}"
EXTRA_ARGS=()

if [[ "${SWANLAB:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--swanlab)
else
  EXTRA_ARGS+=(--no-swanlab)
fi

if [[ -n "${DEVICE:-}" ]]; then
  EXTRA_ARGS+=(--device "$DEVICE")
fi

if [[ -n "${LIMIT_TRANSITIONS:-}" ]]; then
  EXTRA_ARGS+=(--limit-transitions "$LIMIT_TRANSITIONS")
fi

if [[ -n "${STAGE1_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--stage1-checkpoint "$STAGE1_CHECKPOINT")
fi

python -m control.HanWAM.train \
  --config "$CONFIG" \
  --mode "$MODE" \
  --stage "$STAGE" \
  "${EXTRA_ARGS[@]}"
