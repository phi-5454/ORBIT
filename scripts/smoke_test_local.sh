#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TRAIN_VAL_FILES="${TRAIN_VAL_FILES:-${REPO_DIR}/pq_files_train_val.txt}"
TEST_FILES="${TEST_FILES:-${REPO_DIR}/pq_files_test.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/smoke_tests}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"

COMMON_ARGS=(
  "train_val_files=${TRAIN_VAL_FILES}"
  "test_files=${TEST_FILES}"
  "output_dir=${OUTPUT_DIR}"
  "run_test=True"
  "trainer.limit_train_batches=2"
  "trainer.limit_val_batches=1"
  "trainer.limit_test_batches=1"
  "trainer.max_epochs=1"
  "model.batch_size=${BATCH_SIZE}"
  "num_dataload_workers=${NUM_WORKERS}"
  "model.use_attention=True"
)

cd "${REPO_DIR}"

echo "Running local FSQ smoke test"
LINE_PROFILE=0 uv run python src/main.py \
  "${COMMON_ARGS[@]}" \
  "run_name=local_smoke_fsq" \
  "model.quantizer=fsq" \
  "model.mu_quantizer=fsq" \
  "model.alpha_quantizer=fsq" \
  "model.fsq_mu_levels=[21,21,21]" \
  "model.fsq_alpha_levels=[]"

echo "Running local VQ smoke test"
LINE_PROFILE=0 uv run python src/main.py \
  "${COMMON_ARGS[@]}" \
  "run_name=local_smoke_vq" \
  "model.quantizer=vq" \
  "model.mu_quantizer=vq" \
  "model.alpha_quantizer=vq" \
  "model.fsq_mu_levels=[]" \
  "model.fsq_alpha_levels=[]" \
  "model.vq_mu_dim=3" \
  "model.vq_alpha_dim=0" \
  "model.vq_mu_num_codes=256" \
  "model.vq_alpha_num_codes=0" \
  "model.vq_gradient_estimator=ste"

echo "Local smoke tests completed"
