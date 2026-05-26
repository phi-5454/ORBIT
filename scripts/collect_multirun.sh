#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 WANTED_OUTPUT_DIR SUITE_ID [HYDRA_OVERRIDES...]"
  exit 2
fi

WANTED_OUTPUT_DIR="$1"
SUITE_ID="$2"
EXTRA_ARGS=("${@:3}")

PROJECT_DIR="/eos/user/y/yelberke/Orbit_proj"
CONDA_ENV="/eos/user/y/yelberke/conda_condor_orbit_env"
#CONDA_ENV="/eos/user/y/yelberke/conda_Orbit_env"
TRAIN_FILES="/eos/user/y/yelberke/collideV2_train_val"
TEST_FILES="/eos/user/y/yelberke/collideV2_test"

mkdir -p "${WANTED_OUTPUT_DIR}/condor_logs"
mkdir -p "${WANTED_OUTPUT_DIR}/${SUITE_ID}"

export WANDB_DIR="${WANTED_OUTPUT_DIR}/${SUITE_ID}/wandb"
mkdir -p "${WANDB_DIR}"

export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${WANTED_OUTPUT_DIR}/${SUITE_ID}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

COMMON_ARGS=(
  "train_val_files=${TRAIN_FILES}"
  "test_files=${TEST_FILES}"
  "output_dir=${WANTED_OUTPUT_DIR}"
  "trainer.limit_train_batches=10000"
  "trainer.limit_val_batches=1000"
  "trainer.limit_test_batches=1000"
  "trainer.max_epochs=30"
  "model.batch_size=128"
  "model.fsq_alpha_levels=[]"
  "num_dataload_workers=4"
  "model.use_attention=True"
  "run_test=True"
)

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "Collecting multirun suite ${SUITE_ID}"

LINE_PROFILE=0 conda run --no-capture-output -p "${CONDA_ENV}" \
  python -m src.main "${COMMON_ARGS[@]}" \
  "multirun.enabled=True" \
  "multirun.suite_id=${SUITE_ID}" \
  "multirun.collect_only=True" \
  "${EXTRA_ARGS[@]}"
