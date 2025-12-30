#!/bin/bash
#SBATCH --job-name=alg-dir
#SBATCH --output=logs/alg-dir-%j.out
#SBATCH --error=logs/alg-dir-%j.err

#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=23:00:00

set -euo pipefail

# Args:
#   1: INPUT_DIR   (default "graphs/graphs_ros")
#   2: OUTPUT_DIR  (default "res_dir")
#   3: ALG         (default "ros")  e.g., ros, gw, md, gp, bqp, genetic, ros_vanilla, pignn, ANYCSP
#   4: K_VALUE     (default 3)
#   5: SEED        (default 42)
#
# The rest are optional and defaulted:
#   6: LR          (default 1e-2)
#   7: WEIGHT_DECAY(default 1e-4)
#   8: EPOCHS      (default 10000)
#   9: PATIENCE    (default 100)
#  10: TOL         (default 1e-2)
#  11: SAVE_FLAG   (default 1)
#
# Example:
#   sbatch experiments/alg_dir.sh graphs/graphs_ros res_dir ros 3 42
#   sbatch experiments/alg_dir.sh graphs/graphs_ros res_dir ANYCSP 3 42

INPUT_DIR=${1:-"graphs/graphs_ros"}
OUTPUT_DIR=${2:-"res_dir"}
ALG=${3:-"ros"}
K_VALUE=${4:-3}
SEED=${5:-42}

LR=${6:-1e-2}
WEIGHT_DECAY=${7:-1e-4}
EPOCHS=${8:-10000}
PATIENCE=${9:-100}
TOL=${10:-1e-2}
SAVE_FLAG=${11:-1}

echo "Job started on $(hostname)"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "Input dir:  $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "alg: $ALG | k: $K_VALUE | seed: $SEED"
echo "lr: $LR | wd: $WEIGHT_DECAY | epochs: $EPOCHS | patience: $PATIENCE | tol: $TOL | save: $SAVE_FLAG"

# Threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Clean environment per cluster policy
module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0

cd /home/bs82/ROS
mkdir -p logs

set +eu
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./ros_env
set -eu

# ------------------------------------------------------------
# Minimal change: if ALG is ANYCSP, pass hardcoded model_dir
# (directory must contain config.json, e.g. anycsp/models/MAX3CUT/config.json)
# ------------------------------------------------------------
EXTRA_ARGS=""
if [[ "${ALG^^}" == "ANYCSP" ]]; then
  EXTRA_ARGS="--model_dir anycsp/models/MAX3CUT"
  echo "ANYCSP detected. Using model_dir: anycsp/models/MAX3CUT"
fi

python -u main_dir.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --q_prefix "Q" \
  --q_ext ".npy" \
  --alg "$ALG" \
  --graph_type load \
  --k "$K_VALUE" \
  --seed "$SEED" \
  --wd "$WEIGHT_DECAY" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --tol "$TOL" \
  --save "$SAVE_FLAG" \
  $EXTRA_ARGS

echo "Job complete."
