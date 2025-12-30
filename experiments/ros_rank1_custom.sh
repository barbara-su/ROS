#!/bin/bash
#SBATCH --job-name=ros
#SBATCH --output=logs/ros-%j.out
#SBATCH --error=logs/ros-%j.err

#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=200G
#SBATCH --time=23:00:00

set -euo pipefail

GRAPH_PATH=${1:-"graphs/graphs_rank_1/Q_100.npy"}
K_VALUE=${2:-3}
SEED=${3:-42}
LR=${4:-1e-2}
WEIGHT_DECAY=${5:-1e-4}
EPOCHS=${6:-10000}
PATIENCE=${7:-100}
TOL=${8:-1e-2}
SAVE_FLAG=${9:-1}

echo "Job started on $(hostname)"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "Graph path: $GRAPH_PATH"
echo "k: $K_VALUE | seed: $SEED"
echo "lr: $LR | weight decay: $WEIGHT_DECAY"

# OpenBLAS config
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Clean environment per cluster policy.
module purge || true
unset LD_PRELOAD || true
export XALT_EXECUTABLE_TRACKING=0
export XALT_RUNNABLE=0

cd /home/bs82/ROS
mkdir -p logs res

set +eu
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./ros_env

python main.py \
  --graph_type load \
  --graph_path "$GRAPH_PATH" \
  --k "$K_VALUE" \
  --seed "$SEED" \
  --alg ros \
  --wd "$WEIGHT_DECAY" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --tol "$TOL" \
  --save "$SAVE_FLAG"

echo "Job complete."
set -eu