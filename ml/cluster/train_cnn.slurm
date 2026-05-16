#!/bin/bash
#SBATCH --job-name=csi-cnn-gpu
#SBATCH --output=logs/csi_cnn_%j.out
#SBATCH --error=logs/csi_cnn_%j.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# GPU — required for ~15 min training (login nodes have NO GPU)
# If submit fails, run:  sinfo -o "%P %G"   and fix partition name below
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs models

echo "=== SLURM GPU job ==="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Load cluster CUDA (adjust names: module avail cuda)
module load python/3.12 2>/dev/null || module load python/3.11 2>/dev/null || true
module load cuda 2>/dev/null || module load cuda/12 2>/dev/null || module load cuda/12.1 2>/dev/null || true

if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L || { echo "ERROR: GPU not visible on this node. Use sbatch on a GPU partition."; exit 1; }
else
  echo "ERROR: nvidia-smi not found — you are NOT on a GPU node."
  echo "Do NOT run training on the login shell. Use:  sbatch ml/cluster/train_cnn.slurm"
  exit 1
fi

if [[ ! -d .venv-train ]]; then
  python3 -m venv .venv-train
fi
source .venv-train/bin/activate
pip install --upgrade pip
# Linux GPU wheels (bundled CUDA for TF 2.15+)
pip install "tensorflow[and-cuda]>=2.15,<2.20" tensorflow-model-optimization numpy scikit-learn tqdm pyserial

export PYTHONUNBUFFERED=1

python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print('TF GPUs:', gpus); assert gpus, 'No GPU for TensorFlow'"

echo "=== Dataset ==="
python -m ml.csi.analyze --npz data/csi_windows.npz

echo "=== Training (GPU) ==="
python -u -m ml.csi.train \
  --data data/csi_windows.npz \
  --out-dir models/csi_cnn \
  --size large \
  --epochs 80 \
  --batch-size 64 \
  --require-gpu

echo "=== TFLite export ==="
python -m ml.csi.export_tflite --model-dir models/csi_cnn || true

echo "Done. Push: git add -f models/csi_cnn && git commit && git push"
