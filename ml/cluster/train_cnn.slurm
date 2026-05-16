#!/bin/bash
#SBATCH --job-name=csi-cnn
#SBATCH --output=logs/csi_cnn_%j.out
#SBATCH --error=logs/csi_cnn_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# Edit for your school cluster (glas example):
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs models

module load python/3.11 2>/dev/null || true
# module load cuda/12.1 2>/dev/null || true

python3 -m venv .venv-train
source .venv-train/bin/activate
pip install --upgrade pip
pip install -r ml/requirements-train.txt

export PYTHONUNBUFFERED=1

echo "=== Dataset analysis ==="
python -m ml.csi.analyze --npz data/csi_windows.npz --csv rf_training_data.csv

echo "=== Training 1D-CNN + QAT ==="
python -u -m ml.csi.train \
  --data data/csi_windows.npz \
  --out-dir models/csi_cnn \
  --epochs 40 \
  --batch-size 64

echo "=== TFLite export ==="
python -m ml.csi.export_tflite --model-dir models/csi_cnn

echo "Done. Models: models/csi_cnn/zone_model_float.tflite (and zone_model_int8.tflite if supported)"
