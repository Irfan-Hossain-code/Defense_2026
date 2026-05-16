# CSI 1D-CNN — training guide (2 ESP32s)

## Dataset

| File | Shape | Use |
|------|-------|-----|
| `data/csi_windows.npz` | `(N, 20, 64, 2)` | **CNN training** |
| `rf_raw_csi_data.csv` | per-packet amps | rebuild NPZ via `windows.csv_to_npz` |
| `rf_training_data.csv` | 3 ratio features | RandomForest only |

Fetch dataset from branch `new_model`:

```bash
git fetch origin new_model
git show origin/new_model:data/csi_windows.npz > data/csi_windows.npz
```

## Pipeline

```bash
pip install -r ml/requirements-train.txt

# Inspect
python -m ml.csi.analyze --npz data/csi_windows.npz

# Train (~5–15 min GPU, longer on CPU)
python -u -m ml.csi.train --data data/csi_windows.npz --size large   # ~10 min CPU
python -m ml.csi.train --data data/csi_windows.npz --size small --epochs 15   # fast test

# INT8 TFLite for ESP32
python -m ml.csi.export_tflite --model-dir models/csi_cnn

# Run live with CNN
python main.py --model cnn
```

## Collect more data

```bash
python -m ml.csi.collect --per-zone 300 --left COM10 --right COM8
```

## Cluster (glas / Triton) — **must use GPU**

Login node `glass` has **no GPU**. Training there always uses CPU.

```bash
# Check GPU partitions on your cluster
sinfo -o "%P %G %a"

mkdir -p logs
sbatch ml/cluster/train_cnn.slurm
tail -f logs/csi_cnn_*.out
```

Interactive GPU shell (if sbatch partition name differs, fix `--partition`):

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=0:45:00 --pty bash
module load cuda    # if needed: module avail cuda
nvidia-smi
cd ~/Defense_2026 && source .venv-train/bin/activate
pip install "tensorflow[and-cuda]>=2.15,<2.20" tqdm
python -u -m ml.csi.train --data data/csi_windows.npz --size large --require-gpu
```

## Rebuild NPZ from raw CSV

```python
from ml.csi.windows import csv_to_npz
csv_to_npz("rf_raw_csi_data.csv", "data/csi_windows.npz")
```
