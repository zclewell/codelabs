MNIST triplet embedding demo

Overview

- `train.py`: trains a small embedding network (triplet loss) on MNIST and saves model parameters as timestamped checkpoints.
- `checkpoints.py`: helper utilities to list, load, and save checkpoints.
- `visualize.py`: loads the most recent checkpoint (or you can call `checkpoints.load_model` yourself), computes embeddings for the MNIST test set, and produces a 3D Plotly visualization saved as `embeddings.html`.

Quick requirements

Create a virtual environment and install dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- The repo includes `requirements.txt` with the minimal dependencies used by these scripts.
- If you want to force JAX to CPU-only for debugging, either set the environment variable before running or use the existing code that sets it inside the scripts:
  - `export JAX_PLATFORM_NAME=cpu`
  - OR the code already sets `os.environ['JAX_PLATFORM_NAME']='cpu'` at the top of `train.py` and `visualize.py`.

Run training

```bash
python train.py
```

- This will run training and save a checkpoint via `checkpoints.save_checkpoint(...)` into the directory specified by `CHECKPOINT_DIR` in `config.py` (default `/tmp/checkpoints/`).
- Adjust `BATCH_SIZE`, `STEPS`, and `EMBEDDING_DIM` in `train.py` as desired.

Inspect/checkpoints

- To list checkpoints programmatically:

```python
from checkpoints import list_checkpoints
print(list_checkpoints())
```

- To load a specific checkpoint or the most-recent one:

```python
from checkpoints import load_model, load_most_recent_model
model, params = load_most_recent_model(embedding_dim=3)
# or
model, params = load_model(checkpoint_name='checkpoint_20250101120000.cpkt', embedding_dim=3)
```

Run visualization

```bash
python visualize.py
```

What `visualize.py` does:
- Loads the most recent checkpoint (via `checkpoints.load_most_recent_model`).
- Loads the MNIST `test` split from `tensorflow_datasets` and normalizes images to [0,1].
- Computes embeddings for all test examples using the loaded model.
- Produces an interactive Plotly 3D scatter saved as `embeddings.html` in the project directory.

Visualization details

- The visualization uses a 3D scatter (Plotly) with point color mapped to the digit label.
- Axes ranges are clamped to [-1, 1] and the 3D scene uses a square/cubic aspect ratio so visual distances are not distorted.
