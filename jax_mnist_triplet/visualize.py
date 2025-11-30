import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from checkpoints import load_most_recent_model
import plotly.graph_objects as go
import plotly.express as px

# Load the most recent model
try:
    EMBEDDING_DIM = 3  # Match the training configuration
    model, params = load_most_recent_model(embedding_dim=EMBEDDING_DIM)
    print("\nModel parameters loaded successfully")
except ValueError as e:
    print(f"Error: {e}")
    exit(1)

# Load test set from MNIST
print("\nLoading MNIST test set...")
test_ds = tfds.load("mnist", split="test", as_supervised=True)

# Collect all test images and labels
test_images = []
test_labels = []

for img, label in test_ds:
    img = tf.cast(img, tf.float32) / 255.0
    test_images.append(img.numpy())
    test_labels.append(int(label))

test_images = jnp.array(test_images)
test_labels = jnp.array(test_labels)
print(f"Loaded {len(test_images)} test examples")

# Get predicted embeddings for each test example
print("\nComputing embeddings for test set...")
embeddings = model.apply({"params": params}, test_images)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Sample embedding (first test example): {embeddings[0]}")

# Visualize embeddings in 3D and save to local directory
try:
    emb_np = np.array(embeddings)
    if emb_np.shape[1] != 3:
        raise ValueError(f"Expected embeddings with dimension 3, got {emb_np.shape[1]}")

    x = emb_np[:, 0]
    y = emb_np[:, 1]
    z = emb_np[:, 2]

    # Use a discrete qualitative palette for labels (one color per digit)
    palette = px.colors.qualitative.Plotly
    labels_np = np.array(test_labels).astype(int)
    colors = [palette[int(l) % len(palette)] for l in labels_np]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=2,
                    color=colors,
                    opacity=0.8,
                ),
                text=[f"label: {int(l)}" for l in labels_np],
            )
        ]
    )

    fig.update_layout(
        title="MNIST Test Embeddings",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )

    out_path = "embeddings.html"
    fig.write_html(out_path)
    print(f"Saved embeddings visualization to {out_path}")
except Exception as e:
    print(f"Failed to create/save embedding visualization: {e}")
