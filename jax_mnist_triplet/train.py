from datetime import datetime
import os

# Force JAX to use CPU only for debugging
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import flax
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
import tqdm.auto as tqdm
import tensorflow as tf

from config import CHECKPOINT_DIR
from dataset import get_mnist_triplet_generator
from model import EmbeddingNet, train_step
from checkpoints import save_checkpoint

# Hyperparameters
BATCH_SIZE = 128 # @param {"type":"integer"}
LEARNING_RATE = 0.001 # @param {"type":"number"}
STEPS = 1000 # @param {"type":"integer"}
EMBEDDING_DIM = 3 # @param {"type":"number"}

print("JAX devices:", jax.devices())

# Initialize Random Key
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

# Initialize Model and Optimizer
model = EmbeddingNet(EMBEDDING_DIM)
dummy_input = jnp.ones([1, 28, 28, 1])
params = model.init(init_rng, dummy_input)['params']

tx = optax.adam(LEARNING_RATE)
state = train_state.TrainState.create(
  apply_fn=model.apply, params=params, tx=tx
)

train_ds = tfds.load('mnist', split='train', as_supervised=True)
data_gen = get_mnist_triplet_generator(train_ds, BATCH_SIZE)
train_generator = data_gen()

print('starting training...')

for step in tqdm.tqdm(range(STEPS)):
    anchors, positives, negatives = next(train_generator)

    state, loss = train_step(state, anchors, positives, negatives)

    if step % 100 == 0:
        tqdm.tqdm.write(f"Step {step}, Loss: {loss:.4f}")


save_checkpoint(state.params)

