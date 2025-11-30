
# Force JAX to use CPU only for debugging
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'

from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

# Included to resolve issues using jax and tf in the same script
print(jax.devices())

import tensorflow_datasets as tfds
import tqdm.auto as tqdm

from dataset import get_mnist_triplet_generator
from model import EmbeddingNet, train_step
from checkpoints import save_checkpoint

# Hyperparameters
BATCH_SIZE = 256  # @param {"type":"integer"}
LEARNING_RATE = 0.001  # @param {"type":"number"}
STEPS = 2000  # @param {"type":"integer"}
EMBEDDING_DIM = 3  # @param {"type":"number"}

print("JAX devices:", jax.devices())

# Initialize Random Key
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

# Initialize Model and Optimizer
model = EmbeddingNet(EMBEDDING_DIM)
dummy_input = jnp.ones([1, 28, 28, 1])
params = model.init(init_rng, dummy_input)["params"]

tx = optax.adam(LEARNING_RATE)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

train_ds = tfds.load("mnist", split="train", as_supervised=True)
data_gen = get_mnist_triplet_generator(train_ds, BATCH_SIZE)
train_generator = data_gen()

print("starting training...")

for step in tqdm.tqdm(range(STEPS)):
    anchors, positives, negatives = next(train_generator)

    state, loss = train_step(state, anchors, positives, negatives)

    if step % 100 == 0:
        tqdm.tqdm.write(f"Step {step}, Loss: {loss:.4f}")


save_checkpoint(state.params)
