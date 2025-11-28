import jax
import jax.numpy as jnp
from flax import linen as nn

class EmbeddingNet(nn.Module):
    embedding_dim: int

    @nn.compact
    def __call__(self, x):
        # Convolutional Block 1
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Convolutional Block 2
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten and Project
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.embedding_dim)(x)

        # L2 Normalization (Crucial for Triplet Loss)
        # Keeps embeddings on a hypersphere
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x

# --- 3. Loss & Update Steps ---

def triplet_loss_fn(anchor_emb, positive_emb, negative_emb, margin=0.2):
    # Squared Euclidean Distance
    pos_dist = jnp.sum((anchor_emb - positive_emb)**2, axis=-1)
    neg_dist = jnp.sum((anchor_emb - negative_emb)**2, axis=-1)

    # Max(d(a,p) - d(a,n) + margin, 0)
    loss = jnp.maximum(pos_dist - neg_dist + margin, 0.0)
    return jnp.mean(loss)

@jax.jit
def train_step(state, anchor_img, positive_img, negative_img):
    def loss_fn(params):
        # Forward pass for all three heads (Siamese Network style)
        a_emb = state.apply_fn({'params': params}, anchor_img)
        p_emb = state.apply_fn({'params': params}, positive_img)
        n_emb = state.apply_fn({'params': params}, negative_img)

        loss = triplet_loss_fn(a_emb, p_emb, n_emb)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
