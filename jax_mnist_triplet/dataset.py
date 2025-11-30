from collections import defaultdict
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


def get_mnist_triplet_generator(ds, batch_size=64):
    """
    Loads MNIST and creates a generator that yields batches of
    (anchor, positive, negative) images.
    """

    # Preprocess: Normalize and group by label
    data_by_label = defaultdict(list)

    print("Grouping MNIST data by label...")
    for img, label in ds:
        img = tf.cast(img, tf.float32) / 255.0
        data_by_label[int(label)].append(img.numpy())

    # Convert lists to numpy arrays for fast indexing
    for label in data_by_label:
        data_by_label[label] = np.array(data_by_label[label])

    labels = list(data_by_label.keys())

    def generator():
        while True:
            anchors, positives, negatives = [], [], []

            for _ in range(batch_size):
                # 1. Select a random class for Anchor/Positive
                anchor_label = np.random.choice(labels)

                # 2. Select two random images from that class
                # (idx1 for anchor, idx2 for positive)
                idx1, idx2 = np.random.choice(
                    len(data_by_label[anchor_label]), 2, replace=False
                )

                # 3. Select a different class for Negative
                neg_label = np.random.choice([l for l in labels if l != anchor_label])

                # 4. Select one random image from negative class
                idx3 = np.random.choice(len(data_by_label[neg_label]))

                anchors.append(data_by_label[anchor_label][idx1])
                positives.append(data_by_label[anchor_label][idx2])
                negatives.append(data_by_label[neg_label][idx3])

            yield (
                jnp.array(anchors, dtype=np.float32),
                jnp.array(positives, dtype=np.float32),
                jnp.array(negatives, dtype=np.float32),
            )

    return generator


# --- 2. Model Architecture (Flax) ---
