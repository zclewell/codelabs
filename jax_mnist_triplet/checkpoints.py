import os
import flax
from datetime import datetime
from config import CHECKPOINT_DIR
from model import EmbeddingNet


def list_checkpoints():
    """
    List all checkpoint files in CHECKPOINT_DIR.
    
    Returns:
        list: Sorted list of checkpoint filenames
    """
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Checkpoint directory does not exist: {CHECKPOINT_DIR}")
        return []
    
    checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.cpkt')])
    return checkpoints


def load_model(checkpoint_name=None, embedding_dim=3):
    """
    Load an EmbeddingNet model with parameters from a checkpoint.
    
    Args:
        checkpoint_name (str, optional): Name of the checkpoint file to load.
                                        If None, loads the most recent checkpoint.
        embedding_dim (int): Embedding dimension for the model. Default is 3.
    
    Returns:
        tuple: (model, params) where model is EmbeddingNet and params are the loaded parameters
    
    Raises:
        ValueError: If no checkpoints are found or specified checkpoint doesn't exist
    """
    checkpoints = list_checkpoints()
    
    if not checkpoints:
        raise ValueError("No checkpoints found")
    
    if checkpoint_name is None:
        # Load most recent
        checkpoint_name = checkpoints[-1]
    elif checkpoint_name not in checkpoints:
        raise ValueError(f"Checkpoint '{checkpoint_name}' not found. Available: {checkpoints}")
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    
    print(f"Loading checkpoint: {checkpoint_name}")
    with open(checkpoint_path, 'rb') as f:
        params = flax.serialization.from_bytes(None, f.read())
    
    model = EmbeddingNet(embedding_dim)
    print(f"EmbeddingNet created with embedding_dim={embedding_dim}")
    
    return model, params


def load_most_recent_model(embedding_dim=3):
    """
    Load the most recent checkpoint as an EmbeddingNet model.
    
    Args:
        embedding_dim (int): Embedding dimension for the model. Default is 3.
    
    Returns:
        tuple: (model, params)
    """
    return load_model(checkpoint_name=None, embedding_dim=embedding_dim)


def save_checkpoint(params):
    """
    Save model parameters to a checkpoint file with a timestamp.
    
    Args:
        params: The model parameters to save (typically state.params)
    
    Returns:
        str: The path to the saved checkpoint file
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d%H%M%S")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{formatted_time}.cpkt')
    
    with open(checkpoint_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path
