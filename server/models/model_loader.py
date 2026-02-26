"""
Model loader for 3D-UNet segmentation model.
Handles checkpoint loading and model instantiation.
"""
import sys
import os
import logging
import torch
from pathlib import Path
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

# Module-level cache for singleton pattern
_cached_model = None
_cached_checkpoint_path = None
_cached_device = None


def get_unet_model():
    """
    Import and return the model factory function from src.model.py.
    
    Returns:
        get_unet_model function from src.model module
    
    Raises:
        ImportError: If model.py or get_unet_model function not found
    """
    # Add C:\Users\allur\brats7d_old\src to sys.path if not present
    src_path = Path("C:\\Users\\allur\\brats7d_old\\src").resolve()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        logger.debug(f"Added {src_path} to sys.path")
    
    # Import get_unet_model from src.model
    try:
        from model import get_unet_model
        logger.info(f"Successfully imported get_unet_model from {src_path}/model.py")
        return get_unet_model
    except ImportError as e:
        raise ImportError(
            f"Failed to import get_unet_model from src.model.py. "
            f"Make sure C:\\Users\\allur\\brats7d_old\\src\\model.py exists and contains get_unet_model function. "
            f"Original error: {e}"
        )


def load_checkpoint(checkpoint_path: str, device: str) -> torch.nn.Module:
    """
    Instantiate model and load checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint file (.pth)
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Loaded model in eval mode, moved to specified device
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Get model factory function
    get_unet_model_fn = get_unet_model()
    
    # Instantiate model with in_channels=6, out_channels=4
    logger.info("Instantiating model with in_channels=6, out_channels=4")
    model = get_unet_model_fn(in_channels=6, out_channels=4)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both checkpoint dict with 'model_state_dict' and raw state dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.debug("Loaded from checkpoint dict with 'model_state_dict' key")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            logger.debug("Loaded from checkpoint dict with 'state_dict' key")
        else:
            # Assume it's a raw state dict
            model.load_state_dict(checkpoint)
            logger.debug("Loaded from checkpoint dict (assumed raw state_dict)")
    else:
        # Assume it's a raw state dict
        model.load_state_dict(checkpoint)
        logger.debug("Loaded from raw state_dict")
    
    # Move model to device
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    # Set model to eval mode
    model.eval()
    logger.info("Model set to eval mode")
    
    return model


def get_cached_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    """
    Singleton wrapper that caches the loaded model.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on ('cuda' or 'cpu')
    
    Returns:
        Cached model instance
    """
    global _cached_model, _cached_checkpoint_path, _cached_device
    
    # If model is already cached and same checkpoint/device, return it
    if _cached_model is not None:
        if (checkpoint_path == _cached_checkpoint_path and 
            device == _cached_device):
            logger.debug("Returning cached model")
            return _cached_model
    
    # Load new model
    _cached_model = load_checkpoint(checkpoint_path, device)
    _cached_checkpoint_path = checkpoint_path
    _cached_device = device
    
    logger.info(f"Model cached for checkpoint: {checkpoint_path}, device: {device}")
    return _cached_model


if __name__ == "__main__":
    # Test loading
    checkpoint = "C:\\Users\\allur\\brats7d_old\\models\\checkpoint_epoch50.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_cached_model(checkpoint, device)
    print(f"Model loaded successfully on {device}")
    print(f"Model type: {type(model)}")

