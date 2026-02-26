"""
Visualization utilities for creating montages and overlays from 3D medical volumes.

This module provides functions to:
- Create montages with segmentation overlays from 3D volumes
- Generate class-specific overlay images
- Save RGB numpy arrays as PNG files

Example usage:
    import numpy as np
    from utils.visualization import create_montage, create_class_overlay, save_png
    
    # Load volume and segmentation
    volume = np.load('volume.npy')  # Shape: (D, H, W) or (C, D, H, W)
    seg_labels = np.load('segmentation.npy')  # Shape: (D, H, W)
    
    # Create montage with segmentation overlay
    montage = create_montage(volume, seg_labels, view='axial', n_slices=9)
    save_png(montage, 'montage_axial.png')
    
    # Create class overlay
    overlay = create_class_overlay(volume, seg_labels, class_id=1)
    save_png(overlay, 'overlay_class_1.png')
"""
import os
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


def color_map() -> dict:
    """
    Return color mapping for segmentation classes.
    
    Returns:
        Dictionary mapping class_id to RGB tuple:
        - 0: Background (black)
        - 1: Necrotic/Non-enhancing (red)
        - 2: Peritumoral Edema (green)
        - 3: Enhancing Tumor (blue)
    """
    return {
        0: (0, 0, 0),        # Background - black
        1: (255, 0, 0),     # Necrotic/Non-enhancing - red
        2: (0, 255, 0),     # Peritumoral Edema - green
        3: (0, 0, 255),     # Enhancing Tumor - blue
    }


def _normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D slice to uint8 range [0, 255] for display.
    
    Args:
        slice_data: 2D array (any dtype)
    
    Returns:
        uint8 array normalized to [0, 255]
    """
    slice_data = slice_data.astype(np.float32)
    
    # Handle edge cases
    arr_min = slice_data.min()
    arr_max = slice_data.max()
    
    if arr_max == arr_min:
        # Constant array, return zeros
        return np.zeros_like(slice_data, dtype=np.uint8)
    
    # Normalize to [0, 1] then scale to [0, 255]
    arr_norm = (slice_data - arr_min) / (arr_max - arr_min)
    arr_uint8 = (arr_norm * 255).astype(np.uint8)
    
    return arr_uint8


def create_montage(
    volume: np.ndarray,
    seg_labels: np.ndarray,
    view: str = 'axial',
    n_slices: int = 9,
    grid: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Create a montage with segmentation overlay from a 3D volume.
    
    Args:
        volume: 3D array of shape (D, H, W) or 4D array (C, D, H, W).
               If 4D, uses channel 0 (first channel).
        seg_labels: 3D array of shape (D, H, W) - segmentation labels (0-3)
        view: 'axial', 'coronal', or 'sagittal' - which view to extract
        n_slices: Number of slices to include in montage
        grid: Optional tuple (rows, cols) for grid layout. If None, auto-calculated.
    
    Returns:
        RGB numpy array (uint8) of the montage with segmentation overlay and legend
    
    Example:
        >>> volume = np.random.rand(128, 128, 128)
        >>> seg = np.random.randint(0, 4, (128, 128, 128))
        >>> montage = create_montage(volume, seg, view='axial', n_slices=9)
        >>> montage.shape  # (rows*H, cols*W, 3)
    """
    # Handle 4D volume (C, D, H, W) - take first channel
    if volume.ndim == 4:
        volume = volume[0, :, :, :]
        logger.debug(f"4D volume detected, using channel 0, shape: {volume.shape}")
    
    if volume.ndim != 3 or seg_labels.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays, got volume {volume.shape}, seg {seg_labels.shape}"
        )
    if volume.shape != seg_labels.shape:
        raise ValueError(
            f"Shape mismatch: volume {volume.shape} vs seg {seg_labels.shape}"
        )
    
    D, H, W = volume.shape
    
    # Select axis and determine slice dimension
    if view == 'axial':
        slice_dim = D
        get_volume_slice = lambda i: volume[i, :, :]
        get_seg_slice = lambda i: seg_labels[i, :, :]
    elif view == 'coronal':
        slice_dim = H
        get_volume_slice = lambda i: volume[:, i, :]
        get_seg_slice = lambda i: seg_labels[:, i, :]
    elif view == 'sagittal':
        slice_dim = W
        get_volume_slice = lambda i: volume[:, :, i]
        get_seg_slice = lambda i: seg_labels[:, :, i]
    else:
        raise ValueError(
            f"Invalid view '{view}'. Must be 'axial', 'coronal', or 'sagittal'"
        )
    
    # Select evenly spaced slice indices
    if n_slices >= slice_dim:
        indices = list(range(slice_dim))
    else:
        step = slice_dim / (n_slices + 1)
        indices = [int(step * (i + 1)) for i in range(n_slices)]
    
    # Auto-calculate grid if not provided
    if grid is None:
        n_cols = 3
        n_rows = (len(indices) + n_cols - 1) // n_cols
    else:
        n_rows, n_cols = grid
    
    # Extract slices and create overlays
    slice_images = []
    colors = color_map()
    alpha = 0.4
    
    for idx in indices:
        vol_slice = get_volume_slice(idx)
        seg_slice = get_seg_slice(idx)
        
        # Normalize volume slice
        vol_norm = _normalize_slice(vol_slice)
        
        # Create RGB image from grayscale
        rgb_slice = np.stack([vol_norm, vol_norm, vol_norm], axis=2).astype(np.uint8)
        
        # Overlay segmentation with colors for each class
        for class_id in [1, 2, 3]:  # Skip background (0)
            if class_id in colors:
                mask = (seg_slice == class_id).astype(np.float32)
                color = colors[class_id]
                for c in range(3):
                    rgb_slice[:, :, c] = (
                        (1 - alpha * mask) * rgb_slice[:, :, c] +
                        alpha * mask * color[c]
                    ).astype(np.uint8)
        
        slice_images.append(rgb_slice)
    
    # Get slice dimensions
    slice_h, slice_w, _ = slice_images[0].shape
    
    # Create montage canvas (RGB)
    montage_h = n_rows * slice_h
    montage_w = n_cols * slice_w
    montage = np.zeros((montage_h, montage_w, 3), dtype=np.uint8)
    
    # Place slices in grid
    for i, slice_img in enumerate(slice_images):
        row = i // n_cols
        col = i % n_cols
        y_start = row * slice_h
        y_end = y_start + slice_h
        x_start = col * slice_w
        x_end = x_start + slice_w
        montage[y_start:y_end, x_start:x_end, :] = slice_img
    
    # Add legend labels
    montage = _add_legend(montage, colors)
    
    return montage


def _add_legend(montage: np.ndarray, colors: dict) -> np.ndarray:
    """
    Add a small legend to the montage showing class colors.
    
    Args:
        montage: RGB numpy array
        colors: Color mapping dictionary
    
    Returns:
        RGB numpy array with legend added
    """
    # Convert to PIL Image for drawing
    img = Image.fromarray(montage, mode='RGB')
    draw = ImageDraw.Draw(img)
    
    # Legend position (top-right corner)
    legend_x = montage.shape[1] - 150
    legend_y = 10
    box_size = 15
    spacing = 20
    
    # Class labels
    labels = {
        1: "Necrotic",
        2: "Edema",
        3: "Enhancing"
    }
    
    # Draw legend boxes and labels
    for i, class_id in enumerate([1, 2, 3]):
        y_pos = legend_y + i * spacing
        color = colors.get(class_id, (255, 255, 255))
        label = labels.get(class_id, f"Class {class_id}")
        
        # Draw colored box
        draw.rectangle(
            [legend_x, y_pos, legend_x + box_size, y_pos + box_size],
            fill=color,
            outline=(255, 255, 255)
        )
        
        # Draw label text
        try:
            # Try to use default font
            font = ImageFont.load_default()
        except:
            font = None
        
        draw.text(
            (legend_x + box_size + 5, y_pos),
            label,
            fill=(255, 255, 255),
            font=font
        )
    
    # Convert back to numpy
    return np.array(img)


def create_class_overlay(
    volume: np.ndarray,
    seg_labels: np.ndarray,
    class_id: int,
    slice_index: Optional[int] = None
) -> np.ndarray:
    """
    Create a colored overlay image for a specific class from a 3D volume.
    
    Args:
        volume: 3D array of shape (D, H, W) or 4D array (C, D, H, W).
               If 4D, uses channel 0 (first channel).
        seg_labels: 3D array of shape (D, H, W) - segmentation labels (0-3)
        class_id: Class ID to overlay (1, 2, or 3)
        slice_index: Optional slice index. If None, picks center slice with the class.
    
    Returns:
        RGB numpy array (uint8) of a slice with class overlay
    
    Example:
        >>> volume = np.random.rand(128, 128, 128)
        >>> seg = np.random.randint(0, 4, (128, 128, 128))
        >>> overlay = create_class_overlay(volume, seg, class_id=1)
        >>> overlay.shape  # (H, W, 3)
    """
    # Handle 4D volume (C, D, H, W) - take first channel
    if volume.ndim == 4:
        volume = volume[0, :, :, :]
        logger.debug(f"4D volume detected, using channel 0, shape: {volume.shape}")
    
    if volume.ndim != 3 or seg_labels.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays, got volume {volume.shape}, seg {seg_labels.shape}"
        )
    if volume.shape != seg_labels.shape:
        raise ValueError(
            f"Shape mismatch: volume {volume.shape} vs seg {seg_labels.shape}"
        )
    
    D, H, W = seg_labels.shape
    
    # Determine slice index
    if slice_index is None:
        # Find a slice with this class (prefer middle of the class region)
        class_slices = np.any(seg_labels == class_id, axis=(1, 2))
        if np.any(class_slices):
            # Find middle slice with this class
            slice_indices = np.where(class_slices)[0]
            slice_index = slice_indices[len(slice_indices) // 2]
        else:
            # Fallback to middle slice
            slice_index = D // 2
            logger.warning(
                f"Class {class_id} not found in segmentation, using middle slice {slice_index}"
            )
    
    # Validate slice index
    if slice_index < 0 or slice_index >= D:
        raise ValueError(f"Slice index {slice_index} out of range [0, {D})")
    
    # Get axial slice
    volume_slice = volume[slice_index, :, :]
    seg_slice = seg_labels[slice_index, :, :]
    
    # Normalize volume slice
    vol_norm = _normalize_slice(volume_slice)
    
    # Create RGB image from grayscale
    rgb_image = np.stack([vol_norm, vol_norm, vol_norm], axis=2).astype(np.uint8)
    
    # Create mask for the specified class
    mask = (seg_slice == class_id).astype(np.float32)
    
    # Get color for this class
    colors = color_map()
    color = colors.get(class_id, (255, 0, 0))  # Default to red if class not in map
    alpha = 0.4
    
    # Apply overlay: blend color with original image where mask is 1
    for c in range(3):
        rgb_image[:, :, c] = (
            (1 - alpha * mask) * rgb_image[:, :, c] +
            alpha * mask * color[c]
        ).astype(np.uint8)
    
    return rgb_image


def save_png(img: np.ndarray, output_path: str) -> str:
    """
    Save a numpy RGB array as PNG file.
    
    Args:
        img: RGB numpy array of shape (H, W, 3) with dtype uint8
        output_path: Output file path (should end with .png)
    
    Returns:
        Output file path (same as input)
    
    Example:
        >>> img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        >>> path = save_png(img, 'output.png')
        >>> # [INFO] Saved output.png to /path/to/output.png
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(img)}")
    
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"Expected RGB array of shape (H, W, 3), got shape {img.shape}"
        )
    
    if img.dtype != np.uint8:
        logger.warning(f"Converting array from {img.dtype} to uint8")
        img = img.astype(np.uint8)
    
    # Ensure directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to PIL Image and save
    image = Image.fromarray(img, mode='RGB')
    image.save(str(output_path), 'PNG')
    
    # Log the save operation
    abs_path = output_path.resolve()
    filename = output_path.name
    logger.info(f"Saved {filename} to {abs_path}")
    
    return str(output_path)


# Backward compatibility functions (if needed by existing code)
def make_montage(
    volume: np.ndarray,
    n_slices: int = 9,
    axis: str = 'axial'
) -> Image.Image:
    """
    Legacy function for backward compatibility.
    Creates a montage without segmentation overlay.
    
    Args:
        volume: 3D array of shape (D, H, W)
        n_slices: Number of slices to include
        axis: 'axial', 'coronal', or 'sagittal'
    
    Returns:
        PIL Image (grayscale)
    """
    if volume.ndim == 4:
        volume = volume[0, :, :, :]
    
    D, H, W = volume.shape
    
    if axis == 'axial':
        slice_dim = D
        get_slice = lambda i: volume[i, :, :]
    elif axis == 'coronal':
        slice_dim = H
        get_slice = lambda i: volume[:, i, :]
    elif axis == 'sagittal':
        slice_dim = W
        get_slice = lambda i: volume[:, :, i]
    else:
        raise ValueError(f"Invalid axis '{axis}'")
    
    if n_slices >= slice_dim:
        indices = list(range(slice_dim))
    else:
        step = slice_dim / (n_slices + 1)
        indices = [int(step * (i + 1)) for i in range(n_slices)]
    
    slices = []
    for idx in indices:
        slice_data = get_slice(idx)
        slice_img = _normalize_slice(slice_data)
        slices.append(slice_img)
    
    n_cols = 3
    n_rows = (len(slices) + n_cols - 1) // n_cols
    
    slice_h, slice_w = slices[0].shape
    montage_h = n_rows * slice_h
    montage_w = n_cols * slice_w
    montage = np.zeros((montage_h, montage_w), dtype=np.uint8)
    
    for i, slice_img in enumerate(slices):
        row = i // n_cols
        col = i % n_cols
        y_start = row * slice_h
        y_end = y_start + slice_h
        x_start = col * slice_w
        x_end = x_start + slice_w
        montage[y_start:y_end, x_start:x_end] = slice_img
    
    return Image.fromarray(montage, mode='L')


def overlay_class_on_slice(
    volume_slice: np.ndarray,
    seg_slice: np.ndarray,
    class_id: int,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5
) -> Image.Image:
    """
    Legacy function for backward compatibility.
    Overlay a segmentation class on a volume slice.
    
    Args:
        volume_slice: 2D array (H, W)
        seg_slice: 2D array (H, W)
        class_id: Class ID to overlay
        color: RGB tuple for overlay color
        alpha: Transparency factor
    
    Returns:
        PIL Image in RGB mode
    """
    if volume_slice.ndim != 2 or seg_slice.ndim != 2:
        raise ValueError("Both volume_slice and seg_slice must be 2D arrays")
    if volume_slice.shape != seg_slice.shape:
        raise ValueError(f"Shape mismatch: volume {volume_slice.shape} vs seg {seg_slice.shape}")
    
    volume_norm = _normalize_slice(volume_slice)
    rgb_image = np.stack([volume_norm, volume_norm, volume_norm], axis=2).astype(np.uint8)
    
    mask = (seg_slice == class_id).astype(np.float32)
    
    for c in range(3):
        rgb_image[:, :, c] = (
            (1 - alpha * mask) * rgb_image[:, :, c] +
            alpha * mask * color[c]
        ).astype(np.uint8)
    
    return Image.fromarray(rgb_image, mode='RGB')


def create_comparison_overlay(
    mri_slice: np.ndarray,
    pred_slice: np.ndarray,
    gt_slice: np.ndarray,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Create a color-coded overlay comparing predicted segmentation with ground truth.
    
    Color coding:
    - True Positive (pred=1 AND gt=1): GREEN
    - False Positive (pred=1 AND gt=0): YELLOW
    - False Negative (pred=0 AND gt=1): RED
    
    Args:
        mri_slice: 2D array (H, W) - MRI image slice (preferably T1CE)
        pred_slice: 2D array (H, W) - Predicted segmentation (binary: 0 or 1)
        gt_slice: 2D array (H, W) - Ground truth segmentation (binary: 0 or 1)
        alpha: Transparency factor for overlay (0.0 to 1.0)
    
    Returns:
        RGB numpy array (uint8) of shape (H, W, 3) with color-coded overlay
    
    Example:
        >>> mri = np.random.rand(256, 256)
        >>> pred = (np.random.rand(256, 256) > 0.7).astype(np.uint8)
        >>> gt = (np.random.rand(256, 256) > 0.6).astype(np.uint8)
        >>> overlay = create_comparison_overlay(mri, pred, gt)
        >>> overlay.shape  # (256, 256, 3)
    """
    if mri_slice.ndim != 2 or pred_slice.ndim != 2 or gt_slice.ndim != 2:
        raise ValueError("All inputs must be 2D arrays")
    
    if mri_slice.shape != pred_slice.shape or mri_slice.shape != gt_slice.shape:
        raise ValueError(
            f"Shape mismatch: mri {mri_slice.shape}, pred {pred_slice.shape}, gt {gt_slice.shape}"
        )
    
    # Normalize MRI slice to uint8
    mri_norm = _normalize_slice(mri_slice)
    
    # Convert to RGB grayscale image
    rgb_image = np.stack([mri_norm, mri_norm, mri_norm], axis=2).astype(np.float32)
    
    # Binarize predictions and ground truth (handle multi-class by taking > 0)
    pred_binary = (pred_slice > 0).astype(np.float32)
    gt_binary = (gt_slice > 0).astype(np.float32)
    
    # Create masks for each category
    tp_mask = (pred_binary == 1) & (gt_binary == 1)  # True Positive
    fp_mask = (pred_binary == 1) & (gt_binary == 0)  # False Positive
    fn_mask = (pred_binary == 0) & (gt_binary == 1)  # False Negative
    
    # Color definitions (RGB)
    GREEN = np.array([0, 255, 0], dtype=np.float32)    # TP
    YELLOW = np.array([255, 255, 0], dtype=np.float32)  # FP
    RED = np.array([255, 0, 0], dtype=np.float32)       # FN
    
    # Apply overlays with alpha blending
    # Formula: result = (1 - alpha * mask) * base + alpha * mask * color
    
    # True Positives (Green)
    for c in range(3):
        rgb_image[:, :, c] = (
            (1 - alpha * tp_mask) * rgb_image[:, :, c] +
            alpha * tp_mask * GREEN[c]
        )
    
    # False Positives (Yellow)
    for c in range(3):
        rgb_image[:, :, c] = (
            (1 - alpha * fp_mask) * rgb_image[:, :, c] +
            alpha * fp_mask * YELLOW[c]
        )
    
    # False Negatives (Red)
    for c in range(3):
        rgb_image[:, :, c] = (
            (1 - alpha * fn_mask) * rgb_image[:, :, c] +
            alpha * fn_mask * RED[c]
        )
    
    # Clip and convert to uint8
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
    
    return rgb_image


def create_slice_comparison(
    mri_volume: np.ndarray,
    pred_volume: np.ndarray,
    gt_volume: np.ndarray,
    slice_index: Optional[int] = None,
    view: str = 'axial',
    alpha: float = 0.6
) -> np.ndarray:
    """
    Create a comparison overlay for a specific slice from 3D volumes.
    
    Args:
        mri_volume: 3D array (D, H, W) - MRI volume (preferably T1CE)
        pred_volume: 3D array (D, H, W) - Predicted segmentation
        gt_volume: 3D array (D, H, W) - Ground truth segmentation
        slice_index: Optional slice index. If None, uses middle slice.
        view: 'axial', 'coronal', or 'sagittal'
        alpha: Transparency factor for overlay
    
    Returns:
        RGB numpy array (uint8) of shape (H, W, 3) with color-coded overlay
    
    Example:
        >>> mri = np.random.rand(128, 256, 256)
        >>> pred = (np.random.rand(128, 256, 256) > 0.7).astype(np.uint8)
        >>> gt = (np.random.rand(128, 256, 256) > 0.6).astype(np.uint8)
        >>> overlay = create_slice_comparison(mri, pred, gt, view='axial')
    """
    if mri_volume.ndim != 3 or pred_volume.ndim != 3 or gt_volume.ndim != 3:
        raise ValueError("All volumes must be 3D arrays")
    
    if mri_volume.shape != pred_volume.shape or mri_volume.shape != gt_volume.shape:
        raise ValueError(
            f"Shape mismatch: mri {mri_volume.shape}, pred {pred_volume.shape}, gt {gt_volume.shape}"
        )
    
    D, H, W = mri_volume.shape
    
    # Determine slice index
    if slice_index is None:
        slice_index = D // 2  # Middle slice
    
    # Validate slice index
    if slice_index < 0 or slice_index >= D:
        raise ValueError(f"Slice index {slice_index} out of range [0, {D})")
    
    # Extract slice based on view
    if view == 'axial':
        mri_slice = mri_volume[slice_index, :, :]
        pred_slice = pred_volume[slice_index, :, :]
        gt_slice = gt_volume[slice_index, :, :]
    elif view == 'coronal':
        mri_slice = mri_volume[:, slice_index, :]
        pred_slice = pred_volume[:, slice_index, :]
        gt_slice = gt_volume[:, slice_index, :]
    elif view == 'sagittal':
        mri_slice = mri_volume[:, :, slice_index]
        pred_slice = pred_volume[:, :, slice_index]
        gt_slice = gt_volume[:, :, slice_index]
    else:
        raise ValueError(f"Invalid view '{view}'. Must be 'axial', 'coronal', or 'sagittal'")
    
    # Create comparison overlay
    return create_comparison_overlay(mri_slice, pred_slice, gt_slice, alpha=alpha)