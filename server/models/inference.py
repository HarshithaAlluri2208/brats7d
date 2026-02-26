"""
Inference utilities for preprocessing, model execution, and postprocessing.

This module provides functions to:
- Preprocess medical imaging volumes for model input
- Run model inference
- Postprocess predictions to compute metrics
"""
import logging
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from typing import Tuple, Dict, Optional, Union
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


def preprocess_volume(
    volumes: Dict[str, Union[str, np.ndarray]]
) -> Tuple[torch.Tensor, Optional[nib.Nifti1Image], np.ndarray, Dict]:
    """
    Preprocess volumes for model inference.
    
    Accepts flair, t1, t1ce, t2 as NIfTI paths or numpy arrays.
    Optional dist and boundary as .npy paths or arrays.
    Stacks in order [flair, t1, t1ce, t2, dist, boundary].
    
    Args:
        volumes: Dictionary with keys:
            - 'flair': Path to NIfTI file or numpy array
            - 't1': Path to NIfTI file or numpy array
            - 't1ce': Path to NIfTI file or numpy array
            - 't2': Path to NIfTI file or numpy array
            - 'dist': Optional path to .npy file or numpy array
            - 'boundary': Optional path to .npy file or numpy array
    
    Returns:
        Tuple of:
        - input_tensor: torch.Tensor of shape (1, 6, D', H', W') on CPU
        - reference_nii: Nibabel Nifti1Image or None (from first NIfTI file)
        - original_volume_np: numpy array of shape (4, D, H, W) for visualization (MRI modalities only)
        - pad_info: Dictionary with padding information for unpadding
    
    Raises:
        ValueError: If required volumes are missing or shapes don't match
        FileNotFoundError: If file paths don't exist
    """
    from utils.file_processing import load_nifti, load_numpy
    
    required_keys = ['flair', 't1', 't1ce', 't2']
    for key in required_keys:
        if key not in volumes:
            raise ValueError(f"Missing required volume: {key}")
    
    # Load volumes
    volume_arrays = []
    reference_nii = None
    
    for key in ['flair', 't1', 't1ce', 't2']:
        vol = volumes[key]
        
        if isinstance(vol, str):
            # Load from file
            if vol.endswith('.npy'):
                data = load_numpy(vol)
                if reference_nii is None:
                    logger.warning(f"No reference NIfTI found yet, using {key} as reference")
            else:
                data, nii = load_nifti(vol)
                if reference_nii is None:
                    reference_nii = nii
                    logger.info(f"Using {key} as reference NIfTI for header/affine")
        elif isinstance(vol, np.ndarray):
            data = vol.astype(np.float32)
        else:
            raise ValueError(f"Invalid type for {key}: {type(vol)}")
        
        volume_arrays.append(data)
    
    # Validate all volumes have same spatial shape
    spatial_shapes = [arr.shape for arr in volume_arrays]
    if len(set(spatial_shapes)) > 1:
        raise ValueError(
            f"Spatial shape mismatch: {dict(zip(['flair', 't1', 't1ce', 't2'], spatial_shapes))}"
        )
    
    # Load optional dist and boundary
    for key in ['dist', 'boundary']:
        if key in volumes and volumes[key] is not None:
            vol = volumes[key]
            
            if isinstance(vol, str):
                data = load_numpy(vol)
            elif isinstance(vol, np.ndarray):
                data = vol.astype(np.float32)
            else:
                raise ValueError(f"Invalid type for {key}: {type(vol)}")
            
            # Validate shape matches
            if data.shape != volume_arrays[0].shape:
                logger.warning(
                    f"{key} shape {data.shape} doesn't match reference {volume_arrays[0].shape}, "
                    f"creating zeros array"
                )
                data = np.zeros_like(volume_arrays[0])
            
            volume_arrays.append(data)
        else:
            # Create zeros array if not provided
            volume_arrays.append(np.zeros_like(volume_arrays[0]))
            logger.debug(f"{key} not provided, using zeros")
    
    # Stack channels: (6, D, H, W)
    stack = np.stack(volume_arrays, axis=0).astype(np.float32)
    original_shape = stack.shape[1:]  # (D, H, W)
    logger.info(f"Stacked volume shape: {stack.shape}")
    
    # Z-score normalize first 4 channels (MRI modalities), keep dist/boundary as-is
    for c in range(4):
        channel = stack[c].copy()
        nonzero = channel[channel > 0]
        if nonzero.size > 0:
            mean = nonzero.mean()
            std = nonzero.std()
            if std > 0:
                stack[c] = (channel - mean) / std
                logger.debug(f"Normalized channel {c}: mean={mean:.2f}, std={std:.2f}")
        # If all zeros, leave as-is
    
    # Pad spatial dimensions to be divisible by 16
    D, H, W = stack.shape[1:]
    pad_d = (16 - D % 16) % 16
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    
    pad_info = {
        'original_shape': original_shape,
        'padded_shape': (D + pad_d, H + pad_h, W + pad_w),
        'pad_before': (0, 0, 0),
        'pad_after': (pad_d, pad_h, pad_w)
    }
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        # Pad only spatial dimensions: ((0,0), (0,pad_d), (0,pad_h), (0,pad_w))
        stack = np.pad(
            stack,
            ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=0
        )
        logger.info(f"Padded to shape: {stack.shape} (original: {original_shape})")
    
    # Convert to torch tensor and add batch dimension: (1, 6, D', H', W')
    input_tensor = torch.from_numpy(stack).unsqueeze(0)
    logger.info(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    
    # Store original volume for visualization (before padding)
    original_volume_np = np.stack(volume_arrays[:4], axis=0)  # (4, D, H, W) - just MRI modalities
    
    return input_tensor, reference_nii, original_volume_np, pad_info


def run_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: str,
    pad_info: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model inference and return predictions.
    
    Args:
        model: Loaded model in eval mode
        input_tensor: Preprocessed tensor of shape (1, 6, D, H, W) on CPU
        device: Device string ('cuda' or 'cpu')
        pad_info: Optional padding info dictionary for unpadding
    
    Returns:
        Tuple of:
        - pred_labels_np: numpy array uint8, shape (D, H, W) with class labels 0-3 (unpadded)
        - prob_maps: numpy array float32, shape (4, D, H, W) with class probabilities (unpadded)
    
    Raises:
        ValueError: If input tensor shape is invalid
    """
    if input_tensor.dim() != 5:
        raise ValueError(f"Expected 5D tensor (1, 6, D, H, W), got shape {input_tensor.shape}")
    if input_tensor.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {input_tensor.shape[0]}")
    if input_tensor.shape[1] != 6:
        raise ValueError(f"Expected 6 channels, got {input_tensor.shape[1]}")
    
    # Move tensor to device
    input_tensor = input_tensor.to(device)
    logger.debug(f"Moved input tensor to device: {device}")
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)  # Shape: (1, 4, D, H, W)
    
    # Get probabilities (softmax)
    probs = F.softmax(outputs, dim=1)  # Shape: (1, 4, D, H, W)
    
    # Get predicted labels (argmax)
    pred = torch.argmax(outputs, dim=1)  # Shape: (1, D, H, W)
    
    # Convert to numpy and remove batch dimension
    pred_np = pred.cpu().numpy()[0].astype(np.uint8)  # (D, H, W)
    probs_np = probs.cpu().numpy()[0].astype(np.float32)  # (4, D, H, W)
    
    logger.info(f"Prediction shape (before unpad): {pred_np.shape}, Probs shape (before unpad): {probs_np.shape}")
    
    # Unpad to original shape if pad_info provided
    if pad_info is not None:
        pred_np = unpad(pred_np, pad_info)
        # Unpad each class probability map
        probs_unpadded = []
        for c in range(probs_np.shape[0]):
            probs_unpadded.append(unpad(probs_np[c], pad_info))
        probs_np = np.stack(probs_unpadded, axis=0)
        logger.info(f"Unpadded to original shape: {pred_np.shape}, Probs: {probs_np.shape}")
    
    logger.info(f"Label range: [{pred_np.min()}, {pred_np.max()}], unique: {np.unique(pred_np)}")
    
    return pred_np, probs_np


def unpad(arr: np.ndarray, pad_info: Dict) -> np.ndarray:
    """
    Remove padding from an array using pad_info.
    
    Args:
        arr: Padded array
        pad_info: Dictionary with 'original_shape' and 'pad_after' keys
    
    Returns:
        Unpadded array with original shape
    """
    original_shape = pad_info['original_shape']
    pad_after = pad_info['pad_after']
    
    if pad_after == (0, 0, 0):
        return arr
    
    # Slice to remove padding: arr[:D, :H, :W]
    slices = tuple(slice(0, orig) for orig in original_shape)
    return arr[slices]


def postprocess(
    pred_labels: np.ndarray,
    prob_maps: np.ndarray,
    reference_nii: Optional[nib.Nifti1Image] = None
) -> Dict[str, float]:
    """
    Postprocess predictions to compute volume metrics.
    
    Args:
        pred_labels: Prediction array (D, H, W) with labels 0-3
        prob_maps: Probability maps (4, D, H, W) or (D, H, W)
        reference_nii: Optional reference NIfTI image for voxel spacing
    
    Returns:
        Dictionary with metrics:
        - 'wt_volume_cc': Whole Tumor volume in cubic centimeters
        - 'tc_volume_cc': Tumor Core volume in cubic centimeters
        - 'et_volume_cc': Enhancing Tumor volume in cubic centimeters
    
    Note:
        - WT (Whole Tumor) = labels 1 + 2 + 3
        - TC (Tumor Core) = labels 1 + 3
        - ET (Enhancing Tumor) = label 3
    """
    # Get voxel spacing from NIfTI header if available
    if reference_nii is not None:
        try:
            voxel_spacing = np.abs(reference_nii.header.get_zooms()[:3])  # (D, H, W)
            voxel_volume_ml = np.prod(voxel_spacing) / 1000.0  # Convert mm³ to ml (cc)
            logger.info(f"Voxel spacing: {voxel_spacing} mm, volume per voxel: {voxel_volume_ml:.6f} cc")
        except Exception as e:
            logger.warning(f"Failed to get voxel spacing from NIfTI header: {e}, using fallback")
            voxel_volume_ml = 0.001  # Default: 1 mm³ = 0.001 ml
    else:
        # Default BraTS spacing: 1mm x 1mm x 1mm
        voxel_volume_ml = 0.001  # 1 mm³ = 0.001 ml
        logger.info("No reference NIfTI provided, using default voxel volume: 0.001 cc")
    
    # Count voxels per class
    class_1_voxels = np.sum(pred_labels == 1)
    class_2_voxels = np.sum(pred_labels == 2)
    class_3_voxels = np.sum(pred_labels == 3)
    
    logger.debug(
        f"Voxel counts - Class 1: {class_1_voxels}, Class 2: {class_2_voxels}, "
        f"Class 3: {class_3_voxels}"
    )
    
    # Calculate volumes in cubic centimeters
    wt_volume_cc = (class_1_voxels + class_2_voxels + class_3_voxels) * voxel_volume_ml
    tc_volume_cc = (class_1_voxels + class_3_voxels) * voxel_volume_ml
    et_volume_cc = class_3_voxels * voxel_volume_ml
    
    metrics = {
        'wt_volume_cc': float(wt_volume_cc),
        'tc_volume_cc': float(tc_volume_cc),
        'et_volume_cc': float(et_volume_cc)
    }
    
    logger.info(
        f"Volume metrics - WT: {wt_volume_cc:.2f} cc, "
        f"TC: {tc_volume_cc:.2f} cc, ET: {et_volume_cc:.2f} cc"
    )
    
    return metrics

