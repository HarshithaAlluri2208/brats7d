"""
Inference utilities for preprocessing, model execution, and output saving.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from typing import Tuple, Dict, Optional, Any


def preprocess_volume(stack: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Preprocess 6-channel volume stack for model inference.
    
    Args:
        stack: Input array of shape (6, D, H, W) with channels [flair, t1, t1ce, t2, dist, boundary]
    
    Returns:
        Tuple of (preprocessed_tensor, pad_info_dict)
        - tensor: torch.Tensor of shape (1, 6, D', H', W') on CPU, ready for device transfer
        - pad_info: Dict with keys 'original_shape', 'padded_shape', 'pad_before', 'pad_after'
    """
    if stack.ndim != 4:
        raise ValueError(f"Expected 4D array (6, D, H, W), got shape {stack.shape}")
    if stack.shape[0] != 6:
        raise ValueError(f"Expected 6 channels, got {stack.shape[0]} channels")
    
    stack = stack.astype(np.float32)
    original_shape = stack.shape[1:]  # (D, H, W)
    
    # Z-score normalize first 4 channels (MRI modalities), leave last 2 as-is
    for c in range(4):
        channel = stack[c].copy()
        nonzero = channel[channel > 0]
        if nonzero.size > 0:
            mean = nonzero.mean()
            std = nonzero.std()
            if std > 0:
                stack[c] = (channel - mean) / std
        # If all zeros, leave as-is
    
    # Pad spatial dimensions to be divisible by 16
    D, H, W = stack.shape[1:]
    pad_d = (16 - D % 16) % 16
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    
    # np.pad format: ((pad_before_axis0, pad_after_axis0), (pad_before_axis1, pad_after_axis1), ...)
    # For shape (6, D, H, W): no pad on channels, pad spatial dims at the end
    pad_width = ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w))
    padded_stack = np.pad(stack, pad_width, mode='constant', constant_values=0)
    
    # Convert to tensor and add batch dimension: (1, 6, D', H', W')
    tensor = torch.from_numpy(padded_stack).unsqueeze(0).float()
    
    pad_info = {
        'original_shape': original_shape,
        'padded_shape': padded_stack.shape[1:],
        'pad_d': pad_d,
        'pad_h': pad_h,
        'pad_w': pad_w
    }
    
    return tensor, pad_info


def unpad(arr: np.ndarray, pad_info: Dict[str, Any]) -> np.ndarray:
    """
    Remove padding from array based on pad_info.
    
    Args:
        arr: Array to unpad, shape should match pad_info['padded_shape']
        pad_info: Dictionary from preprocess_volume() with padding information
    
    Returns:
        Unpadded array with shape matching pad_info['original_shape']
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (D, H, W), got shape {arr.shape}")
    
    original_shape = pad_info['original_shape']
    padded_shape = pad_info['padded_shape']
    
    if arr.shape != padded_shape:
        raise ValueError(
            f"Array shape {arr.shape} does not match padded_shape {padded_shape} from pad_info"
        )
    
    D_orig, H_orig, W_orig = original_shape
    
    # Slice to remove padding from the end (padding was added at the end)
    unpadded = arr[:D_orig, :H_orig, :W_orig]
    
    return unpadded


def run_model(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model inference and return predictions and probabilities.
    
    Args:
        model: Loaded model in eval mode
        input_tensor: Preprocessed tensor of shape (1, 6, D, H, W)
        device: Device string ('cuda' or 'cpu')
    
    Returns:
        Tuple of (pred_labels, prob_maps)
        - pred_labels: np.ndarray uint8, shape (D, H, W) with class labels 0-3
        - prob_maps: np.ndarray float32, shape (4, D, H, W) with class probabilities
    """
    if input_tensor.dim() != 5:
        raise ValueError(f"Expected 5D tensor (1, 6, D, H, W), got shape {input_tensor.shape}")
    if input_tensor.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {input_tensor.shape[0]}")
    if input_tensor.shape[1] != 6:
        raise ValueError(f"Expected 6 channels, got {input_tensor.shape[1]}")
    
    # Move tensor to device
    input_tensor = input_tensor.to(device)
    
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
    
    return pred_np, probs_np


def save_outputs(
    outdir: str,
    job_id: str,
    pred: np.ndarray,
    probs: np.ndarray,
    reference_nii: Optional[nib.Nifti1Image] = None
) -> Dict[str, str]:
    """
    Save segmentation predictions and probability maps to files.
    
    Args:
        outdir: Output directory path
        job_id: Unique job/session identifier for file naming
        pred: Prediction array, shape (D, H, W), dtype uint8
        probs: Probability maps, shape (4, D, H, W), dtype float32
        reference_nii: Optional reference NIfTI image for preserving affine/header
    
    Returns:
        Dictionary with keys and file paths:
        - 'segmentation_npy': path to .npy file
        - 'segmentation_nifti': path to .nii.gz file (if reference_nii provided)
        - 'probability_maps': path to .npz file
    """
    os.makedirs(outdir, exist_ok=True)
    
    saved_paths = {}
    
    # Save segmentation as .npy
    seg_npy_path = os.path.join(outdir, f"{job_id}_segmentation.npy")
    np.save(seg_npy_path, pred)
    saved_paths['segmentation_npy'] = seg_npy_path
    
    # Save segmentation as .nii.gz if reference provided
    if reference_nii is not None:
        seg_nii_path = os.path.join(outdir, f"{job_id}_segmentation.nii.gz")
        seg_nii = nib.Nifti1Image(
            pred.astype(np.uint8),
            reference_nii.affine,
            reference_nii.header
        )
        nib.save(seg_nii, seg_nii_path)
        saved_paths['segmentation_nifti'] = seg_nii_path
    else:
        saved_paths['segmentation_nifti'] = None
    
    # Save probability maps as .npz with keys class_0, class_1, class_2, class_3
    prob_npz_path = os.path.join(outdir, f"{job_id}_probabilities.npz")
    np.savez(
        prob_npz_path,
        class_0=probs[0],
        class_1=probs[1],
        class_2=probs[2],
        class_3=probs[3]
    )
    saved_paths['probability_maps'] = prob_npz_path
    
    return saved_paths

