"""
File processing utilities for loading and saving medical imaging data.

This module provides functions to:
- Load and save NIfTI files (.nii, .nii.gz)
- Load and save NumPy arrays (.npy, .npz)
- Create output directory structures
- Handle probability maps with class keys
- Extract and validate ZIP files

Example usage:
    from utils.file_processing import load_nifti, save_nifti, make_output_dirs
    
    # Load NIfTI file
    data, nii = load_nifti('volume.nii.gz')
    
    # Create output directory
    out_dir = make_output_dirs('/path/to/outputs', 'job_123')
    
    # Save segmentation with reference header
    save_nifti(segmentation, nii, f'{out_dir}/segmentation.nii.gz')
"""
import os
import logging
import zipfile
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional, Dict, List

# Configure logging
logger = logging.getLogger(__name__)


def load_nifti(file_path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load a NIfTI file and return data array with image object.
    
    Args:
        file_path: Path to NIfTI file (.nii or .nii.gz)
    
    Returns:
        Tuple of (data_array, nifti_image):
        - data_array: numpy array of shape (D, H, W) or (C, D, H, W), dtype float32
        - nifti_image: Nibabel Nifti1Image object with header and affine
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be loaded as NIfTI
    
    Example:
        >>> data, nii = load_nifti('brain.nii.gz')
        >>> print(f"Shape: {data.shape}, Affine shape: {nii.affine.shape}")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")
    
    try:
        nii = nib.load(str(file_path))
        data = nii.get_fdata().astype(np.float32)
        
        logger.info(f"Loaded NIfTI: {file_path.name}, shape: {data.shape}, dtype: {data.dtype}")
        
        return data, nii
    
    except Exception as e:
        raise ValueError(f"Failed to load NIfTI file {file_path}: {e}")


def save_nifti(
    arr: np.ndarray,
    reference_nii: nib.Nifti1Image,
    out_path: str
) -> str:
    """
    Save a numpy array as NIfTI file using reference header and affine.
    
    Args:
        arr: Array to save, shape (D, H, W) or (C, D, H, W)
        reference_nii: Reference NIfTI image to copy affine and header from
        out_path: Output file path (should end with .nii.gz)
    
    Returns:
        Absolute path to saved file
    
    Raises:
        ValueError: If array shape doesn't match reference spatial dimensions
    
    Example:
        >>> seg = np.random.randint(0, 4, (128, 128, 128), dtype=np.uint8)
        >>> save_nifti(seg, reference_nii, 'segmentation.nii.gz')
    """
    out_path = Path(out_path)
    
    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate array shape matches reference spatial dimensions
    ref_shape = reference_nii.shape[:3]  # (D, H, W)
    arr_spatial_shape = arr.shape[:3] if arr.ndim >= 3 else arr.shape
    
    if arr_spatial_shape != ref_shape:
        raise ValueError(
            f"Array spatial shape {arr_spatial_shape} doesn't match "
            f"reference shape {ref_shape}"
        )
    
    # Cast to uint8 for label arrays (assumes labels are 0-255)
    if arr.dtype != np.uint8:
        if arr.max() <= 255 and arr.min() >= 0:
            arr = arr.astype(np.uint8)
            logger.debug(f"Casting array to uint8 for NIfTI save")
        else:
            logger.warning(
                f"Array values outside [0, 255] range, saving as {arr.dtype}"
            )
    
    # Create NIfTI image with reference affine and header
    nii_img = nib.Nifti1Image(
        arr.astype(np.uint8) if arr.dtype != np.uint8 and arr.max() <= 255 else arr,
        reference_nii.affine,
        reference_nii.header
    )
    
    # Ensure .nii.gz extension
    if not str(out_path).endswith('.nii.gz'):
        if str(out_path).endswith('.nii'):
            out_path = Path(str(out_path) + '.gz')
        else:
            out_path = Path(str(out_path) + '.nii.gz')
    
    # Save
    nib.save(nii_img, str(out_path))
    
    abs_path = out_path.resolve()
    logger.info(f"Saved NIfTI: {out_path.name} to {abs_path}")
    
    return str(abs_path)


def load_numpy(path: str) -> np.ndarray:
    """
    Load a NumPy array from .npy file.
    
    Args:
        path: Path to .npy file
    
    Returns:
        Loaded numpy array
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be loaded
    
    Example:
        >>> arr = load_numpy('data.npy')
        >>> print(f"Shape: {arr.shape}, dtype: {arr.dtype}")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"NumPy file not found: {path}")
    
    try:
        arr = np.load(str(path))
        logger.info(f"Loaded NumPy: {path.name}, shape: {arr.shape}, dtype: {arr.dtype}")
        return arr
    
    except Exception as e:
        raise ValueError(f"Failed to load NumPy file {path}: {e}")


def save_numpy(arr: np.ndarray, out_path: str) -> str:
    """
    Save a numpy array to .npy file.
    
    Args:
        arr: Array to save
        out_path: Output file path (should end with .npy)
    
    Returns:
        Absolute path to saved file
    
    Example:
        >>> arr = np.random.rand(128, 128, 128)
        >>> save_numpy(arr, 'volume.npy')
    """
    out_path = Path(out_path)
    
    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure .npy extension
    if not str(out_path).endswith('.npy'):
        out_path = Path(str(out_path) + '.npy')
    
    # Save
    np.save(str(out_path), arr)
    
    abs_path = out_path.resolve()
    logger.info(f"Saved NumPy: {out_path.name} to {abs_path}, shape: {arr.shape}, dtype: {arr.dtype}")
    
    return str(abs_path)


def save_prob_maps(prob_maps: np.ndarray, out_path: str) -> str:
    """
    Save probability maps as compressed NumPy .npz file with class keys.
    
    Args:
        prob_maps: Probability maps array, shape (num_classes, D, H, W) or (D, H, W)
                   If 3D, assumes single class. If 4D, each channel is a class.
        out_path: Output file path (should end with .npz)
    
    Returns:
        Absolute path to saved file
    
    Raises:
        ValueError: If array shape is invalid
    
    Example:
        >>> probs = np.random.rand(4, 128, 128, 128).astype(np.float32)
        >>> save_prob_maps(probs, 'probabilities.npz')
        >>> # Creates file with keys: class_0, class_1, class_2, class_3
    """
    out_path = Path(out_path)
    
    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure .npz extension
    if not str(out_path).endswith('.npz'):
        out_path = Path(str(out_path) + '.npz')
    
    # Validate and prepare data
    if prob_maps.ndim == 3:
        # Single class, wrap in list
        num_classes = 1
        prob_dict = {'class_0': prob_maps}
    elif prob_maps.ndim == 4:
        # Multiple classes: (num_classes, D, H, W)
        num_classes = prob_maps.shape[0]
        prob_dict = {
            f'class_{i}': prob_maps[i] for i in range(num_classes)
        }
    else:
        raise ValueError(
            f"Expected 3D or 4D array, got shape {prob_maps.shape}"
        )
    
    # Save as compressed NumPy archive
    np.savez_compressed(str(out_path), **prob_dict)
    
    abs_path = out_path.resolve()
    logger.info(
        f"Saved probability maps: {out_path.name} to {abs_path}, "
        f"classes: {num_classes}, shape per class: {prob_maps.shape[-3:]}"
    )
    
    return str(abs_path)


def make_output_dirs(base_out: str, job_id: str) -> str:
    """
    Create output directory structure for a job.
    
    Args:
        base_out: Base output directory path (e.g., 'outputs' or '/path/to/outputs')
        job_id: Unique job identifier
    
    Returns:
        Absolute path to created job output directory
    
    Example:
        >>> out_dir = make_output_dirs('outputs', 'job_123')
        >>> # Creates: outputs/job_123/
        >>> # Returns: /absolute/path/to/outputs/job_123
    """
    base_out = Path(base_out)
    job_dir = base_out / job_id
    
    # Create directory (and parents if needed)
    job_dir.mkdir(parents=True, exist_ok=True)
    
    abs_path = job_dir.resolve()
    logger.info(f"Created output directory: {abs_path}")
    
    return str(abs_path)


def extract_zip(zip_path: str, extract_to: str) -> Dict[str, str]:
    """
    Extract a ZIP file and return paths to extracted files.
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract files to
    
    Returns:
        Dictionary mapping file type to extracted file path:
        {
            'flair': path,
            't1': path,
            't1ce': path,
            't2': path,
            'seg': path (optional),
            'boundary': path (optional),
            'distance': path (optional)
        }
    
    Raises:
        FileNotFoundError: If ZIP file doesn't exist
        ValueError: If ZIP file is invalid or required files are missing
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    
    # Create extraction directory
    extract_to.mkdir(parents=True, exist_ok=True)
    
    extracted_files = {}
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files in ZIP
            file_list = zip_ref.namelist()
            logger.info(f"ZIP contains {len(file_list)} files")
            
            # Extract all files
            zip_ref.extractall(extract_to)
            
            # Find NIfTI files (case-insensitive, flexible matching)
            nii_extensions = ['.nii', '.nii.gz']
            found_files = {}
            
            # Helper function to check if filename contains modality pattern
            def matches_modality(filename_lower: str, modality: str, extensions: list) -> bool:
                """Check if filename matches a modality pattern."""
                # Remove directory path, get just the filename
                base_name = Path(filename_lower).name
                
                # Patterns to match:
                # 1. Ends with: modality.ext or modality.ext
                # 2. Contains: _modality.ext or -modality.ext
                # 3. Starts with: modality_ or modality-
                # 4. Just the modality name (for files like "flair.nii")
                
                for ext in extensions:
                    # Direct match: flair.nii, flair.nii.gz
                    if base_name == f'{modality}{ext}' or base_name == f'{modality}.{ext}':
                        return True
                    # Underscore/hyphen patterns: patient_flair.nii, patient-flair.nii.gz
                    if f'_{modality}{ext}' in base_name or f'-{modality}{ext}' in base_name:
                        return True
                    if f'_{modality}.{ext}' in base_name or f'-{modality}.{ext}' in base_name:
                        return True
                    # Prefix patterns: flair_patient.nii, flair-patient.nii.gz
                    if base_name.startswith(f'{modality}_') or base_name.startswith(f'{modality}-'):
                        if base_name.endswith(ext):
                            return True
                
                return False
            
            for file_name in file_list:
                file_path = extract_to / file_name
                if not file_path.is_file():
                    continue
                
                file_lower = file_name.lower()
                
                # Check for required files
                for mod in ['flair', 't1', 't1ce', 't2']:
                    if mod not in found_files:
                        if matches_modality(file_lower, mod, nii_extensions):
                            found_files[mod] = str(file_path)
                            logger.info(f"Found {mod}: {file_path.name}")
                
                # Special handling for t1 vs t1ce (t1ce should take priority if both exist)
                if 't1ce' in found_files and 't1' in found_files:
                    # Check if the t1 file is actually t1ce
                    t1_path_lower = found_files['t1'].lower()
                    if 't1ce' in t1_path_lower and 't1ce' not in t1_path_lower.replace('t1ce', ''):
                        # The t1 file is actually t1ce, remove it
                        del found_files['t1']
                        logger.info("Removed t1 entry that was actually t1ce")
                
                # Check for optional files
                for opt in ['seg', 'boundary', 'distance']:
                    if opt not in found_files:
                        if matches_modality(file_lower, opt, nii_extensions):
                            found_files[opt] = str(file_path)
                            logger.info(f"Found {opt}: {file_path.name}")
            
            # Validate required files
            required = ['flair', 't1', 't1ce', 't2']
            missing = [r for r in required if r not in found_files]
            if missing:
                raise ValueError(f"Missing required files in ZIP: {', '.join(missing)}")
            
            extracted_files = found_files
            logger.info(f"Successfully extracted and validated {len(extracted_files)} files")
            
    except zipfile.BadZipFile:
        raise ValueError(f"Invalid ZIP file: {zip_path}")
    except Exception as e:
        raise ValueError(f"Failed to extract ZIP file: {e}")
    
    return extracted_files


def validate_nifti_files(file_paths: Dict[str, str]) -> Dict[str, Tuple[np.ndarray, nib.Nifti1Image]]:
    """
    Validate and load NIfTI files, ensuring they have matching spatial dimensions.
    
    Args:
        file_paths: Dictionary mapping file type to path
    
    Returns:
        Dictionary mapping file type to (data_array, nifti_image) tuple
    
    Raises:
        FileNotFoundError: If any file doesn't exist
        ValueError: If files have mismatched spatial dimensions
    """
    loaded = {}
    spatial_shapes = {}
    
    for file_type, file_path in file_paths.items():
        if file_path is None:
            continue
        
        data, nii = load_nifti(file_path)
        loaded[file_type] = (data, nii)
        spatial_shapes[file_type] = data.shape[:3] if data.ndim >= 3 else data.shape
    
    # Validate spatial dimensions match
    if len(spatial_shapes) > 1:
        shapes_list = list(spatial_shapes.values())
        if len(set(shapes_list)) > 1:
            raise ValueError(
                f"Spatial dimension mismatch: {spatial_shapes}"
            )
    
    logger.info(f"Validated {len(loaded)} NIfTI files with matching spatial dimensions")
    return loaded

