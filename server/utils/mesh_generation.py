"""
3D mesh generation utilities for tumor visualization.

This module provides functions to:
- Extract binary masks from segmentation labels
- Generate 3D surface meshes using marching cubes
- Export meshes in STL and OBJ formats

Example usage:
    from utils.mesh_generation import generate_tumor_meshes
    
    meshes = generate_tumor_meshes(seg_data, output_dir="outputs/job_123")
    # Returns: {
    #   'necrotic': 'path/to/necrotic.stl',
    #   'edema': 'path/to/edema.stl',
    #   'enhancing': 'path/to/enhancing.stl'
    # }
"""
from skimage import morphology
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from skimage import measure
import trimesh

# Configure logging
logger = logging.getLogger(__name__)


def extract_tumor_masks(seg_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract binary masks for each tumor subregion from segmentation.
    
    Args:
        seg_data: 3D segmentation array with labels (0, 1, 2, 3, 4)
                  - Label 1: Necrotic Tumor (NET)
                  - Label 2: Edema (ED)
                  - Label 3: Enhancing Tumor (ET) - model output
                  - Label 4: Enhancing Tumor (ET) - ground truth format
    
    Returns:
        Dictionary with binary masks:
        {
            'necrotic': binary mask for label 1,
            'edema': binary mask for label 2,
            'enhancing': binary mask for label 3 or 4
        }
    """
    # Handle both label 3 (model output) and label 4 (ground truth format)
    enhancing_mask = ((seg_data == 3) | (seg_data == 4)).astype(np.uint8)
    
    masks = {
        'necrotic': (seg_data == 1).astype(np.uint8),
        'edema': (seg_data == 2).astype(np.uint8),
        'enhancing': enhancing_mask
    }
    
    # Log mask statistics
    for name, mask in masks.items():
        voxel_count = np.sum(mask)
        logger.info(f"{name.capitalize()} mask: {voxel_count} voxels")
    
    return masks


def generate_mesh_from_mask(
    mask: np.ndarray,
    spacing: Optional[Tuple[float, float, float]] = None,
    smoothing: bool = True
) -> Optional[trimesh.Trimesh]:
    """
    Generate a 3D surface mesh from a binary mask using marching cubes.
    
    Args:
        mask: 3D binary mask (0 or 1)
        spacing: Optional voxel spacing (D, H, W) in mm. If None, assumes 1mm isotropic.
        smoothing: Whether to apply Laplacian smoothing
    
    Returns:
        Trimesh object or None if mask is empty
    """
    if np.sum(mask) == 0:
        logger.warning("Empty mask, skipping mesh generation")
        return None
    
    # Default spacing: 1mm isotropic
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)
    
    # Scale coordinates by spacing
    # Note: skimage.measure.marching_cubes expects spacing in (z, y, x) order
    spacing_zyx = (spacing[0], spacing[1], spacing[2])
    
    try:
        # Generate mesh using marching cubes
        verts, faces, normals, values = measure.marching_cubes(
            mask.astype(float),
            level=0.5,
            spacing=spacing_zyx,
            allow_degenerate=False
        )
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        
        # Apply smoothing if requested
        if smoothing and len(mesh.vertices) > 0:
            try:
                mesh = mesh.smoothed()
                logger.debug("Applied Laplacian smoothing to mesh")
            except Exception as e:
                logger.warning(f"Smoothing failed: {e}, using unsmoothed mesh")
        
        # Remove degenerate faces
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        
        logger.info(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        return mesh
    
    except Exception as e:
        logger.error(f"Failed to generate mesh: {e}")
        return None


def export_mesh(mesh: trimesh.Trimesh, output_path: str, format: str = 'stl') -> str:
    """
    Export a mesh to file (STL or OBJ format).
    
    Args:
        mesh: Trimesh object
        output_path: Output file path
        format: 'stl' or 'obj'
    
    Returns:
        Absolute path to exported file
    
    Raises:
        ValueError: If format is not supported
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure correct extension
    if format.lower() == 'stl':
        if not str(output_path).endswith('.stl'):
            output_path = output_path.with_suffix('.stl')
        mesh.export(str(output_path))
    elif format.lower() == 'obj':
        if not str(output_path).endswith('.obj'):
            output_path = output_path.with_suffix('.obj')
        mesh.export(str(output_path))
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'stl' or 'obj'")
    
    abs_path = output_path.resolve()
    logger.info(f"Exported mesh to {abs_path} ({format.upper()})")
    
    return str(abs_path)


def generate_tumor_meshes(
    seg_data: np.ndarray,
    output_dir: str,
    spacing: Optional[Tuple[float, float, float]] = None,
    format: str = 'stl'
) -> Dict[str, Optional[str]]:
    """
    Generate 3D meshes for all tumor subregions.
    
    Args:
        seg_data: 3D segmentation array with labels (0, 1, 2, 4)
        output_dir: Directory to save mesh files
        spacing: Optional voxel spacing (D, H, W) in mm
        format: Export format ('stl' or 'obj')
    
    Returns:
        Dictionary mapping subregion names to file paths (None if mask is empty):
        {
            'necrotic': 'path/to/necrotic.stl' or None,
            'edema': 'path/to/edema.stl' or None,
            'enhancing': 'path/to/enhancing.stl' or None
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract masks
    masks = extract_tumor_masks(seg_data)
    
    # Generate meshes
    mesh_paths = {}
    
    for name, mask in masks.items():
        logger.info(f"Generating mesh for {name}...")
        
        mesh = generate_mesh_from_mask(mask, spacing=spacing)
        
        if mesh is not None:
            mesh_path = output_dir / f"{name}_tumor.{format}"
            exported_path = export_mesh(mesh, str(mesh_path), format=format)
            mesh_paths[name] = exported_path
        else:
            logger.warning(f"No mesh generated for {name} (empty mask)")
            mesh_paths[name] = None
    
    return mesh_paths
def generate_brain_mesh(
    mri_volume: np.ndarray,
    output_dir: str,
    spacing: Optional[Tuple[float, float, float]] = None,
    format: str = 'stl'
) -> Optional[str]:
    """
    Generate a coarse 3D brain surface mesh for spatial reference.

    Args:
        mri_volume: 3D MRI volume (T1 or T1ce)
        output_dir: Directory to save brain mesh
        spacing: Optional voxel spacing (D, H, W) in mm
        format: Export format ('stl' or 'obj')

    Returns:
        Path to exported brain mesh or None if generation fails
    """

    logger.info("Generating coarse brain mesh for spatial context...")

    # Normalize MRI intensities to [0, 1]
    vol = mri_volume.astype(np.float32)
    vol = vol - vol.min()
    vol = vol / (vol.max() + 1e-8)

    # High threshold to extract main brain mass (coarse mask)
    brain_mask = vol > 0.25

    # Morphological smoothing to remove holes and noise
    brain_mask = morphology.binary_closing(brain_mask, morphology.ball(2))
    brain_mask = morphology.binary_fill_holes(brain_mask)

    # Generate mesh using existing utility
    brain_mesh = generate_mesh_from_mask(
        brain_mask.astype(np.uint8),
        spacing=spacing,
        smoothing=True
    )

    if brain_mesh is None:
        logger.warning("Brain mesh generation failed")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = output_dir / f"brain_shell.{format}"
    exported_path = export_mesh(brain_mesh, str(mesh_path), format=format)

    logger.info(f"Brain mesh exported to {exported_path}")

    return exported_path