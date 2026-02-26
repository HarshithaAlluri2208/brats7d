"""
FastAPI application for 3D-UNet medical segmentation inference server.
"""
import os
import uuid
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
import shutil

from models.model_loader import get_cached_model
from models.inference import preprocess_volume, run_inference, postprocess
from utils.file_processing import (
    load_nifti, save_nifti, save_numpy, save_prob_maps, make_output_dirs,
    extract_zip, validate_nifti_files
)
from utils.visualization import (
    create_montage, create_class_overlay, save_png,
    create_slice_comparison
)
from utils.mesh_generation import generate_tumor_meshes, generate_brain_mesh

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="NeuroVision Inference Server", version="1.0.0")

# Configuration
CHECKPOINT_PATH = r"C:\Users\allur\brats7d_old\models\checkpoint_epoch50.pth"
TEMP_DIR = Path(r"C:\Users\allur\brats7d_old\server\temp")
OUTPUT_DIR = Path(r"C:\Users\allur\brats7d_old\server\outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_URL = "http://localhost:8000"

# Get CORS origin from environment or default
CORS_ORIGIN_ENV = os.getenv("CORS_ORIGIN") or os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:3000")
if "://" in CORS_ORIGIN_ENV:
    from urllib.parse import urlparse
    parsed = urlparse(CORS_ORIGIN_ENV)
    CORS_ORIGIN = f"{parsed.scheme}://{parsed.netloc}"
else:
    CORS_ORIGIN = CORS_ORIGIN_ENV

# Ensure localhost dev ports are always allowed for Next.js
CORS_ORIGINS = [CORS_ORIGIN, "http://localhost:3000", "http://localhost:3004", "http://localhost:3005"]
# Remove duplicates while preserving order
CORS_ORIGINS = list(dict.fromkeys(CORS_ORIGINS))

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "x-api-key"],
)

# API Key configuration (optional)
API_KEY = os.getenv("API_KEY", None)
API_KEY_ENABLED = API_KEY is not None

# Global model variable
model = None
model_loaded = False


def verify_api_key(request: Request):
    """Dependency to verify API key from request header."""
    if not API_KEY_ENABLED:
        return True
    
    api_key = request.headers.get("x-api-key")
    
    if not api_key:
        logger.warning("API key missing from request")
        raise HTTPException(
            status_code=401,
            detail="API key required. Please provide 'x-api-key' header."
        )
    
    if api_key != API_KEY:
        logger.warning(f"Invalid API key provided")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return True


def create_simple_montage(volume: np.ndarray, n_slices: int = 9, axis: str = 'axial') -> np.ndarray:
    """
    Fallback function to create a simple montage without segmentation overlay.
    """
    if volume.ndim == 4:
        volume = volume[0]  # Take first channel if 4D
    
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
        raise ValueError(f"Invalid axis: {axis}")
    
    # Select evenly spaced slices
    if n_slices >= slice_dim:
        indices = list(range(slice_dim))
    else:
        step = slice_dim / (n_slices + 1)
        indices = [int(step * (i + 1)) for i in range(n_slices)]
    
    # Normalize and create slices
    slices = []
    for idx in indices:
        slice_data = get_slice(idx)
        arr_min, arr_max = slice_data.min(), slice_data.max()
        if arr_max > arr_min:
            slice_norm = ((slice_data - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            slice_norm = np.zeros_like(slice_data, dtype=np.uint8)
        slices.append(slice_norm)
    
    # Arrange in grid
    n_cols = 3
    n_rows = (len(slices) + n_cols - 1) // n_cols
    slice_h, slice_w = slices[0].shape
    montage = np.zeros((n_rows * slice_h, n_cols * slice_w, 3), dtype=np.uint8)
    
    for i, slice_img in enumerate(slices):
        row = i // n_cols
        col = i % n_cols
        y_start = row * slice_h
        y_end = y_start + slice_h
        x_start = col * slice_w
        x_end = x_start + slice_w
        montage[y_start:y_end, x_start:x_end, :] = np.stack([slice_img, slice_img, slice_img], axis=2)
    
    return montage


@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    global model, model_loaded
    try:
        logger.info(f"Loading model from {CHECKPOINT_PATH} on device {DEVICE}")
        logger.info(f"CORS origin: {CORS_ORIGIN}")
        logger.info(f"API key authentication: {'ENABLED' if API_KEY_ENABLED else 'DISABLED'}")
        model = get_cached_model(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
        model_loaded = True
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        model_loaded = False
        # Don't raise - allow server to start but mark model as not loaded


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": DEVICE
    }


@app.get("/debug/list_outputs")
async def list_outputs():
    """List all files in the outputs directory for debugging."""
    try:
        output_files = []
        if OUTPUT_DIR.exists():
            for job_dir in OUTPUT_DIR.iterdir():
                if job_dir.is_dir():
                    job_files = []
                    for file_path in job_dir.rglob("*"):
                        if file_path.is_file():
                            rel_path = file_path.relative_to(OUTPUT_DIR)
                            job_files.append({
                                "path": str(rel_path),
                                "size": file_path.stat().st_size,
                                "name": file_path.name
                            })
                    if job_files:
                        output_files.append({
                            "job_id": job_dir.name,
                            "files": job_files,
                            "count": len(job_files)
                        })
        
        return {
            "output_dir": str(OUTPUT_DIR),
            "exists": OUTPUT_DIR.exists(),
            "jobs": output_files,
            "total_jobs": len(output_files)
        }
    except Exception as e:
        logger.error(f"Error listing outputs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing outputs: {str(e)}")


@app.post("/infer/upload-zip")
async def upload_zip(
    request: Request,
    zip_file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    _: bool = Depends(verify_api_key)
):
    """
    Upload a ZIP file containing NIfTI files and extract/validate them.
    
    Expected files in ZIP:
    - Required: flair.nii, t1.nii, t1ce.nii, t2.nii
    - Optional: seg.nii (ground truth), boundary.nii, distance.nii
    
    Returns:
        JSON with job_id and extracted file paths
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save ZIP file
        zip_path = job_dir / "upload.zip"
        with open(zip_path, "wb") as f:
            content = await zip_file.read()
            f.write(content)
        logger.info(f"Saved ZIP file to {zip_path}")
        
        # Extract and validate ZIP
        extracted_files = extract_zip(str(zip_path), str(job_dir))
        
        # Store patient_id if provided
        if patient_id:
            patient_id_path = job_dir / "patient_id.txt"
            with open(patient_id_path, "w") as f:
                f.write(patient_id)
            extracted_files["patient_id"] = str(patient_id_path)
        
        return {
            "job_id": job_id,
            "status": "extracted",
            "files": extracted_files
        }
    
    except Exception as e:
        logger.error(f"ZIP upload/extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ZIP processing failed: {str(e)}")


@app.post("/infer/upload")
async def upload_files(
    request: Request,
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    dist: Optional[UploadFile] = File(None),
    boundary: Optional[UploadFile] = File(None),
    patient_id: Optional[str] = Form(None),
    _: bool = Depends(verify_api_key)
):
    """
    Upload patient files and save to temporary directory.
    
    Returns:
        JSON with job_id and saved file paths
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        saved_paths = {}
        
        # Save required files with predictable names
        required_files = {
            "flair": flair,
            "t1": t1,
            "t1ce": t1ce,
            "t2": t2
        }
        
        for name, file in required_files.items():
            # Determine extension from original filename
            ext = Path(file.filename).suffix
            if not ext:
                ext = ".nii"  # Default to .nii
            
            file_path = job_dir / f"{name}{ext}"
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_paths[name] = str(file_path)
            print(f"[INFO] Saved {name}{ext} to {file_path.resolve()}")
        
        # Save optional files
        if dist:
            file_path = job_dir / "dist.npy"
            with open(file_path, "wb") as f:
                content = await dist.read()
                f.write(content)
            saved_paths["dist"] = str(file_path)
            print(f"[INFO] Saved dist.npy to {file_path.resolve()}")
        
        if boundary:
            file_path = job_dir / "boundary.npy"
            with open(file_path, "wb") as f:
                content = await boundary.read()
                f.write(content)
            saved_paths["boundary"] = str(file_path)
            print(f"[INFO] Saved boundary.npy to {file_path.resolve()}")
        
        # Store patient_id if provided
        if patient_id:
            patient_id_path = job_dir / "patient_id.txt"
            with open(patient_id_path, "w") as f:
                f.write(patient_id)
            saved_paths["patient_id"] = str(patient_id_path)
        
        return {
            "job_id": job_id,
            "status": "uploaded",
            "files": saved_paths
        }
    
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/infer/run")
async def run_inference_endpoint(
    request: Request,
    job_id: str = Form(...),
    _: bool = Depends(verify_api_key)
):
    """
    Run inference on uploaded files.
    
    Returns:
        JSON with download URLs and metrics
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    job_dir = TEMP_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Create output directory
    output_dir = make_output_dirs(str(OUTPUT_DIR), job_id)
    output_path = Path(output_dir)
    
    try:
        # Load files into volumes dict
        volumes = {}
        
        # Load required NIfTI files
        flair_path = job_dir / "flair.nii" if (job_dir / "flair.nii").exists() else next(job_dir.glob("flair.*"), None)
        t1_path = job_dir / "t1.nii" if (job_dir / "t1.nii").exists() else next(job_dir.glob("t1.*"), None)
        t1ce_path = job_dir / "t1ce.nii" if (job_dir / "t1ce.nii").exists() else next(job_dir.glob("t1ce.*"), None)
        t2_path = job_dir / "t2.nii" if (job_dir / "t2.nii").exists() else next(job_dir.glob("t2.*"), None)
        
        if not all([flair_path, t1_path, t1ce_path, t2_path]):
            raise HTTPException(status_code=400, detail="Missing required files (flair, t1, t1ce, t2)")
        
        volumes['flair'] = str(flair_path)
        volumes['t1'] = str(t1_path)
        volumes['t1ce'] = str(t1ce_path)
        volumes['t2'] = str(t2_path)
        
        # Load optional files (check both .npy and .nii formats)
        dist_path = job_dir / "dist.npy"
        if not dist_path.exists():
            # Check for distance.nii or distance.nii.gz
            dist_path = next(job_dir.glob("distance.*"), None)
        if dist_path and dist_path.exists():
            volumes['dist'] = str(dist_path)
        
        boundary_path = job_dir / "boundary.npy"
        if not boundary_path.exists():
            # Check for boundary.nii or boundary.nii.gz
            boundary_path = next(job_dir.glob("boundary.*"), None)
        if boundary_path and boundary_path.exists():
            volumes['boundary'] = str(boundary_path)
        
        # Check for ground truth segmentation
        seg_path = next(job_dir.glob("seg.*"), None)
        gt_volume = None
        if seg_path and seg_path.exists():
            try:
                gt_data, _ = load_nifti(str(seg_path))
                gt_volume = gt_data
                logger.info(f"Loaded ground truth segmentation from {seg_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load ground truth: {e}")
        
        # Preprocess
        logger.info(f"Preprocessing volumes for job {job_id}")
        input_tensor, reference_nii, original_volume_np, pad_info = preprocess_volume(volumes)
        
        # Run inference
        logger.info(f"Running inference for job {job_id}")
        pred_labels, prob_maps = run_inference(model, input_tensor, DEVICE, pad_info)
        
        # Postprocess to get metrics
        logger.info(f"Postprocessing predictions for job {job_id}")
        metrics = postprocess(pred_labels, prob_maps, reference_nii)
        
        # Save outputs
        logger.info(f"Saving outputs for job {job_id}")
        
        # Save segmentation
        seg_npy_path = save_numpy(pred_labels, str(output_path / f"{job_id}_segmentation.npy"))
        print(f"[INFO] Saved {job_id}_segmentation.npy to {Path(seg_npy_path).resolve()}")
        
        seg_nii_path = None
        if reference_nii is not None:
            try:
                seg_nii_path = save_nifti(pred_labels, reference_nii, str(output_path / f"{job_id}_segmentation.nii.gz"))
                print(f"[INFO] Saved {job_id}_segmentation.nii.gz to {Path(seg_nii_path).resolve()}")
            except Exception as e:
                logger.warning(f"Failed to save NIfTI segmentation: {e}")
        
        # Save probability maps
        prob_path = save_prob_maps(prob_maps, str(output_path / f"{job_id}_probabilities.npz"))
        print(f"[INFO] Saved {job_id}_probabilities.npz to {Path(prob_path).resolve()}")
        
        # Create visualizations
        logger.info(f"Creating visualizations for job {job_id}")
        base_url = f"{BASE_URL}/api/download"
        
        # Use first channel of original volume (flair) for visualization
        flair_volume = original_volume_np[0] if original_volume_np.ndim == 4 else original_volume_np
        # Use t1ce for comparison overlay (index 2 in [flair, t1, t1ce, t2])
        t1ce_volume = original_volume_np[2] if original_volume_np.ndim == 4 else original_volume_np
        
        # Montages
        montage_urls = {}
        for view in ['axial', 'coronal', 'sagittal']:
            try:
                montage_array = create_montage(flair_volume, pred_labels, view=view)
                montage_path = output_path / f"montage_{view}.png"
                save_png(montage_array, str(montage_path))
                montage_urls[view] = f"{base_url}/{job_id}/montage_{view}.png"
            except Exception as e:
                logger.warning(f"Failed to create montage {view}: {e}, using fallback")
                try:
                    montage_array = create_simple_montage(flair_volume, axis=view)
                    montage_path = output_path / f"montage_{view}.png"
                    save_png(montage_array, str(montage_path))
                    montage_urls[view] = f"{base_url}/{job_id}/montage_{view}.png"
                except Exception as e2:
                    logger.error(f"Fallback montage also failed: {e2}")
        
        # Class overlays
        overlay_urls = {}
        for class_id in [1, 2, 3]:
            try:
                overlay_array = create_class_overlay(flair_volume, pred_labels, class_id)
                overlay_path = output_path / f"overlay_class_{class_id}.png"
                save_png(overlay_array, str(overlay_path))
                overlay_urls[class_id] = f"{base_url}/{job_id}/overlay_class_{class_id}.png"
            except Exception as e:
                logger.warning(f"Failed to create overlay for class {class_id}: {e}")
        
        # Comparison overlay (if ground truth is available)
        comparison_overlay_url = None
        if gt_volume is not None:
            try:
                logger.info(f"Creating comparison overlay for job {job_id}")
                # Binarize predictions and ground truth for comparison (any non-zero = positive)
                pred_binary = (pred_labels > 0).astype(np.uint8)
                gt_binary = (gt_volume > 0).astype(np.uint8)
                
                # Create comparison overlay on middle axial slice
                comparison_array = create_slice_comparison(
                    t1ce_volume, pred_binary, gt_binary,
                    slice_index=None, view='axial', alpha=0.6
                )
                comparison_path = output_path / "comparison_overlay.png"
                save_png(comparison_array, str(comparison_path))
                comparison_overlay_url = f"{base_url}/{job_id}/comparison_overlay.png"
                logger.info(f"Saved comparison overlay to {comparison_path}")
            except Exception as e:
                logger.warning(f"Failed to create comparison overlay: {e}", exc_info=True)
        
        # Generate 3D meshes from segmentation
        # Use ground truth if available, otherwise use predictions
        seg_for_mesh = gt_volume if gt_volume is not None else pred_labels
        mesh_urls = {}
        
        if seg_for_mesh is not None:
            try:
                logger.info(f"Generating 3D meshes for job {job_id}")
                
                # Get voxel spacing from reference NIfTI
                spacing = None
                if reference_nii is not None:
                    try:
                        spacing = tuple(reference_nii.header.get_zooms()[:3])
                        logger.info(f"Using voxel spacing: {spacing} mm")
                    except Exception as e:
                        logger.warning(f"Could not get voxel spacing: {e}, using 1mm isotropic")
                
                # Generate meshes
                mesh_paths = generate_tumor_meshes(
                    seg_for_mesh,
                    str(output_path),
                    spacing=spacing,
                    format='stl'
                )

                # Generate coarse brain mesh for spatial context
                brain_mesh_url = generate_brain_mesh(
                    mri_volume=t1ce_volume,
                    output_dir=str(output_path),
                    spacing=spacing,
                    format='stl'
                )

                # Create URLs for available meshes
                for name, mesh_path in mesh_paths.items():
                    if mesh_path is not None:
                        mesh_filename = Path(mesh_path).name
                        mesh_urls[name] = f"{base_url}/{job_id}/{mesh_filename}"
                        logger.info(f"Mesh {name} available at {mesh_urls[name]}")
                
                if brain_mesh_path is not None:
                    brain_mesh_filename = Path(brain_mesh_path).name
                    mesh_urls["brain"] = f"{base_url}/{job_id}/{brain_mesh_filename}"
                    logger.info(f"Brain mesh available at {mesh_urls['brain']}")

                
            except Exception as e:
                logger.error(f"Failed to generate 3D meshes: {e}", exc_info=True)
        
        # Build response
        response = {
            "job_id": job_id,
            "status": "success",
            "outputs": {
                "montage_axial": montage_urls.get('axial'),
                "montage_coronal": montage_urls.get('coronal'),
                "montage_sagittal": montage_urls.get('sagittal'),
                "overlay_class_1": overlay_urls.get(1),
                "overlay_class_2": overlay_urls.get(2),
                "overlay_class_3": overlay_urls.get(3),
                "comparison_overlay": comparison_overlay_url,
                "segmentation_nifti": f"{base_url}/{job_id}/{job_id}_segmentation.nii.gz" if seg_nii_path else None,
                "segmentation_numpy": f"{base_url}/{job_id}/{job_id}_segmentation.npy",
                "probability_maps": f"{base_url}/{job_id}/{job_id}_probabilities.npz",
                "mesh_necrotic": mesh_urls.get('necrotic'),
                "mesh_edema": mesh_urls.get('edema'),
                "mesh_enhancing": mesh_urls.get('enhancing'),
                "mesh_brain": mesh_urls.get('brain')
            },
            "metrics": metrics
        }
        
        logger.info(f"Inference completed successfully for job {job_id}")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference failed for job {job_id}: {e}", exc_info=True)
        
        # Try to save raw segmentation even on error
        try:
            if 'pred_labels' in locals():
                seg_npy_path = save_numpy(pred_labels, str(output_path / f"{job_id}_segmentation.npy"))
                print(f"[INFO] Saved raw segmentation after error: {seg_npy_path}")
        except:
            pass
        
        # Return partial response with error details
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "job_id": job_id,
                "message": "Inference failed. Raw segmentation may have been saved."
            }
        )


@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download or display generated output files."""
    file_path = OUTPUT_DIR / job_id / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File {filename} not found for job {job_id}"
        )
    
    # Determine if this is an image file or mesh
    filename_lower = filename.lower()
    is_image = filename_lower.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg'))
    is_mesh = filename_lower.endswith(('.stl', '.obj'))
    
    # Set appropriate media type
    if filename_lower.endswith('.png'):
        media_type = "image/png"
    elif filename_lower.endswith(('.jpg', '.jpeg')):
        media_type = "image/jpeg"
    elif filename_lower.endswith('.gif'):
        media_type = "image/gif"
    elif filename_lower.endswith('.webp'):
        media_type = "image/webp"
    elif filename_lower.endswith('.svg'):
        media_type = "image/svg+xml"
    elif filename_lower.endswith('.stl'):
        media_type = "model/stl"
    elif filename_lower.endswith('.obj'):
        media_type = "model/obj"
    else:
        media_type = "application/octet-stream"
    
    # Read file content
    with open(file_path, "rb") as f:
        file_content = f.read()
    
    # For images, use inline disposition; for meshes and other files, use attachment
    headers = {
        "Content-Type": media_type,
    }
    if is_image:
        # For images, use inline to display in browser
        headers["Content-Disposition"] = f'inline; filename="{filename}"'
        return Response(
            content=file_content,
            media_type=media_type,
            headers=headers
        )
    elif is_mesh:
        # For meshes, allow CORS and set appropriate headers for Three.js loading
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        headers["Access-Control-Allow-Origin"] = "*"
        return Response(
            content=file_content,
            media_type=media_type,
            headers=headers
        )
    else:
        # For other files, force download
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return Response(
            content=file_content,
            media_type=media_type,
            headers=headers
        )


@app.delete("/api/cleanup/{job_id}")
async def cleanup_job(job_id: str, _: bool = Depends(verify_api_key)):
    """Remove temp and outputs for a job."""
    temp_dir = TEMP_DIR / job_id
    output_dir = OUTPUT_DIR / job_id
    
    removed = []
    
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            removed.append(f"temp/{job_id}")
            logger.info(f"Removed temp directory for job {job_id}")
        
        if output_dir.exists():
            shutil.rmtree(output_dir)
            removed.append(f"outputs/{job_id}")
            logger.info(f"Removed output directory for job {job_id}")
        
        if not removed:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return {
            "status": "deleted",
            "job_id": job_id,
            "removed": removed
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cleanup failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
