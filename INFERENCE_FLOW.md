# Complete Inference Pipeline Flow

## Overview
This document describes the complete flow of what happens when you click "Start Inference" in the frontend.

---

## Step-by-Step Flow

### **1. Frontend: User Clicks "Start Inference"**

**Location:** `neurovision/src/components/FileUploadForm.tsx`

**What happens:**
- Form validation checks that all required files (flair, t1, t1ce, t2) are present
- Calls `inferWithFiles()` or `inferWithZip()` from `neurovision/src/lib/api.ts`
- Shows progress updates: "uploading" → "processing" → "complete"

---

### **2. Backend: File Upload Endpoint**

**Endpoint:** `POST /infer/upload` or `POST /infer/upload-zip`

**Location:** `server/app.py` (lines 218-350)

**What happens:**

#### For ZIP Upload (`/infer/upload-zip`):
1. Receives ZIP file in request
2. Generates unique `job_id` (UUID)
3. Creates temporary directory: `server/temp/{job_id}/`
4. Saves ZIP file to `server/temp/{job_id}/upload.zip`
5. Calls `extract_zip()` from `server/utils/file_processing.py`:
   - Extracts all files from ZIP
   - Searches for files matching patterns:
     - Required: `flair.*`, `t1.*`, `t1ce.*`, `t2.*`
     - Optional: `seg.*`, `boundary.*`, `distance.*`
   - Validates all required files are present
   - Returns dictionary mapping file types to paths
6. Returns JSON with `job_id` and extracted file paths

#### For Individual File Upload (`/infer/upload`):
1. Receives individual files (flair, t1, t1ce, t2, optional dist/boundary)
2. Generates unique `job_id` (UUID)
3. Creates temporary directory: `server/temp/{job_id}/`
4. Saves each file with predictable names:
   - `flair.nii` (or `.nii.gz` if that's the extension)
   - `t1.nii`, `t1ce.nii`, `t2.nii`
   - `dist.npy` (if provided)
   - `boundary.npy` (if provided)
5. Returns JSON with `job_id` and saved file paths

**Files accessed:**
- `server/temp/{job_id}/` - Temporary storage for uploaded files
- ZIP extraction uses `zipfile` Python library

---

### **3. Backend: Run Inference Endpoint**

**Endpoint:** `POST /infer/run`

**Location:** `server/app.py` (lines 353-560)

**What happens:**

#### 3.1 File Discovery and Loading
1. Looks in `server/temp/{job_id}/` for files
2. Searches for required files with flexible matching:
   - `flair.nii` or any file matching `flair.*` pattern
   - `t1.nii` or any file matching `t1.*` pattern (but not `t1ce.*`)
   - `t1ce.nii` or any file matching `t1ce.*` pattern
   - `t2.nii` or any file matching `t2.*` pattern
3. Searches for optional files:
   - `distance.*` or `dist.npy` → used as distance map
   - `boundary.*` or `boundary.npy` → used as boundary map
   - `seg.*` → loaded as ground truth for comparison
4. Creates output directory: `server/outputs/{job_id}/`

**Files accessed:**
- `server/temp/{job_id}/flair.*`
- `server/temp/{job_id}/t1.*`
- `server/temp/{job_id}/t1ce.*`
- `server/temp/{job_id}/t2.*`
- `server/temp/{job_id}/distance.*` or `dist.npy` (optional)
- `server/temp/{job_id}/boundary.*` or `boundary.npy` (optional)
- `server/temp/{job_id}/seg.*` (optional, for ground truth)

---

#### 3.2 Preprocessing

**Location:** `server/models/inference.py` → `preprocess_volume()` (lines 21-163)

**What happens:**

1. **Load NIfTI Files:**
   - Uses `nibabel` library to load each NIfTI file
   - Extracts data array and NIfTI header/affine
   - Stores reference NIfTI (from first file) for later use

2. **Validate Spatial Dimensions:**
   - Checks all volumes have same spatial shape (D, H, W)
   - Raises error if shapes don't match

3. **Load Optional Maps:**
   - Loads `distance.npy` or `distance.nii` if provided
   - Loads `boundary.npy` or `boundary.nii` if provided
   - Creates zero arrays if not provided

4. **Stack Channels:**
   - Creates 6-channel volume: `[flair, t1, t1ce, t2, dist, boundary]`
   - Shape: `(6, D, H, W)`

5. **Normalize MRI Modalities:**
   - Z-score normalization for first 4 channels (flair, t1, t1ce, t2)
   - Formula: `(value - mean) / std` (only for non-zero voxels)
   - Distance and boundary maps kept as-is

6. **Padding:**
   - Pads spatial dimensions to be divisible by 16
   - Required for 3D-UNet architecture
   - Stores padding info for later unpadding

7. **Convert to Tensor:**
   - Converts numpy array to PyTorch tensor
   - Adds batch dimension: `(1, 6, D', H', W')`
   - Keeps on CPU (moved to GPU later)

**Libraries used:**
- `nibabel` - NIfTI file loading
- `numpy` - Array operations
- `torch` - Tensor operations

---

#### 3.3 Model Inference

**Location:** `server/models/inference.py` → `run_inference()` (lines 166-228)

**Model Information:**
- **Architecture:** 3D-UNet from MONAI library (from `src/model.py`)
- **Framework:** MONAI (Medical Open Network for AI)
- **Input channels:** 6 (flair, t1, t1ce, t2, dist, boundary)
- **Output channels:** 4 (background, class 1, class 2, class 3)
- **Base filters:** 32 (channels: 32, 64, 128, 256, 512)
- **Strides:** (2, 2, 2, 2) - 4 downsampling/upsampling levels
- **Residual units:** 2 per level
- **Dropout:** 0.1
- **Checkpoint:** `models/checkpoint_epoch50.pth`
- **Device:** CUDA (if available) or CPU

**What happens:**

1. **Model Loading (on server startup):**
   - Location: `server/app.py` → `startup_event()` (lines 156-170)
   - Calls `get_cached_model()` from `server/models/model_loader.py`
   - Model is cached in memory (singleton pattern)
   - Model is loaded once at startup, reused for all requests

2. **Model Architecture:**
   - Loaded from `src/model.py` → `get_unet_model()` function
   - Instantiated with `in_channels=6, out_channels=4`
   - Checkpoint loaded from `models/checkpoint_epoch50.pth`

3. **Inference Execution:**
   - Moves input tensor to device (CUDA/CPU)
   - Runs model forward pass: `model(input_tensor)`
   - Output shape: `(1, 4, D', H', W')` - 4 class logits
   - Applies softmax to get probabilities: `(1, 4, D', H', W')`
   - Gets predicted labels via argmax: `(1, D', H', W')`
   - Converts to numpy arrays

4. **Unpadding:**
   - Removes padding to restore original spatial dimensions
   - Final prediction shape: `(D, H, W)` with labels 0-3
   - Final probability shape: `(4, D, H, W)`

**Files accessed:**
- `models/checkpoint_epoch50.pth` - Model weights (loaded once at startup)
- `src/model.py` - Model architecture definition

**Libraries used:**
- `torch` - PyTorch for model inference
- `torch.nn.functional` - For softmax operation

---

#### 3.4 Postprocessing

**Location:** `server/models/inference.py` → `postprocess()` (lines 253-317)

**What happens:**

1. **Get Voxel Spacing:**
   - Extracts voxel spacing from NIfTI header
   - Default: 1mm × 1mm × 1mm if not available

2. **Calculate Volumes:**
   - Counts voxels for each class:
     - Class 1 (Necrotic/Non-enhancing)
     - Class 2 (Peritumoral Edema)
     - Class 3 (Enhancing Tumor)
   - Calculates volumes in cubic centimeters (cc):
     - **WT (Whole Tumor):** Classes 1 + 2 + 3
     - **TC (Tumor Core):** Classes 1 + 3
     - **ET (Enhancing Tumor):** Class 3 only

3. **Returns Metrics:**
   ```python
   {
       'wt_volume_cc': float,
       'tc_volume_cc': float,
       'et_volume_cc': float
   }
   ```

---

#### 3.5 Save Outputs

**Location:** `server/app.py` (lines 432-449)

**What happens:**

1. **Save Segmentation:**
   - Saves as NumPy: `server/outputs/{job_id}/{job_id}_segmentation.npy`
   - Saves as NIfTI: `server/outputs/{job_id}/{job_id}_segmentation.nii.gz`
     - Uses reference NIfTI header/affine for proper spatial information

2. **Save Probability Maps:**
   - Saves as compressed NumPy: `server/outputs/{job_id}/{job_id}_probabilities.npz`
   - Contains 4 arrays (one per class): `class_0`, `class_1`, `class_2`, `class_3`

**Files created:**
- `server/outputs/{job_id}/{job_id}_segmentation.npy`
- `server/outputs/{job_id}/{job_id}_segmentation.nii.gz`
- `server/outputs/{job_id}/{job_id}_probabilities.npz`

**Libraries used:**
- `nibabel` - For saving NIfTI files
- `numpy` - For saving .npy and .npz files

---

#### 3.6 Create Visualizations

**Location:** `server/app.py` (lines 451-510) and `server/utils/visualization.py`

**What happens:**

1. **Montages (3 views):**
   - Creates montages for axial, coronal, and sagittal views
   - Shows 9 evenly-spaced slices with segmentation overlay
   - Saves as PNG: `server/outputs/{job_id}/montage_{view}.png`
   - Uses FLAIR volume as base image

2. **Class Overlays (3 classes):**
   - Creates overlay for each tumor class (1, 2, 3)
   - Shows segmentation colored by class on MRI slice
   - Saves as PNG: `server/outputs/{job_id}/overlay_class_{class_id}.png`

3. **Comparison Overlay (if ground truth available):**
   - Checks if `seg.*` file was loaded as ground truth
   - Binarizes predictions and ground truth (any non-zero = positive)
   - Creates color-coded overlay on middle axial T1CE slice:
     - **Green** = True Positive (predicted correctly)
     - **Yellow** = False Positive (predicted but not in ground truth)
     - **Red** = False Negative (missed by prediction)
   - Alpha-blends overlay with grayscale MRI
   - Saves as PNG: `server/outputs/{job_id}/comparison_overlay.png`

**Files created:**
- `server/outputs/{job_id}/montage_axial.png`
- `server/outputs/{job_id}/montage_coronal.png`
- `server/outputs/{job_id}/montage_sagittal.png`
- `server/outputs/{job_id}/overlay_class_1.png`
- `server/outputs/{job_id}/overlay_class_2.png`
- `server/outputs/{job_id}/overlay_class_3.png`
- `server/outputs/{job_id}/comparison_overlay.png` (if ground truth available)

**Libraries used:**
- `numpy` - Array operations
- `PIL` (Pillow) - Image creation and saving
- `matplotlib` - (if needed for visualization)

---

#### 3.7 Build Response

**Location:** `server/app.py` (lines 512-540)

**What happens:**

1. Creates JSON response with:
   - `job_id`: Unique identifier
   - `status`: "success"
   - `outputs`: Dictionary with URLs to all generated files
   - `metrics`: Volume metrics (WT, TC, ET in cc)

2. Returns response to frontend

**Response format:**
```json
{
  "job_id": "uuid-here",
  "status": "success",
  "outputs": {
    "montage_axial": "http://localhost:8000/api/download/{job_id}/montage_axial.png",
    "montage_coronal": "...",
    "montage_sagittal": "...",
    "overlay_class_1": "...",
    "overlay_class_2": "...",
    "overlay_class_3": "...",
    "comparison_overlay": "..." (if available),
    "segmentation_nifti": "...",
    "segmentation_numpy": "...",
    "probability_maps": "..."
  },
  "metrics": {
    "wt_volume_cc": 30.91,
    "tc_volume_cc": 12.27,
    "et_volume_cc": 5.16
  }
}
```

---

### **4. Frontend: Display Results**

**Location:** `neurovision/src/components/InferenceResults.tsx`

**What happens:**

1. Receives JSON response from backend
2. Displays metrics cards (WT, TC, ET volumes)
3. Shows montage images (axial, coronal, sagittal)
4. Shows class overlay images
5. Shows comparison overlay (if available) with color legend
6. Provides download buttons for all outputs

---

## Summary of Files Accessed

### Input Files (from user):
- `flair.nii` or `flair.nii.gz` (required)
- `t1.nii` or `t1.nii.gz` (required)
- `t1ce.nii` or `t1ce.nii.gz` (required)
- `t2.nii` or `t2.nii.gz` (required)
- `distance.nii` or `dist.npy` (optional)
- `boundary.nii` or `boundary.npy` (optional)
- `seg.nii` or `seg.nii.gz` (optional, for ground truth)

### Model Files:
- `models/checkpoint_epoch50.pth` - Model weights (loaded once at startup)
- `src/model.py` - Model architecture (imported at startup)

### Temporary Files:
- `server/temp/{job_id}/*` - Uploaded/extracted files

### Output Files:
- `server/outputs/{job_id}/{job_id}_segmentation.npy`
- `server/outputs/{job_id}/{job_id}_segmentation.nii.gz`
- `server/outputs/{job_id}/{job_id}_probabilities.npz`
- `server/outputs/{job_id}/montage_*.png` (3 files)
- `server/outputs/{job_id}/overlay_class_*.png` (3 files)
- `server/outputs/{job_id}/comparison_overlay.png` (if ground truth available)

---

## Model Details

- **Architecture:** 3D-UNet from MONAI
- **Library:** MONAI (Medical Open Network for AI)
- **Input:** 6 channels (flair, t1, t1ce, t2, distance, boundary)
- **Output:** 4 classes (background, class 1, class 2, class 3)
- **Architecture Details:**
  - Spatial dimensions: 3D
  - Base filters: 32
  - Channel progression: 32 → 64 → 128 → 256 → 512
  - Strides: (2, 2, 2, 2) - 4 levels of downsampling/upsampling
  - Residual units: 2 per level
  - Dropout: 0.1
- **Checkpoint:** `checkpoint_epoch50.pth` (epoch 50 of training)
- **Device:** CUDA (if available) or CPU
- **Mode:** Evaluation mode (no gradient computation)

---

## Key Libraries Used

- **nibabel** - NIfTI file I/O
- **numpy** - Array operations
- **torch** - PyTorch for model inference
- **monai** - Medical imaging deep learning framework (for 3D-UNet)
- **PIL (Pillow)** - Image creation and saving
- **fastapi** - Web framework
- **uvicorn** - ASGI server

---

## Performance Notes

- Model is loaded once at server startup and cached in memory
- Inference runs on GPU if available, otherwise CPU
- All file I/O operations use efficient numpy/nibabel operations
- Visualizations are generated on-demand and cached as PNG files
