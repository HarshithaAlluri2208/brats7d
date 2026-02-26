# 3D-UNet Medical Segmentation Model Integration Plan

## Project Context
- **Project Root**: `C:\Users\allur\brats7d_old`
- **Model Checkpoint**: `C:\Users\allur\brats7d_old\models\checkpoint_epoch50.pth`
- **Model Code**: `C:\Users\allur\brats7d_old\src\model.py`, `C:\Users\allur\brats7d_old\src\dataset.py`
- **Frontend**: `C:\Users\allur\brats7d_old\neurovision` (Next.js/React)
- **Model Architecture**: 3D-UNet (MONAI), 6 input channels, 4 output classes (0-3)
- **Input Format**: Stacked channels [flair, t1, t1ce, t2, dist, boundary] shaped (6, D, H, W)
- **Output Format**: Segmentation labels (0..3) + probability maps per class

---

## Phase 1: Backend Inference Server Setup

### 1.1 Create Server Directory Structure
- Create directory: `C:\Users\allur\brats7d_old\server`
- Create subdirectories:
  - `C:\Users\allur\brats7d_old\server\__init__.py` (empty file)
  - `C:\Users\allur\brats7d_old\server\models\` (for model loading utilities)
  - `C:\Users\allur\brats7d_old\server\api\` (for API route handlers)
  - `C:\Users\allur\brats7d_old\server\utils\` (for image processing, visualization utilities)
  - `C:\Users\allur\brats7d_old\server\temp\` (for temporary uploaded files, auto-cleanup)
  - `C:\Users\allur\brats7d_old\server\outputs\` (for generated outputs, auto-cleanup)

### 1.2 Create Server Dependencies File
- Create file: `C:\Users\allur\brats7d_old\server\requirements.txt`
- Include dependencies:
  - `fastapi==0.115.0` (or Flask if preferred)
  - `uvicorn[standard]==0.32.0` (for FastAPI) or `flask==3.0.0` (for Flask)
  - `python-multipart==0.0.9` (for file uploads)
  - `torch>=2.0.0`
  - `monai>=1.3.0`
  - `nibabel>=5.0.0`
  - `numpy>=1.24.0`
  - `Pillow>=10.0.0`
  - `matplotlib>=3.7.0`
  - `scipy>=1.10.0`
  - `pydantic>=2.0.0` (for FastAPI request/response models)

### 1.3 Create Model Loading Module
- Create file: `C:\Users\allur\brats7d_old\server\models\model_loader.py`
- Functions to implement:
  - `load_checkpoint(checkpoint_path: str, device: str) -> torch.nn.Module`
    - Load model architecture from `C:\Users\allur\brats7d_old\src\model.py` (import `get_unet_model`)
    - Load weights from `C:\Users\allur\brats7d_old\models\checkpoint_epoch50.pth`
    - Set model to eval mode
    - Return model on specified device (cuda/cpu)
  - `get_model()` -> singleton pattern to cache loaded model

### 1.4 Create Inference Engine Module
- Create file: `C:\Users\allur\brats7d_old\server\models\inference.py`
- Functions to implement:
  - `preprocess_volume(volume_data: np.ndarray, channels: List[str]) -> torch.Tensor`
    - Accept 6-channel input: [flair, t1, t1ce, t2, dist, boundary]
    - Apply z-score normalization to first 4 channels (MRI modalities)
    - Handle padding if depth not divisible by 16
    - Return tensor shape: (1, 6, D, H, W)
  - `run_inference(model: torch.nn.Module, input_tensor: torch.Tensor, device: str) -> Tuple[np.ndarray, np.ndarray]`
    - Run forward pass with `torch.no_grad()`
    - Return: (predicted_labels: np.ndarray shape (D,H,W), probability_maps: np.ndarray shape (4,D,H,W))
  - `postprocess_segmentation(pred_labels: np.ndarray, prob_maps: np.ndarray) -> Dict`
    - Compute volume statistics (WT, TC, ET volumes in cc)
    - Return dictionary with segmentation and metrics

### 1.5 Create Visualization Utilities Module
- Create file: `C:\Users\allur\brats7d_old\server\utils\visualization.py`
- Functions to implement:
  - `create_montage(volume: np.ndarray, seg_labels: np.ndarray, view: str) -> np.ndarray`
    - Generate per-slice montages for axial, coronal, sagittal views
    - Overlay segmentation labels on original volume slices
    - Return RGB image array
  - `create_class_overlay(volume: np.ndarray, seg_labels: np.ndarray, class_id: int) -> List[np.ndarray]`
    - Generate colored overlay images per class (0=background, 1=NET, 2=ED, 3=ET)
    - Return list of PNG image arrays (one per representative slice)
  - `save_montage_png(image_array: np.ndarray, output_path: str) -> str`
    - Save montage as PNG file
    - Return file path

### 1.6 Create File Processing Utilities Module
- Create file: `C:\Users\allur\brats7d_old\server\utils\file_processing.py`
- Functions to implement:
  - `load_nifti_file(file_path: str) -> Tuple[np.ndarray, nibabel.Nifti1Image]`
    - Load NIfTI file, return data array and NIfTI object (for affine/header)
  - `load_numpy_file(file_path: str) -> np.ndarray`
    - Load .npy file
  - `save_segmentation_nifti(seg_array: np.ndarray, reference_nii: nibabel.Nifti1Image, output_path: str) -> str`
    - Save segmentation as NIfTI with same affine/header as reference
  - `save_segmentation_numpy(seg_array: np.ndarray, output_path: str) -> str`
    - Save segmentation as .npy file
  - `save_probability_maps_npz(prob_maps: np.ndarray, output_path: str) -> str`
    - Save probability maps as compressed .npz file
    - Keys: 'class_0', 'class_1', 'class_2', 'class_3'

### 1.7 Create Main API Server File
- Create file: `C:\Users\allur\brats7d_old\server\main.py` (FastAPI) OR `C:\Users\allur\brats7d_old\server\app.py` (Flask)
- Endpoints to implement:

#### Endpoint 1: Health Check
- **Path**: `GET /health`
- **Response**: `{"status": "ok", "model_loaded": bool}`

#### Endpoint 2: Upload and Run Inference
- **Path**: `POST /api/infer`
- **Request**: 
  - `multipart/form-data` with fields:
    - `flair`: File (NIfTI or .npy)
    - `t1`: File (NIfTI or .npy)
    - `t1ce`: File (NIfTI or .npy)
    - `t2`: File (NIfTI or .npy)
    - `dist`: File (optional, .npy) - if missing, use zeros
    - `boundary`: File (optional, .npy) - if missing, use zeros
    - `patient_id`: String (optional, for naming outputs)
- **Response JSON**:
  ```json
  {
    "patient_id": "string",
    "status": "success",
    "outputs": {
      "montage_axial": "http://localhost:8000/api/download/{id}/montage_axial.png",
      "montage_coronal": "http://localhost:8000/api/download/{id}/montage_coronal.png",
      "montage_sagittal": "http://localhost:8000/api/download/{id}/montage_sagittal.png",
      "overlay_class_1": "http://localhost:8000/api/download/{id}/overlay_class_1.png",
      "overlay_class_2": "http://localhost:8000/api/download/{id}/overlay_class_2.png",
      "overlay_class_3": "http://localhost:8000/api/download/{id}/overlay_class_3.png",
      "segmentation_nifti": "http://localhost:8000/api/download/{id}/segmentation.nii.gz",
      "segmentation_numpy": "http://localhost:8000/api/download/{id}/segmentation.npy",
      "probability_maps": "http://localhost:8000/api/download/{id}/probabilities.npz"
    },
    "metrics": {
      "wt_volume_cc": float,
      "tc_volume_cc": float,
      "et_volume_cc": float
    }
  }
  ```

#### Endpoint 3: Download Generated Files
- **Path**: `GET /api/download/{session_id}/{filename}`
- **Response**: File download (PNG, NIfTI, .npy, .npz)
- **Files served from**: `C:\Users\allur\brats7d_old\server\outputs\{session_id}\{filename}`

#### Endpoint 4: Cleanup Session (Optional)
- **Path**: `DELETE /api/cleanup/{session_id}`
- **Response**: `{"status": "deleted"}`

### 1.8 Create Server Configuration File
- Create file: `C:\Users\allur\brats7d_old\server\config.py`
- Configuration variables:
  - `MODEL_CHECKPOINT_PATH = "C:\\Users\\allur\\brats7d_old\\models\\checkpoint_epoch50.pth"`
  - `MODEL_SOURCE_PATH = "C:\\Users\\allur\\brats7d_old\\src"`
  - `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`
  - `TEMP_DIR = "C:\\Users\\allur\\brats7d_old\\server\\temp"`
  - `OUTPUT_DIR = "C:\\Users\\allur\\brats7d_old\\server\\outputs"`
  - `MAX_FILE_SIZE_MB = 500`
  - `SERVER_PORT = 8000`
  - `CORS_ORIGINS = ["http://localhost:3000"]` (for Next.js frontend)

### 1.9 Create Server Startup Script
- Create file: `C:\Users\allur\brats7d_old\server\run_server.py`
- Script to:
  - Import and configure FastAPI/Flask app
  - Set up CORS middleware
  - Load model on startup (warm-up)
  - Start uvicorn server (FastAPI) or Flask development server
  - Run on `http://localhost:8000`

---

## Phase 2: Frontend Integration

### 2.1 Update API Client Library
- Update file: `C:\Users\allur\brats7d_old\neurovision\src\lib\api.ts`
- Replace placeholder `inferWithFile` function with real implementation:
  - Function signature: `inferWithFiles(files: {flair: File, t1: File, t1ce: File, t2: File, dist?: File, boundary?: File, patientId?: string}): Promise<InferResponse>`
  - Create `FormData` with all file fields
  - POST to `process.env.NEXT_PUBLIC_API_URL + "/api/infer"` (default: `http://localhost:8000/api/infer`)
  - Handle response JSON with download URLs and metrics
  - Return typed response matching backend schema

### 2.2 Create Type Definitions
- Create file: `C:\Users\allur\brats7d_old\neurovision\src\lib\types.ts`
- Define TypeScript interfaces:
  - `InferResponse` (matches backend JSON response)
  - `InferOutputs` (download URLs structure)
  - `InferMetrics` (volume metrics)

### 2.3 Create Image Display Component
- Create file: `C:\Users\allur\brats7d_old\neurovision\src\components\inference-results.tsx`
- Component props:
  - `results: InferResponse`
  - `onDownload: (url: string, filename: string) => void`
- Component features:
  - Display montage images (axial, coronal, sagittal) using `<img>` tags
  - Display overlay images per class in a grid
  - Show metrics (WT, TC, ET volumes) in a card
  - Download buttons for each output file (NIfTI, .npy, .npz)

### 2.4 Create File Upload Component
- Create file: `C:\Users\allur\brats7d_old\neurovision\src\components\file-upload-form.tsx`
- Component features:
  - File input fields for: flair, t1, t1ce, t2, dist (optional), boundary (optional)
  - Patient ID input field (optional)
  - Upload button that calls `inferWithFiles` from `api.ts`
  - Loading state with progress indicator
  - Error handling and display
  - On success, render `InferenceResults` component

### 2.5 Update Dashboard/Visualization Page
- Update file: `C:\Users\allur\brats7d_old\neurovision\src\app\visualization\page.tsx` OR create new page
- Integrate:
  - Import `FileUploadForm` component
  - Import `InferenceResults` component
  - Add state management for inference results
  - Render upload form and results display

### 2.6 Update Environment Configuration
- Create/update file: `C:\Users\allur\brats7d_old\neurovision\.env.local`
- Add: `NEXT_PUBLIC_API_URL=http://localhost:8000`

---

## Phase 3: Testing Instructions

### 3.1 Backend Server Testing

#### Test 1: Health Check
- **Command**: `curl http://localhost:8000/health`
- **Expected Response**: `{"status": "ok", "model_loaded": true}`

#### Test 2: Inference with Sample Data
- **Test Data Location**: Use `C:\Users\allur\brats7d_old\data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\`
- **Files to upload**:
  - `BraTS20_Training_001_flair.nii`
  - `BraTS20_Training_001_t1.nii`
  - `BraTS20_Training_001_t1ce.nii`
  - `BraTS20_Training_001_t2.nii`
  - Optional: `C:\Users\allur\brats7d_old\data\processed_maps\distance\BraTS20_Training_001_dist.npy`
  - Optional: `C:\Users\allur\brats7d_old\data\processed_maps\boundary\BraTS20_Training_001_boundary.npy`
- **Command** (using curl):
  ```bash
  curl -X POST http://localhost:8000/api/infer \
    -F "flair=@BraTS20_Training_001_flair.nii" \
    -F "t1=@BraTS20_Training_001_t1.nii" \
    -F "t1ce=@BraTS20_Training_001_t1ce.nii" \
    -F "t2=@BraTS20_Training_001_t2.nii" \
    -F "dist=@BraTS20_Training_001_dist.npy" \
    -F "boundary=@BraTS20_Training_001_boundary.npy" \
    -F "patient_id=test_001"
  ```
- **Expected Response JSON**:
  ```json
  {
    "patient_id": "test_001",
    "status": "success",
    "outputs": {
      "montage_axial": "http://localhost:8000/api/download/test_001_<timestamp>/montage_axial.png",
      "montage_coronal": "http://localhost:8000/api/download/test_001_<timestamp>/montage_coronal.png",
      "montage_sagittal": "http://localhost:8000/api/download/test_001_<timestamp>/montage_sagittal.png",
      "overlay_class_1": "http://localhost:8000/api/download/test_001_<timestamp>/overlay_class_1.png",
      "overlay_class_2": "http://localhost:8000/api/download/test_001_<timestamp>/overlay_class_2.png",
      "overlay_class_3": "http://localhost:8000/api/download/test_001_<timestamp>/overlay_class_3.png",
      "segmentation_nifti": "http://localhost:8000/api/download/test_001_<timestamp>/segmentation.nii.gz",
      "segmentation_numpy": "http://localhost:8000/api/download/test_001_<timestamp>/segmentation.npy",
      "probability_maps": "http://localhost:8000/api/download/test_001_<timestamp>/probabilities.npz"
    },
    "metrics": {
      "wt_volume_cc": <float>,
      "tc_volume_cc": <float>,
      "et_volume_cc": <float>
    }
  }
  ```

#### Test 3: Download Files
- **Command**: `curl http://localhost:8000/api/download/{session_id}/montage_axial.png -o test_montage.png`
- **Expected**: PNG file downloaded and viewable
- **Repeat for**: segmentation.nii.gz, segmentation.npy, probabilities.npz

#### Test 4: Verify Output Files
- **Check directory**: `C:\Users\allur\brats7d_old\server\outputs\{session_id}\`
- **Expected files**:
  - `montage_axial.png`
  - `montage_coronal.png`
  - `montage_sagittal.png`
  - `overlay_class_1.png`
  - `overlay_class_2.png`
  - `overlay_class_3.png`
  - `segmentation.nii.gz`
  - `segmentation.npy`
  - `probabilities.npz`

### 3.2 Frontend Integration Testing

#### Test 1: Start Frontend and Backend
- **Backend**: Run `python C:\Users\allur\brats7d_old\server\run_server.py` (should start on port 8000)
- **Frontend**: Run `npm run dev` in `C:\Users\allur\brats7d_old\neurovision` (should start on port 3000)

#### Test 2: Upload Files via UI
- Navigate to `http://localhost:3000/visualization` (or appropriate page)
- Upload test files via file upload form
- Verify loading state appears
- Verify results display after inference completes

#### Test 3: Verify Image Display
- Check that montage images (axial, coronal, sagittal) render correctly
- Check that overlay images per class render correctly
- Verify images are loaded from backend URLs

#### Test 4: Verify Download Links
- Click download buttons for:
  - Segmentation NIfTI file
  - Segmentation NumPy file
  - Probability maps NPZ file
- Verify files download and are valid (can open NIfTI in viewer, load .npy/.npz in Python)

#### Test 5: Verify Metrics Display
- Check that volume metrics (WT, TC, ET) are displayed correctly
- Verify values are reasonable (positive floats, in cubic centimeters)

### 3.3 Expected Output File Formats

#### PNG Files (Montages and Overlays)
- **Format**: RGB PNG, 8-bit per channel
- **Dimensions**: Variable (depends on volume dimensions and slice selection)
- **Content**: 
  - Montages: Grid of slices with segmentation overlay
  - Overlays: Single representative slice per class with colored segmentation overlay

#### NIfTI File (Segmentation)
- **Format**: `.nii.gz` (compressed NIfTI)
- **Data Type**: `uint8` (labels 0-3)
- **Dimensions**: (D, H, W) matching input volume
- **Affine/Header**: Preserved from input reference volume

#### NumPy File (Segmentation)
- **Format**: `.npy` (uncompressed NumPy array)
- **Data Type**: `uint8` (labels 0-3)
- **Shape**: (D, H, W)
- **Load in Python**: `np.load('segmentation.npy')`

#### NPZ File (Probability Maps)
- **Format**: `.npz` (compressed NumPy archive)
- **Keys**: `'class_0'`, `'class_1'`, `'class_2'`, `'class_3'`
- **Data Type**: `float32` (probabilities 0.0-1.0)
- **Shape per key**: (D, H, W)
- **Load in Python**: 
  ```python
  data = np.load('probabilities.npz')
  prob_class_1 = data['class_1']
  ```

---

## Phase 4: Deployment Considerations

### 4.1 Error Handling
- Backend should handle:
  - Missing optional files (dist, boundary) → use zeros
  - Invalid file formats → return 400 error
  - Model loading failures → return 500 error
  - CUDA out of memory → fallback to CPU or return 507 error
- Frontend should handle:
  - Network errors → display error message
  - Invalid file types → validate before upload
  - Timeout errors → show retry option

### 4.2 Performance Optimization
- Model loading: Load once on server startup (singleton pattern)
- File cleanup: Implement automatic cleanup of temp/output files after 24 hours
- Caching: Consider caching inference results by patient ID (if same patient uploaded twice)

### 4.3 Security Considerations
- File size limits: Enforce MAX_FILE_SIZE_MB
- File type validation: Only accept .nii, .nii.gz, .npy files
- CORS: Configure CORS to only allow frontend origin
- Input sanitization: Validate patient_id to prevent path traversal

---

## File Structure Summary

```
C:\Users\allur\brats7d_old\
├── models\
│   └── checkpoint_epoch50.pth
├── src\
│   ├── model.py
│   ├── dataset.py
│   └── utils.py
├── server\                          [NEW]
│   ├── __init__.py
│   ├── main.py (or app.py)
│   ├── run_server.py
│   ├── config.py
│   ├── requirements.txt
│   ├── models\
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   └── inference.py
│   ├── api\
│   │   ├── __init__.py
│   │   └── routes.py (if using Flask)
│   ├── utils\
│   │   ├── __init__.py
│   │   ├── visualization.py
│   │   └── file_processing.py
│   ├── temp\                        [AUTO-CREATED]
│   └── outputs\                     [AUTO-CREATED]
│       └── {session_id}\
│           ├── montage_axial.png
│           ├── montage_coronal.png
│           ├── montage_sagittal.png
│           ├── overlay_class_1.png
│           ├── overlay_class_2.png
│           ├── overlay_class_3.png
│           ├── segmentation.nii.gz
│           ├── segmentation.npy
│           └── probabilities.npz
└── neurovision\
    └── src\
        ├── lib\
        │   ├── api.ts              [UPDATE]
        │   └── types.ts            [NEW]
        └── components\
            ├── file-upload-form.tsx [NEW]
            └── inference-results.tsx [NEW]
```

---

## Implementation Order

1. **Backend Setup** (Phase 1):
   - 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 1.7 → 1.8 → 1.9
2. **Backend Testing** (Phase 3.1):
   - Test health check, inference endpoint, file downloads
3. **Frontend Integration** (Phase 2):
   - 2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6
4. **End-to-End Testing** (Phase 3.2):
   - Test full workflow from frontend upload to results display

