# Diagnostics and Troubleshooting Guide

This document provides commands and troubleshooting steps for running the NeuroVision inference server and tests.

## Quick Start Commands

### 1. Start Server

```bash
cd C:\Users\allur\brats7d_old\server
python run_server.py
```

Or using uvicorn directly:

```bash
cd C:\Users\allur\brats7d_old\server
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**
```
NeuroVision Inference Server - Starting...
Python version: 3.x.x
PyTorch version: 2.x.x
CUDA available: True/False
...
Starting server on http://0.0.0.0:8000
```

### 2. Run Test Script

```bash
cd C:\Users\allur\brats7d_old\server
python TESTS\run_inference_test.py
```

**Expected output:**
```
✓ Server is running
✓ Found flair: BraTS20_Training_001_flair.nii
...
✓ Upload successful
✓ Inference successful
✓ All tests passed!
```

### 3. Inspect Outputs

After running inference, check the output directory:

```bash
# List all job output directories
dir C:\Users\allur\brats7d_old\server\outputs

# List files for a specific job (replace JOB_ID with actual job ID)
dir C:\Users\allur\brats7d_old\server\outputs\JOB_ID

# View specific file (example)
# Use a file explorer or image viewer to open:
C:\Users\allur\brats7d_old\server\outputs\JOB_ID\montage_axial.png
```

**Expected files in output directory:**
- `montage_axial.png` - Axial view montage
- `montage_coronal.png` - Coronal view montage
- `montage_sagittal.png` - Sagittal view montage
- `overlay_class_1.png` - Class 1 overlay
- `overlay_class_2.png` - Class 2 overlay
- `overlay_class_3.png` - Class 3 overlay
- `{job_id}_segmentation.npy` - Segmentation as NumPy array
- `{job_id}_segmentation.nii.gz` - Segmentation as NIfTI (if reference available)
- `{job_id}_probabilities.npz` - Probability maps

## Troubleshooting Common Errors

### Error: Missing MONAI

**Symptoms:**
```
ModuleNotFoundError: No module named 'monai'
```

**Solution:**
```bash
cd C:\Users\allur\brats7d_old\server
pip install monai
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import monai; print(f'MONAI version: {monai.__version__}')"
```

---

### Error: GPU Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size or use CPU:**
   - Set environment variable: `set DEVICE=cpu`
   - Or modify `app.py` to force CPU mode

2. **Clear GPU cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Use smaller input volumes:**
   - Preprocess to smaller dimensions if possible

4. **Check GPU memory:**
   ```python
   import torch
   if torch.cuda.is_available():
       print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
       print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
   ```

---

### Error: Missing Sample Data Files

**Symptoms:**
```
✗ Sample data directory not found: C:\Users\allur\brats7d_old\data\BraTS20_Training_001
✗ Missing flair file
```

**Solution:**

1. **Verify data directory exists:**
   ```bash
   dir C:\Users\allur\brats7d_old\data\BraTS20_Training_001
   ```

2. **Check for expected files:**
   - `BraTS20_Training_001_flair.nii` or `.nii.gz`
   - `BraTS20_Training_001_t1.nii` or `.nii.gz`
   - `BraTS20_Training_001_t1ce.nii` or `.nii.gz`
   - `BraTS20_Training_001_t2.nii` or `.nii.gz`

3. **Update test script path:**
   - Edit `TESTS\run_inference_test.py`
   - Change `SAMPLE_DATA_DIR` to your actual data location

---

### Error: Model Not Loaded

**Symptoms:**
```
"model_loaded": false
503 Service Unavailable: Model not loaded
```

**Solutions:**

1. **Check checkpoint file exists:**
   ```bash
   dir C:\Users\allur\brats7d_old\models\checkpoint_epoch50.pth
   ```

2. **Check model source code:**
   ```bash
   dir C:\Users\allur\brats7d_old\src\model.py
   ```

3. **Test model loading manually:**
   ```python
   from models.model_loader import get_cached_model
   import torch
   
   checkpoint = r"C:\Users\allur\brats7d_old\models\checkpoint_epoch50.pth"
   device = "cuda" if torch.cuda.is_available() else "cpu"
   model = get_cached_model(checkpoint, device)
   print("Model loaded successfully!")
   ```

4. **Check server logs:**
   - Look for import errors or checkpoint loading errors in server output

---

### Error: Connection Refused

**Symptoms:**
```
✗ Cannot connect to server at http://localhost:8000
ConnectionError: [Errno 10061] No connection could be made
```

**Solutions:**

1. **Verify server is running:**
   - Check if uvicorn process is active
   - Look for "Starting server" message in terminal

2. **Check port availability:**
   ```bash
   netstat -ano | findstr :8000
   ```

3. **Try different port:**
   - Set `UVICORN_PORT=8001` environment variable
   - Update API_BASE_URL in test script

4. **Check firewall:**
   - Ensure Windows Firewall allows connections on port 8000

---

### Error: File Upload Fails

**Symptoms:**
```
✗ Upload failed with status 400/500
```

**Solutions:**

1. **Check file sizes:**
   - Large files (>500MB) may timeout
   - Check server logs for specific error

2. **Verify file formats:**
   - Ensure files are valid NIfTI (.nii, .nii.gz) or NumPy (.npy)
   - Test with smaller sample files first

3. **Check API key (if enabled):**
   - If API key is required, add header: `x-api-key: YOUR_KEY`
   - Or disable API key in server configuration

4. **Check temp directory permissions:**
   ```bash
   # Ensure temp directory is writable
   dir C:\Users\allur\brats7d_old\server\temp
   ```

---

### Error: Inference Timeout

**Symptoms:**
```
✗ Inference timed out (exceeded 5 minutes)
```

**Solutions:**

1. **Increase timeout in test script:**
   - Edit `TESTS\run_inference_test.py`
   - Change `timeout=300` to higher value (e.g., `timeout=600`)

2. **Check server logs:**
   - Look for preprocessing/inference progress
   - Verify model is actually running

3. **Test with smaller volumes:**
   - Use cropped or downsampled test data

---

### Error: Missing Dependencies

**Symptoms:**
```
ModuleNotFoundError: No module named 'X'
```

**Solution:**
```bash
cd C:\Users\allur\brats7d_old\server
pip install -r requirements.txt
```

**Common missing packages:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `torch` - PyTorch
- `nibabel` - NIfTI file handling
- `Pillow` - Image processing
- `numpy` - Numerical computing

---

## Manual Testing with curl

If Python requests library is not available, use curl:

### 1. Upload Files

```bash
curl -X POST http://localhost:8000/infer/upload ^
  -F "flair=@C:\Users\allur\brats7d_old\data\BraTS20_Training_001\BraTS20_Training_001_flair.nii" ^
  -F "t1=@C:\Users\allur\brats7d_old\data\BraTS20_Training_001\BraTS20_Training_001_t1.nii" ^
  -F "t1ce=@C:\Users\allur\brats7d_old\data\BraTS20_Training_001\BraTS20_Training_001_t1ce.nii" ^
  -F "t2=@C:\Users\allur\brats7d_old\data\BraTS20_Training_001\BraTS20_Training_001_t2.nii"
```

Save the `job_id` from response.

### 2. Run Inference

```bash
curl -X POST http://localhost:8000/infer/run ^
  -F "job_id=YOUR_JOB_ID_HERE"
```

### 3. Download File

```bash
curl -O http://localhost:8000/api/download/YOUR_JOB_ID_HERE/montage_axial.png
```

---

## Verification Checklist

After successful test run, verify:

- [ ] Server starts without errors
- [ ] Health endpoint returns `model_loaded: true`
- [ ] Upload endpoint accepts files and returns `job_id`
- [ ] Inference endpoint returns `status: "success"`
- [ ] Outputs contain all expected URLs (montages, overlays, segmentation)
- [ ] Metrics contain valid volume values (WT, TC, ET)
- [ ] Download endpoint serves files correctly
- [ ] Output directory contains all expected files
- [ ] PNG images can be opened and viewed
- [ ] Segmentation files (.npy, .nii.gz) are valid

---

## Getting Help

If issues persist:

1. **Check server logs:**
   - Look for detailed error messages in server terminal output
   - Check for stack traces or import errors

2. **Verify environment:**
   ```bash
   python --version
   pip list | findstr "torch monai fastapi"
   ```

3. **Test individual components:**
   - Test model loading: `python -c "from models.model_loader import get_cached_model; ..."`
   - Test file loading: `python -c "from utils.file_processing import load_nifti; ..."`

4. **Check file paths:**
   - Ensure all paths use correct separators for Windows
   - Verify no typos in directory names

