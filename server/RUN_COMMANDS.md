# Server Setup and Testing Commands

## 1. Install Dependencies

### Windows (PowerShell)
```powershell
cd C:\Users\allur\brats7d_old\server
python -m pip install -r requirements.txt
```

### Linux
```bash
cd /path/to/brats7d/server
pip install -r requirements.txt
```

**Note**: If using a virtual environment:
- Windows: `python -m venv venv` then `.\venv\Scripts\Activate.ps1`
- Linux: `python3 -m venv venv` then `source venv/bin/activate`

---

## 2. Run Server

### Windows (PowerShell)
```powershell
cd C:\Users\allur\brats7d_old\server
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Linux
```bash
cd /path/to/brats7d/server
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Alternative** (using Python directly):
```python
python app.py
```

---

## 3. Test API with curl

### Step 1: Upload Files

#### Windows (PowerShell)
```powershell
curl -X POST http://localhost:8000/infer/upload `
  -F "flair=@C:\path\to\BraTS20_Training_001_flair.nii" `
  -F "t1=@C:\path\to\BraTS20_Training_001_t1.nii" `
  -F "t1ce=@C:\path\to\BraTS20_Training_001_t1ce.nii" `
  -F "t2=@C:\path\to\BraTS20_Training_001_t2.nii" `
  -F "dist=@C:\path\to\BraTS20_Training_001_dist.npy" `
  -F "boundary=@C:\path\to\BraTS20_Training_001_boundary.npy"
```

#### Linux
```bash
curl -X POST http://localhost:8000/infer/upload \
  -F "flair=@/path/to/BraTS20_Training_001_flair.nii" \
  -F "t1=@/path/to/BraTS20_Training_001_t1.nii" \
  -F "t1ce=@/path/to/BraTS20_Training_001_t1ce.nii" \
  -F "t2=@/path/to/BraTS20_Training_001_t2.nii" \
  -F "dist=@/path/to/BraTS20_Training_001_dist.npy" \
  -F "boundary=@/path/to/BraTS20_Training_001_boundary.npy"
```

**Expected Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploaded",
  "files": {
    "flair": "C:\\Users\\allur\\brats7d_old\\server\\temp\\550e8400-e29b-41d4-a716-446655440000\\flair.nii",
    "t1": "C:\\Users\\allur\\brats7d_old\\server\\temp\\550e8400-e29b-41d4-a716-446655440000\\t1.nii",
    "t1ce": "C:\\Users\\allur\\brats7d_old\\server\\temp\\550e8400-e29b-41d4-a716-446655440000\\t1ce.nii",
    "t2": "C:\\Users\\allur\\brats7d_old\\server\\temp\\550e8400-e29b-41d4-a716-446655440000\\t2.nii",
    "dist": "C:\\Users\\allur\\brats7d_old\\server\\temp\\550e8400-e29b-41d4-a716-446655440000\\dist.npy",
    "boundary": "C:\\Users\\allur\\brats7d_old\\server\\temp\\550e8400-e29b-41d4-a716-446655440000\\boundary.npy"
  }
}
```

### Step 2: Run Inference

#### Windows (PowerShell)
```powershell
curl -X POST http://localhost:8000/infer/run `
  -F "job_id=550e8400-e29b-41d4-a716-446655440000"
```

#### Linux
```bash
curl -X POST http://localhost:8000/infer/run \
  -F "job_id=550e8400-e29b-41d4-a716-446655440000"
```

**Expected Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "outputs": {
    "montage_axial": "http://localhost:8000/api/download/550e8400-e29b-41d4-a716-446655440000/montage_axial.png",
    "montage_coronal": "http://localhost:8000/api/download/550e8400-e29b-41d4-a716-446655440000/montage_coronal.png",
    "montage_sagittal": "http://localhost:8000/api/download/550e8400-e29b-41d4-a716-446655440000/montage_sagittal.png",
    "overlay_class_1": "http://localhost:8000/api/download/550e8400-e29b-41d4-a716-446655440000/overlay_class_1.png",
    "overlay_class_2": "http://localhost:8000/api/download/550e8400-e29b-41d4-a716-446655440000/overlay_class_2.png",
    "overlay_class_3": "http://localhost:8000/api/download/550e8400-e29b-41d4-a716-446655440000/overlay_class_3.png",
    "segmentation_nifti": "http://localhost:8000/api/download/550e8400-e29b-41d4-a716-446655440000/550e8400-e29b-41d4-a716-446655440000_segmentation.nii.gz",
    "segmentation_numpy": "http://localhost:8000/api/download/550e8400-e29b-41d4-a716-446655440000/550e8400-e29b-41d4-a716-446655440000_segmentation.npy",
    "probability_maps": "http://localhost:8000/api/download/550e8400-e29b-41d4-a716-446655440000/550e8400-e29b-41d4-a716-446655440000_probabilities.npz"
  },
  "metrics": {
    "wt_volume_cc": 45.2,
    "tc_volume_cc": 28.7,
    "et_volume_cc": 15.3
  }
}
```

---

## Additional Test Commands

### Health Check
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

### Download Example File
```bash
curl -O http://localhost:8000/api/download/550e8400-e29b-41d4-a716-446655440000/montage_axial.png
```

