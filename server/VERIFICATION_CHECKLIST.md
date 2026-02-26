# End-to-End Verification Checklist

## Prerequisites
- [ ] Server dependencies installed: `pip install -r requirements.txt`
- [ ] Model checkpoint exists: `C:\Users\allur\brats7d_old\models\checkpoint_epoch50.pth`
- [ ] Server running: `python -m uvicorn app:app --host 0.0.0.0 --port 8000`
- [ ] Sample data files available (e.g., from `C:\Users\allur\brats7d_old\data\BraTS2020_TrainingData\...`)

---

## Step 1: Verify Model Loads

**Command:**
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

**Check:** ✅ `model_loaded` is `true`

---

## Step 2: Upload Sample Files

**Command (Windows PowerShell):**
```powershell
$jobId = (curl -X POST http://localhost:8000/infer/upload `
  -F "flair=@C:\Users\allur\brats7d_old\data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii" `
  -F "t1=@C:\Users\allur\brats7d_old\data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1.nii" `
  -F "t1ce=@C:\Users\allur\brats7d_old\data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1ce.nii" `
  -F "t2=@C:\Users\allur\brats7d_old\data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t2.nii" `
  -F "dist=@C:\Users\allur\brats7d_old\data\processed_maps\distance\BraTS20_Training_001_dist.npy" `
  -F "boundary=@C:\Users\allur\brats7d_old\data\processed_maps\boundary\BraTS20_Training_001_boundary.npy" | ConvertFrom-Json).job_id
Write-Host "Job ID: $jobId"
```

**Command (Linux):**
```bash
JOB_ID=$(curl -X POST http://localhost:8000/infer/upload \
  -F "flair=@/path/to/BraTS20_Training_001_flair.nii" \
  -F "t1=@/path/to/BraTS20_Training_001_t1.nii" \
  -F "t1ce=@/path/to/BraTS20_Training_001_t1ce.nii" \
  -F "t2=@/path/to/BraTS20_Training_001_t2.nii" \
  -F "dist=@/path/to/BraTS20_Training_001_dist.npy" \
  -F "boundary=@/path/to/BraTS20_Training_001_boundary.npy" | jq -r '.job_id')
echo "Job ID: $JOB_ID"
```

**Expected Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploaded",
  "files": { ... }
}
```

**Check:** ✅ `job_id` is returned and saved to variable

**Note:** If `dist` or `boundary` files are missing, omit those `-F` lines (they're optional).

---

## Step 3: Run Inference Job

**Command (Windows PowerShell - use $jobId from Step 2):**
```powershell
curl -X POST http://localhost:8000/infer/run -F "job_id=$jobId" | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

**Command (Linux - use $JOB_ID from Step 2):**
```bash
curl -X POST http://localhost:8000/infer/run -F "job_id=$JOB_ID" | jq
```

**Expected Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "outputs": {
    "montage_axial": "http://localhost:8000/api/download/.../montage_axial.png",
    ...
  },
  "metrics": {
    "wt_volume_cc": 45.2,
    "tc_volume_cc": 28.7,
    "et_volume_cc": 15.3
  }
}
```

**Check:** ✅ `status` is `"success"`, metrics are positive numbers

---

## Step 4: Download and Verify Montage

**Command (Windows PowerShell):**
```powershell
curl http://localhost:8000/api/download/$jobId/montage_axial.png -o test_montage.png
```

**Command (Linux):**
```bash
curl http://localhost:8000/api/download/$JOB_ID/montage_axial.png -o test_montage.png
```

**Check:** 
- ✅ File `test_montage.png` is created
- ✅ Open file in image viewer - should show grid of brain slices with segmentation overlay

---

## Step 5: Inspect Segmentation Files in Python

**Create Python script `inspect_seg.py`:**

```python
import numpy as np
import nibabel as nib
import os

# Replace with your actual job_id
job_id = "550e8400-e29b-41d4-a716-446655440000"
output_dir = r"C:\Users\allur\brats7d_old\server\outputs"  # or your path

# Check .npy file
npy_path = os.path.join(output_dir, job_id, f"{job_id}_segmentation.npy")
if os.path.exists(npy_path):
    seg_npy = np.load(npy_path)
    print(f"NumPy segmentation shape: {seg_npy.shape}")
    print(f"NumPy dtype: {seg_npy.dtype}")
    print(f"Unique labels: {np.unique(seg_npy)}")
    print(f"Label counts: {np.bincount(seg_npy.flatten())}")
    print(f"Expected labels: {{0, 1, 2, 3}}")
    assert set(np.unique(seg_npy)) <= {0, 1, 2, 3}, "Invalid labels found!"
    print("✅ NumPy file valid")
else:
    print(f"❌ NumPy file not found: {npy_path}")

# Check .nii.gz file
nii_path = os.path.join(output_dir, job_id, f"{job_id}_segmentation.nii.gz")
if os.path.exists(nii_path):
    seg_nii = nib.load(nii_path)
    seg_data = seg_nii.get_fdata()
    print(f"\nNIfTI segmentation shape: {seg_data.shape}")
    print(f"NIfTI dtype: {seg_data.dtype}")
    print(f"Unique labels: {np.unique(seg_data)}")
    print(f"Label counts: {np.bincount(seg_data.flatten().astype(int))}")
    assert set(np.unique(seg_data)) <= {0, 1, 2, 3}, "Invalid labels found!"
    print("✅ NIfTI file valid")
else:
    print(f"❌ NIfTI file not found: {nii_path}")

# Check probability maps
npz_path = os.path.join(output_dir, job_id, f"{job_id}_probabilities.npz")
if os.path.exists(npz_path):
    probs = np.load(npz_path)
    print(f"\nProbability maps keys: {list(probs.keys())}")
    for key in ['class_0', 'class_1', 'class_2', 'class_3']:
        prob = probs[key]
        print(f"{key}: shape={prob.shape}, dtype={prob.dtype}, range=[{prob.min():.3f}, {prob.max():.3f}]")
    print("✅ Probability maps valid")
else:
    print(f"❌ Probability maps not found: {npz_path}")
```

**Run:**
```bash
python inspect_seg.py
```

**Expected Output:**
```
NumPy segmentation shape: (155, 240, 240)
NumPy dtype: uint8
Unique labels: [0 1 2 3]
Label counts: [12345678 1234 5678 9012]
Expected labels: {0, 1, 2, 3}
✅ NumPy file valid

NIfTI segmentation shape: (155, 240, 240)
NIfTI dtype: float64
Unique labels: [0. 1. 2. 3.]
Label counts: [12345678 1234 5678 9012]
✅ NIfTI file valid

Probability maps keys: ['class_0', 'class_1', 'class_2', 'class_3']
class_0: shape=(155, 240, 240), dtype=float32, range=[0.000, 1.000]
class_1: shape=(155, 240, 240), dtype=float32, range=[0.000, 1.000]
class_2: shape=(155, 240, 240), dtype=float32, range=[0.000, 1.000]
class_3: shape=(155, 240, 240), dtype=float32, range=[0.000, 1.000]
✅ Probability maps valid
```

**Check:** ✅ All files exist, labels are exactly {0, 1, 2, 3}, probabilities are in [0, 1]

---

## Quick One-Liner Verification (Python)

**Quick check script:**
```python
import numpy as np
import sys

job_id = sys.argv[1] if len(sys.argv) > 1 else input("Enter job_id: ")
npy_path = f"C:\\Users\\allur\\brats7d_old\\server\\outputs\\{job_id}\\{job_id}_segmentation.npy"

seg = np.load(npy_path)
labels = set(np.unique(seg))
expected = {0, 1, 2, 3}

print(f"Shape: {seg.shape}, Dtype: {seg.dtype}")
print(f"Labels found: {labels}")
print(f"Expected: {expected}")
print(f"✅ Valid" if labels <= expected else f"❌ Invalid labels: {labels - expected}")
```

**Run:**
```bash
python -c "import numpy as np; seg=np.load('path/to/segmentation.npy'); print(f'Labels: {set(np.unique(seg))}')"
```

---

## Summary Checklist

- [ ] Step 1: Health check returns `model_loaded: true`
- [ ] Step 2: Upload returns valid `job_id`
- [ ] Step 3: Inference returns `status: "success"` with metrics
- [ ] Step 4: Montage PNG downloads and displays correctly
- [ ] Step 5: Segmentation files contain only labels {0, 1, 2, 3}
- [ ] Step 5: Probability maps are valid (4 classes, range [0, 1])

**All checks passed?** ✅ End-to-end integration verified!

