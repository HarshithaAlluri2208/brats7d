# Diagnostic Commands for Common Failures

## 1. Model Import Errors

**Command:**
```bash
python -c "import sys; sys.path.insert(0, r'C:\Users\allur\brats7d_old\src'); from model import get_unet_model; print('✅ Import OK')"
```

**What it checks:** Verifies `src/model.py` can be imported and `get_unet_model` function exists.

**Common fixes:**
- Ensure `C:\Users\allur\brats7d_old\src\model.py` exists
- Check Python path includes `C:\Users\allur\brats7d_old\src`
- Verify function name is exactly `get_unet_model`

---

## 2. Checkpoint Not Found

**Command:**
```bash
python -c "import os; path=r'C:\Users\allur\brats7d_old\models\checkpoint_epoch50.pth'; print(f'Exists: {os.path.exists(path)}', f'Size: {os.path.getsize(path)/1e6:.1f}MB' if os.path.exists(path) else 'NOT FOUND')"
```

**What it checks:** Verifies checkpoint file exists and shows file size.

**Common fixes:**
- Check file path is correct
- Verify file permissions (readable)
- Check if file is corrupted (size should be > 0)

---

## 3. Shapes Mismatch (Tensor Concat Error)

**Command:**
```bash
python -c "import numpy as np; import nibabel as nib; f=nib.load(r'C:\Users\allur\brats7d_old\data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii'); d=f.get_fdata(); print(f'Shape: {d.shape}, Dtype: {d.dtype}, Range: [{d.min():.1f}, {d.max():.1f}]')"
```

**What it checks:** Verifies input volume shape and data type consistency.

**Extended diagnostic:**
```python
python -c "
import numpy as np
import nibabel as nib
import os

files = ['flair', 't1', 't1ce', 't2']
base = r'C:\Users\allur\brats7d_old\data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001'
for f in files:
    path = os.path.join(base, f'BraTS20_Training_001_{f}.nii')
    if os.path.exists(path):
        d = nib.load(path).get_fdata()
        print(f'{f}: shape={d.shape}, dtype={d.dtype}')
    else:
        print(f'{f}: NOT FOUND')
"
```

**Common fixes:**
- Ensure all input volumes have same spatial dimensions (D, H, W)
- Check for NaN or Inf values: `np.isnan(d).any()` or `np.isinf(d).any()`
- Verify channel stacking: should be (6, D, H, W) after stacking

---

## 4. Missing MONAI

**Command:**
```bash
python -c "import monai; print(f'MONAI version: {monai.__version__}'); from monai.networks.nets import UNet; print('✅ UNet import OK')"
```

**What it checks:** Verifies MONAI is installed and UNet can be imported.

**Common fixes:**
- Install: `pip install monai>=1.5.0`
- Check version compatibility: `pip show monai`
- Reinstall if corrupted: `pip uninstall monai && pip install monai`

---

## 5. GPU OOM (Out of Memory)

**Command:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB' if torch.cuda.is_available() else 'No GPU')"
```

**Extended diagnostic:**
```python
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Total memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
    torch.cuda.empty_cache()
    print(f'Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f}GB')
    print(f'Reserved: {torch.cuda.memory_reserved(0)/1e9:.2f}GB')
else:
    print('CUDA not available - will use CPU')
"
```

**Common fixes:**
- Use CPU: Set `DEVICE = "cpu"` in `app.py` config
- Reduce batch size: Already using batch_size=1, but can try smaller input patches
- Clear cache: `torch.cuda.empty_cache()` before inference
- Check other processes using GPU: `nvidia-smi`

---

## Quick All-in-One Diagnostic

**Run all checks at once:**
```python
python -c "
import sys, os
print('=== Model Import ===')
try:
    sys.path.insert(0, r'C:\Users\allur\brats7d_old\src')
    from model import get_unet_model
    print('✅ Model import OK')
except Exception as e:
    print(f'❌ Model import failed: {e}')

print('\n=== Checkpoint ===')
cp = r'C:\Users\allur\brats7d_old\models\checkpoint_epoch50.pth'
print(f'Exists: {os.path.exists(cp)}')
if os.path.exists(cp):
    print(f'Size: {os.path.getsize(cp)/1e6:.1f}MB')

print('\n=== MONAI ===')
try:
    import monai
    from monai.networks.nets import UNet
    print(f'✅ MONAI {monai.__version__} OK')
except Exception as e:
    print(f'❌ MONAI failed: {e}')

print('\n=== GPU ===')
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
"
```

---

## Error-Specific Quick Fixes

### ImportError: cannot import name 'get_unet_model'
```bash
python -c "import sys; sys.path.insert(0, r'C:\Users\allur\brats7d_old\src'); import model; print(dir(model))"
```

### FileNotFoundError: checkpoint_epoch50.pth
```bash
python -c "import os; print([f for f in os.listdir(r'C:\Users\allur\brats7d_old\models') if f.endswith('.pth')])"
```

### RuntimeError: Sizes of tensors must match
```bash
python -c "import numpy as np, nibabel as nib; d=nib.load(r'C:\Users\allur\brats7d_old\data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii').get_fdata(); print(f'Shape: {d.shape}, Min: {d.min()}, Max: {d.max()}, Has NaN: {np.isnan(d).any()}')"
```

### ModuleNotFoundError: No module named 'monai'
```bash
pip install monai && python -c "import monai; print(monai.__version__)"
```

### CUDA out of memory
```bash
python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared. Try setting DEVICE=\"cpu\" in app.py')"
```

