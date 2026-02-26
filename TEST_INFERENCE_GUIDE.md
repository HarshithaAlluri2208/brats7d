# How to Test Inference from Frontend

## Step-by-Step Guide

### 1. Make sure both servers are running:
- **Backend**: `http://localhost:8000` (should show health check OK)
- **Frontend**: `http://localhost:3000` (Next.js dev server)

### 2. Open the frontend in your browser:
- Go to: `http://localhost:3000`
- Find the page with file upload form (usually the main page or a specific route)

### 3. Upload files:
- You need 4 required files:
  - **FLAIR** (.nii or .nii.gz)
  - **T1** (.nii or .nii.gz)
  - **T1CE** (.nii or .nii.gz)
  - **T2** (.nii or .nii.gz)
- Optional files:
  - **Distance map** (.npy)
  - **Boundary map** (.npy)

### 4. Click "Run Inference" or "Start Inference"

### 5. Watch the browser console (F12 → Console tab):
You should see:
```
Uploading to: http://localhost:8000/infer/upload
Triggering run at: http://localhost:8000/infer/run job: <job_id>
[Inference] Response received: {job_id: "...", status: "success", ...}
[Inference] Output URLs: {...}
[ImageDisplay] Fetching image as blob: http://localhost:8000/api/download/...
[ImageDisplay] Successfully created blob URL for: ...
```

### 6. Check the Network tab (F12 → Network):
- Filter by "Fetch/XHR" or "Img"
- Look for:
  - POST `/infer/upload` → Should return 200 with `job_id`
  - POST `/infer/run` → Should return 200 with full JSON response
  - GET `/api/download/{job_id}/montage_*.png` → Should return 200

### 7. The JSON response should look like:
```json
{
  "job_id": "some-uuid-here",
  "status": "success",
  "outputs": {
    "montage_axial": "http://localhost:8000/api/download/{job_id}/montage_axial.png",
    "montage_coronal": "http://localhost:8000/api/download/{job_id}/montage_coronal.png",
    "montage_sagittal": "http://localhost:8000/api/download/{job_id}/montage_sagittal.png",
    "overlay_class_1": "http://localhost:8000/api/download/{job_id}/overlay_class_1.png",
    "overlay_class_2": "http://localhost:8000/api/download/{job_id}/overlay_class_2.png",
    "overlay_class_3": "http://localhost:8000/api/download/{job_id}/overlay_class_3.png",
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

## Troubleshooting

### If you don't see a JSON response:
1. **Check browser console for errors** - Look for red error messages
2. **Check Network tab** - See if requests are failing (status 400, 500, etc.)
3. **Check backend logs** - The server console should show processing logs
4. **Verify files are valid** - Make sure you're uploading actual NIfTI files

### If images still download instead of displaying:
- The ImageDisplay component should handle this by fetching as blob
- Check browser console for `[ImageDisplay]` messages
- If you see errors, share them

### If you get "detail: not found":
- The job_id might not exist yet (inference didn't complete)
- Wait for inference to finish
- Check that files exist in `server/outputs/{job_id}/`

## Quick Test URLs

Once you have a job_id from a successful inference, test these URLs:
- `http://localhost:8000/api/download/{your_job_id}/montage_axial.png`
- Should display the image inline (not download)

## What Changed

1. **Backend**: Now uses `Response` instead of `FileResponse` for images to prevent forced downloads
2. **Frontend**: `ImageDisplay` component always fetches images as blobs to ensure they display inline
3. **CORS**: Updated to allow ports 3000, 3004, 3005

