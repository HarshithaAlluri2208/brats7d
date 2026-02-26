# Image Display Fix - Summary

## Changes Made

### 1. Backend Changes (`server/app.py`)

#### CORS Update (Lines 52-55)
- **Added ports 3004 and 3005** to the CORS allowed origins list
- Now includes: `http://localhost:3000`, `http://localhost:3004`, `http://localhost:3005`

#### Download Endpoint Fix (Lines 462-503)
- **Detects image files** by extension (.png, .jpg, .jpeg, .gif, .webp, .svg)
- **Sets correct Content-Type** headers for each image type
- **Uses `Content-Disposition: inline`** for images (displays in browser)
- **Uses `Content-Disposition: attachment`** for non-image files (triggers download)

### 2. Frontend Changes

#### New Component: `neurovision/src/components/image-display.tsx`
- **Robust image display component** with blob fallback
- Handles direct image URLs
- Falls back to blob URL if server sends `Content-Disposition: attachment` or wrong `Content-Type`
- Properly cleans up blob URLs on unmount
- Shows loading and error states

#### Updated Components:
- `neurovision/src/components/FileUploadForm.tsx` - Replaced `<img>` with `<ImageDisplay>`
- `neurovision/src/components/inference-results.tsx` - Replaced `<img>` with `<ImageDisplay>`

## File Paths

### Backend
- `C:\Users\allur\brats7d_old\server\app.py`

### Frontend
- `C:\Users\allur\brats7d_old\neurovision\src\components\image-display.tsx` (NEW)
- `C:\Users\allur\brats7d_old\neurovision\src\components\FileUploadForm.tsx` (MODIFIED)
- `C:\Users\allur\brats7d_old\neurovision\src\components\inference-results.tsx` (MODIFIED)

## Restart Instructions

### Backend (FastAPI)
1. **Stop the current server** (if running):
   ```powershell
   Get-Process | Where-Object {$_.ProcessName -eq "python"} | Stop-Process -Force
   ```

2. **Restart the server**:
   ```powershell
   cd C:\Users\allur\brats7d_old\server
   python -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```

   OR if using the run script:
   ```powershell
   cd C:\Users\allur\brats7d_old\server
   python run_server.py
   ```

### Frontend (Next.js)
1. **The Next.js dev server should auto-reload** on file changes
2. If needed, **restart manually**:
   ```powershell
   cd C:\Users\allur\brats7d_old\neurovision
   npm run dev
   ```

## Test Instructions

### 1. Verify Backend Changes

**Test the download endpoint with an image:**
```powershell
# Get a job_id from a previous inference or run a new one
$jobId = "1f2c2840-6c92-4f99-8284-53eafefc949e"  # Replace with actual job_id
$filename = "montage_axial.png"

# Check headers
curl -I "http://localhost:8000/api/download/$jobId/$filename"
```

**Expected response headers:**
```
Content-Type: image/png
Content-Disposition: inline; filename="montage_axial.png"
Access-Control-Allow-Origin: http://localhost:3000
```

### 2. Test Frontend Image Display

1. **Open the frontend** at `http://localhost:3000`
2. **Navigate to the page with FileUploadForm** (or wherever inference results are shown)
3. **Run an inference** with sample files:
   - Upload FLAIR, T1, T1CE, T2 files
   - Click "Run Inference"
   - Wait for completion

4. **Verify images display:**
   - Check that montage images (axial, coronal, sagittal) appear
   - Check that overlay images (class 1, 2, 3) appear
   - Images should render inline (not trigger downloads)

5. **Check browser console** for log messages:
   ```
   [Inference] Response received: {...}
   [Inference] Output URLs: {...}
   [ImageDisplay] Axial montage loaded: http://localhost:8000/api/download/...
   [ImageDisplay] Status: 200 OK
   ```

6. **Check Network tab:**
   - Open DevTools → Network tab
   - Filter by "Img" or "XHR"
   - Look for requests to `/api/download/{job_id}/montage_*.png`
   - Verify status: **200 OK**
   - Check Response Headers:
     - `Content-Type: image/png`
     - `Content-Disposition: inline; filename="..."`

### 3. Test Blob Fallback (Optional)

If you want to test the blob fallback mechanism:
1. Temporarily modify the backend to send `Content-Disposition: attachment` for images
2. The ImageDisplay component should automatically fetch as blob and display inline
3. Check console for: `[ImageDisplay] Direct load failed, fetching as blob: ...`

## Expected Behavior

### Before Fix
- Images might trigger downloads instead of displaying
- CORS errors for ports 3004/3005
- Wrong Content-Type headers

### After Fix
- ✅ Images display inline in the browser
- ✅ Correct Content-Type headers (image/png, image/jpeg, etc.)
- ✅ CORS enabled for all dev ports (3000, 3004, 3005)
- ✅ Robust fallback to blob URLs if needed
- ✅ Proper cleanup of blob URLs

## Troubleshooting

### Images still downloading instead of displaying
1. Check backend logs to verify `Content-Disposition: inline` is being sent
2. Check browser Network tab for response headers
3. Verify CORS headers are present

### CORS errors
1. Verify backend CORS includes your frontend port
2. Check browser console for CORS error messages
3. Restart backend server after CORS changes

### Images not loading
1. Check browser console for error messages
2. Verify job_id and filename are correct in the URL
3. Check that files exist in `server/outputs/{job_id}/`
4. Verify backend server is running on port 8000

## Summary

All changes have been implemented:
- ✅ Backend serves images with `Content-Disposition: inline` and correct `Content-Type`
- ✅ CORS updated for ports 3000, 3004, 3005
- ✅ Robust ImageDisplay component with blob fallback
- ✅ Integrated into FileUploadForm and inference-results components
- ✅ Proper cleanup of blob URLs

The frontend should now display images inline after inference completes, with automatic fallback to blob URLs if the server sends attachment headers.

