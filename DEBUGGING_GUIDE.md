# Debugging Guide: "Nothing Shows After Clicking Start Inference"

## Quick Checks

### 1. Open Browser Console (F12)
Look for these messages:
- ✅ `[Form] Calling inferWithFiles API...` - Form submitted
- ✅ `[API] Uploading to: http://localhost:8000/infer/upload` - Upload started
- ✅ `[API] Upload successful. Response: {...}` - Upload completed
- ✅ `[API] Triggering inference at: http://localhost:8000/infer/run` - Inference started
- ✅ `[API] ✅ Inference completed successfully!` - Inference done
- ❌ Any red error messages

### 2. Check Network Tab (F12 → Network)
Filter by "Fetch/XHR" and look for:
- `POST /infer/upload` - Should return 200
- `POST /infer/run` - Should return 200
- If you see 404, 500, or other errors, note the error message

### 3. Check Which Page You're On
- **Dashboard (`/dashboard`)**: Uses simplified flow, may not show full results
- **Other pages**: Should use `FileUploadForm` component with full results display

### 4. Check Backend Logs
Look at the terminal where the backend is running. You should see:
```
INFO: Preprocessing volumes for job {job_id}
INFO: Running inference for job {job_id}
INFO: Postprocessing predictions for job {job_id}
INFO: Creating visualizations for job {job_id}
```

## Common Issues

### Issue 1: Files Not Uploaded
**Symptoms:** Error message about missing files
**Solution:** Make sure all 4 required files (flair, t1, t1ce, t2) are selected

### Issue 2: Backend Not Responding
**Symptoms:** Network errors, connection refused
**Solution:** 
- Check if backend is running: `http://localhost:8000/health`
- Restart backend if needed

### Issue 3: Results Not Displaying
**Symptoms:** No error, but no results shown
**Solution:**
- Check browser console for `[Form] ✅ Inference completed successfully!`
- Check if `result` state is set (should see results card appear)
- Scroll down on the page - results appear below the form

### Issue 4: CORS Errors
**Symptoms:** Console shows CORS policy errors
**Solution:** Backend CORS is configured, but check if frontend URL matches

## What to Share for Help

If still having issues, share:
1. Browser console output (copy all messages)
2. Network tab screenshot or list of failed requests
3. Backend terminal output (any error messages)
4. Which page/route you're on
