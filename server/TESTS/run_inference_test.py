"""
Test script for inference server endpoints.

This script:
- Loads sample patient data from BraTS dataset
- Tests upload and run endpoints
- Verifies outputs and downloads files
- Prints success/failure status
"""
import os
import sys
import requests
import json
from pathlib import Path

# ===============================
# Project Paths (Portable)
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SERVER_DIR = PROJECT_ROOT / "server"

sys.path.insert(0, str(SERVER_DIR))

API_BASE_URL = "http://localhost:8000"

SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "BraTS20_Training_001"

# Expected file names in sample data directory
EXPECTED_FILES = {
    "flair": "BraTS20_Training_001_flair.nii",
    "t1": "BraTS20_Training_001_t1.nii",
    "t1ce": "BraTS20_Training_001_t1ce.nii",
    "t2": "BraTS20_Training_001_t2.nii",
}

# Alternative file patterns (in case of .nii.gz extension)
ALT_PATTERNS = {
    "flair": ["BraTS20_Training_001_flair.nii.gz", "BraTS20_Training_001_flair.nii"],
    "t1": ["BraTS20_Training_001_t1.nii.gz", "BraTS20_Training_001_t1.nii"],
    "t1ce": ["BraTS20_Training_001_t1ce.nii.gz", "BraTS20_Training_001_t1ce.nii"],
    "t2": ["BraTS20_Training_001_t2.nii.gz", "BraTS20_Training_001_t2.nii"],
}


def find_file(directory: Path, patterns: list) -> Path:
    """Find a file matching one of the patterns."""
    for pattern in patterns:
        file_path = directory / pattern
        if file_path.exists():
            return file_path
    return None


def check_server_health():
    """Check if server is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Server is running")
            print(f"  Model loaded: {data.get('model_loaded', False)}")
            print(f"  Status: {data.get('status', 'unknown')}")
            return data.get('model_loaded', False)
        else:
            print(f"[FAIL] Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] Cannot connect to server at {API_BASE_URL}")
        print(f"  Make sure the server is running: python run_server.py")
        return False
    except Exception as e:
        print(f"[FAIL] Error checking server health: {e}")
        return False


def find_sample_files():
    """Find sample patient files."""
    if not SAMPLE_DATA_DIR.exists():
        print(f"[FAIL] Sample data directory not found: {SAMPLE_DATA_DIR}")
        print(f"  Please ensure BraTS data is available at this location")
        return None
    
    files = {}
    for key, patterns in ALT_PATTERNS.items():
        file_path = find_file(SAMPLE_DATA_DIR, patterns)
        if file_path:
            files[key] = file_path
            print(f"[OK] Found {key}: {file_path.name}")
        else:
            print(f"[FAIL] Missing {key} file (tried: {', '.join(patterns)})")
            return None
    
    return files


def test_upload(files: dict):
    """Test file upload endpoint."""
    print("\n" + "=" * 60)
    print("Testing /infer/upload endpoint...")
    print("=" * 60)
    
    try:
        # Prepare form data
        form_data = {}
        file_data = {}
        
        for key, file_path in files.items():
            file_data[key] = (
                file_path.name,
                open(file_path, 'rb'),
                'application/octet-stream'
            )
        
        # Upload files
        response = requests.post(
            f"{API_BASE_URL}/infer/upload",
            files=file_data,
            timeout=60
        )
        
        # Close file handles
        for key, (_, file_handle, _) in file_data.items():
            file_handle.close()
        
        if response.status_code != 200:
            print(f"[FAIL] Upload failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return None
        
        data = response.json()
        job_id = data.get('job_id')
        
        if not job_id:
            print(f"[FAIL] Upload response missing job_id")
            print(f"  Response: {json.dumps(data, indent=2)}")
            return None
        
        print(f"[OK] Upload successful")
        print(f"  Job ID: {job_id}")
        print(f"  Status: {data.get('status', 'unknown')}")
        print(f"  Files saved: {len(data.get('files', {}))}")
        
        return job_id
    
    except Exception as e:
        print(f"[FAIL] Upload test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_run_inference(job_id: str):
    """Test inference run endpoint."""
    print("\n" + "=" * 60)
    print("Testing /infer/run endpoint...")
    print("=" * 60)
    
    try:
        # Run inference
        form_data = {'job_id': job_id}
        response = requests.post(
            f"{API_BASE_URL}/infer/run",
            data=form_data,
            timeout=300  # 5 minutes for inference
        )
        
        if response.status_code != 200:
            print(f"[FAIL] Inference failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return None
        
        data = response.json()
        
        if data.get('status') != 'success':
            print(f"[FAIL] Inference returned non-success status: {data.get('status')}")
            return None
        
        print(f"[OK] Inference successful")
        print(f"  Job ID: {data.get('job_id')}")
        
        # Check outputs
        outputs = data.get('outputs', {})
        print(f"\n  Outputs:")
        for key, url in outputs.items():
            if url:
                print(f"    - {key}: {url}")
            else:
                print(f"    - {key}: (null)")
        
        # Check metrics
        metrics = data.get('metrics', {})
        print(f"\n  Metrics:")
        print(f"    - WT Volume: {metrics.get('wt_volume_cc', 0):.2f} cc")
        print(f"    - TC Volume: {metrics.get('tc_volume_cc', 0):.2f} cc")
        print(f"    - ET Volume: {metrics.get('et_volume_cc', 0):.2f} cc")
        
        return data
    
    except requests.exceptions.Timeout:
        print(f"[FAIL] Inference timed out (exceeded 5 minutes)")
        return None
    except Exception as e:
        print(f"[FAIL] Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_download(result_data: dict):
    """Test downloading output files."""
    print("\n" + "=" * 60)
    print("Testing file downloads...")
    print("=" * 60)
    
    if not result_data:
        print("[FAIL] No result data to test downloads")
        return False
    
    outputs = result_data.get('outputs', {})
    job_id = result_data.get('job_id')
    
    if not job_id:
        print("[FAIL] Missing job_id in result data")
        return False
    
    # Test downloading montage_axial
    montage_url = outputs.get('montage_axial')
    if not montage_url:
        print("[FAIL] montage_axial URL not found in outputs")
        return False
    
    try:
        print(f"  Downloading: {montage_url}")
        response = requests.get(montage_url, timeout=30)
        
        if response.status_code != 200:
            print(f"[FAIL] Download failed with status {response.status_code}")
            return False
        
        # Check if it's actually an image
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type.lower():
            print(f"[FAIL] Downloaded file is not an image (content-type: {content_type})")
            return False
        
        # Save to test output directory
        test_output_dir = SERVER_DIR / "TESTS" / "test_outputs"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = test_output_dir / f"{job_id}_montage_axial.png"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        file_size = output_file.stat().st_size
        print(f"[OK] Downloaded montage_axial successfully")
        print(f"  Saved to: {output_file}")
        print(f"  Size: {file_size / 1024:.2f} KB")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] Download test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_output_files(job_id: str):
    """List all output files for a job."""
    print("\n" + "=" * 60)
    print(f"Listing output files for job {job_id}...")
    print("=" * 60)
    
    output_dir = SERVER_DIR / "outputs" / job_id
    
    if not output_dir.exists():
        print(f"[FAIL] Output directory not found: {output_dir}")
        return
    
    files = list(output_dir.glob("*"))
    
    if not files:
        print(f"[FAIL] No files found in output directory")
        return
    
    print(f"[OK] Found {len(files)} file(s):")
    for file_path in sorted(files):
        size = file_path.stat().st_size
        print(f"  - {file_path.name} ({size / 1024:.2f} KB)")


def main():
    """Main test function."""
    print("=" * 60)
    print("NeuroVision Inference Server - Test Script")
    print("=" * 60)
    
    # Step 1: Check server health
    print("\n[1/5] Checking server health...")
    if not check_server_health():
        print("\n[FAIL] Server health check failed. Please start the server first.")
        sys.exit(1)
    
    # Step 2: Find sample files
    print("\n[2/5] Finding sample patient files...")
    files = find_sample_files()
    if not files:
        print("\n[FAIL] Could not find sample files. Test aborted.")
        sys.exit(1)
    
    # Step 3: Test upload
    print("\n[3/5] Testing file upload...")
    job_id = test_upload(files)
    if not job_id:
        print("\n[FAIL] Upload test failed. Test aborted.")
        sys.exit(1)
    
    # Step 4: Test inference
    print("\n[4/5] Testing inference...")
    result_data = test_run_inference(job_id)
    if not result_data:
        print("\n[FAIL] Inference test failed. Test aborted.")
        sys.exit(1)
    
    # Step 5: Test download
    print("\n[5/5] Testing file download...")
    download_success = test_download(result_data)
    
    # List all output files
    list_output_files(job_id)
    
    # Final summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"[OK] Upload: PASSED")
    print(f"[OK] Inference: PASSED")
    print(f"{'[OK]' if download_success else '[FAIL]'} Download: {'PASSED' if download_success else 'FAILED'}")
    print(f"\nJob ID: {job_id}")
    print(f"Output directory: {SERVER_DIR / 'outputs' / job_id}")
    
    if download_success:
        print("\n[OK] All tests passed!")
        sys.exit(0)
    else:
        print("\n[FAIL] Some tests failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

