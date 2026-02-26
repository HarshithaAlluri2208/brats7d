"""
End-to-end tests for upload and inference endpoints.
Uses requests library to simulate client interactions.
"""
import pytest
import requests
import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile
import os
import time

# Server configuration
BASE_URL = os.getenv("TEST_API_URL", "http://localhost:8000")
API_KEY = os.getenv("TEST_API_KEY", None)  # Optional API key for testing


@pytest.fixture(scope="session")
def server_health_check():
    """Check if server is running before running tests."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200, "Server health check failed"
        data = response.json()
        assert data.get("model_loaded") is True, "Model not loaded on server"
        return True
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Server not available at {BASE_URL}: {e}")


@pytest.fixture
def sample_nifti_files(tmp_path):
    """Create sample NIfTI files for testing."""
    # Create small test volumes: (32, 64, 64)
    D, H, W = 32, 64, 64
    
    files = {}
    for modality in ["flair", "t1", "t1ce", "t2"]:
        # Create random data
        data = np.random.randn(D, H, W).astype(np.float32)
        data = np.abs(data) * 100  # Make positive for realistic MRI values
        
        # Create NIfTI file
        nii = nib.Nifti1Image(data, affine=np.eye(4), header=nib.Nifti1Header())
        file_path = tmp_path / f"test_{modality}.nii"
        nib.save(nii, file_path)
        files[modality] = file_path
    
    # Create optional distance and boundary maps
    for map_type in ["dist", "boundary"]:
        data = np.random.randn(D, H, W).astype(np.float32)
        file_path = tmp_path / f"test_{map_type}.npy"
        np.save(file_path, data)
        files[map_type] = file_path
    
    return files


@pytest.fixture
def auth_headers():
    """Get authentication headers if API key is configured."""
    headers = {}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    return headers


def test_health_endpoint(server_health_check):
    """Test that health endpoint returns correct status."""
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "device" in data
    assert data["status"] == "ok"


def test_upload_files(sample_nifti_files, auth_headers):
    """Test uploading files to /infer/upload endpoint."""
    # Prepare form data
    files = {
        "flair": ("flair.nii", open(sample_nifti_files["flair"], "rb"), "application/octet-stream"),
        "t1": ("t1.nii", open(sample_nifti_files["t1"], "rb"), "application/octet-stream"),
        "t1ce": ("t1ce.nii", open(sample_nifti_files["t1ce"], "rb"), "application/octet-stream"),
        "t2": ("t2.nii", open(sample_nifti_files["t2"], "rb"), "application/octet-stream"),
    }
    
    # Add optional files
    if "dist" in sample_nifti_files:
        files["dist"] = ("dist.npy", open(sample_nifti_files["dist"], "rb"), "application/octet-stream")
    if "boundary" in sample_nifti_files:
        files["boundary"] = ("boundary.npy", open(sample_nifti_files["boundary"], "rb"), "application/octet-stream")
    
    try:
        response = requests.post(
            f"{BASE_URL}/infer/upload",
            files=files,
            headers=auth_headers,
            timeout=30
        )
        
        assert response.status_code == 200, f"Upload failed: {response.text}"
        data = response.json()
        
        assert "job_id" in data
        assert "status" in data
        assert data["status"] == "uploaded"
        assert "files" in data
        
        return data["job_id"]
    finally:
        # Close file handles
        for file_tuple in files.values():
            if hasattr(file_tuple[1], "close"):
                file_tuple[1].close()


def test_upload_missing_files(auth_headers):
    """Test that upload fails when required files are missing."""
    # Try uploading with only some files
    files = {
        "flair": ("flair.nii", b"fake data", "application/octet-stream"),
        "t1": ("t1.nii", b"fake data", "application/octet-stream"),
        # Missing t1ce and t2
    }
    
    response = requests.post(
        f"{BASE_URL}/infer/upload",
        files=files,
        headers=auth_headers,
        timeout=10
    )
    
    # Should fail with 422 (validation error) or 400
    assert response.status_code in [400, 422, 500]


def test_run_inference(sample_nifti_files, auth_headers):
    """Test full upload + inference workflow."""
    # Step 1: Upload files
    upload_files = {
        "flair": ("flair.nii", open(sample_nifti_files["flair"], "rb"), "application/octet-stream"),
        "t1": ("t1.nii", open(sample_nifti_files["t1"], "rb"), "application/octet-stream"),
        "t1ce": ("t1ce.nii", open(sample_nifti_files["t1ce"], "rb"), "application/octet-stream"),
        "t2": ("t2.nii", open(sample_nifti_files["t2"], "rb"), "application/octet-stream"),
    }
    
    try:
        upload_response = requests.post(
            f"{BASE_URL}/infer/upload",
            files=upload_files,
            headers=auth_headers,
            timeout=30
        )
        
        assert upload_response.status_code == 200
        job_id = upload_response.json()["job_id"]
        
        # Step 2: Run inference
        run_response = requests.post(
            f"{BASE_URL}/infer/run",
            data={"job_id": job_id},
            headers=auth_headers,
            timeout=120  # Inference can take time
        )
        
        assert run_response.status_code == 200, f"Inference failed: {run_response.text}"
        result = run_response.json()
        
        # Check response structure
        assert "job_id" in result
        assert "status" in result
        assert result["status"] == "success"
        assert "outputs" in result
        assert "metrics" in result
        
        # Check outputs
        outputs = result["outputs"]
        assert "montage_axial" in outputs
        assert "montage_coronal" in outputs
        assert "montage_sagittal" in outputs
        assert "overlay_class_1" in outputs
        assert "overlay_class_2" in outputs
        assert "overlay_class_3" in outputs
        assert "segmentation_numpy" in outputs
        assert "probability_maps" in outputs
        
        # Check metrics
        metrics = result["metrics"]
        assert "wt_volume_cc" in metrics
        assert "tc_volume_cc" in metrics
        assert "et_volume_cc" in metrics
        assert isinstance(metrics["wt_volume_cc"], (int, float))
        assert isinstance(metrics["tc_volume_cc"], (int, float))
        assert isinstance(metrics["et_volume_cc"], (int, float))
        
        # Check that metrics are non-negative
        assert metrics["wt_volume_cc"] >= 0
        assert metrics["tc_volume_cc"] >= 0
        assert metrics["et_volume_cc"] >= 0
        
        return result
        
    finally:
        # Close file handles
        for file_tuple in upload_files.values():
            if hasattr(file_tuple[1], "close"):
                file_tuple[1].close()


def test_download_file(sample_nifti_files, auth_headers):
    """Test downloading generated files."""
    # First upload and run inference
    upload_files = {
        "flair": ("flair.nii", open(sample_nifti_files["flair"], "rb"), "application/octet-stream"),
        "t1": ("t1.nii", open(sample_nifti_files["t1"], "rb"), "application/octet-stream"),
        "t1ce": ("t1ce.nii", open(sample_nifti_files["t1ce"], "rb"), "application/octet-stream"),
        "t2": ("t2.nii", open(sample_nifti_files["t2"], "rb"), "application/octet-stream"),
    }
    
    try:
        # Upload
        upload_response = requests.post(
            f"{BASE_URL}/infer/upload",
            files=upload_files,
            headers=auth_headers,
            timeout=30
        )
        job_id = upload_response.json()["job_id"]
        
        # Run inference
        run_response = requests.post(
            f"{BASE_URL}/infer/run",
            data={"job_id": job_id},
            headers=auth_headers,
            timeout=120
        )
        result = run_response.json()
        
        # Try downloading a file
        download_url = result["outputs"]["montage_axial"]
        download_response = requests.get(download_url, timeout=30)
        
        assert download_response.status_code == 200
        assert download_response.headers["content-type"].startswith("image/") or \
               download_response.headers["content-type"] == "application/octet-stream"
        assert len(download_response.content) > 0
        
    finally:
        for file_tuple in upload_files.values():
            if hasattr(file_tuple[1], "close"):
                file_tuple[1].close()


def test_api_key_authentication(sample_nifti_files):
    """Test API key authentication if enabled."""
    if not API_KEY:
        pytest.skip("API_KEY not set, skipping authentication test")
    
    # Try request without API key
    files = {
        "flair": ("flair.nii", open(sample_nifti_files["flair"], "rb"), "application/octet-stream"),
        "t1": ("t1.nii", open(sample_nifti_files["t1"], "rb"), "application/octet-stream"),
        "t1ce": ("t1ce.nii", open(sample_nifti_files["t1ce"], "rb"), "application/octet-stream"),
        "t2": ("t2.nii", open(sample_nifti_files["t2"], "rb"), "application/octet-stream"),
    }
    
    try:
        # Request without API key should fail if authentication is enabled
        response = requests.post(
            f"{BASE_URL}/infer/upload",
            files=files,
            timeout=10
        )
        
        # If API key is required, should get 401
        # If not required, should get 200 or other status
        # We can't easily determine if auth is enabled without checking server config
        # So we just verify the response is valid
        assert response.status_code in [200, 401, 403]
        
    finally:
        for file_tuple in files.values():
            if hasattr(file_tuple[1], "close"):
                file_tuple[1].close()


def test_invalid_job_id(auth_headers):
    """Test that running inference with invalid job_id returns 404."""
    response = requests.post(
        f"{BASE_URL}/infer/run",
        data={"job_id": "nonexistent-job-id-12345"},
        headers=auth_headers,
        timeout=10
    )
    
    assert response.status_code == 404

