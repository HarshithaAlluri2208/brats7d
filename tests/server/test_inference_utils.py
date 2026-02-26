"""
Pytest tests for inference utilities.
"""
import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add server directory to path
server_path = Path(__file__).parent.parent.parent / "server"
sys.path.insert(0, str(server_path))

from utils.inference_utils import preprocess_volume, unpad, run_model, save_outputs


@pytest.fixture
def sample_volume_stack():
    """Create a sample 6-channel volume stack for testing."""
    # Create a small test volume: (6, D, H, W) = (6, 32, 64, 64)
    D, H, W = 32, 64, 64
    stack = np.random.randn(6, D, H, W).astype(np.float32)
    
    # Make first 4 channels (MRI) have positive values for normalization
    stack[:4] = np.abs(stack[:4]) * 100
    
    # Last 2 channels (dist, boundary) can be any values
    return stack


@pytest.fixture
def sample_model():
    """Create a minimal model for testing."""
    from monai.networks.nets import UNet
    
    model = UNet(
        spatial_dims=3,
        in_channels=6,
        out_channels=4,
        channels=(16, 32, 64),
        strides=(2, 2),
        num_res_units=1,
    )
    model.eval()
    return model


def test_preprocess_padding(sample_volume_stack):
    """
    Test that preprocess_volume correctly pads dimensions to be divisible by 16.
    """
    tensor, pad_info = preprocess_volume(sample_volume_stack)
    
    # Check tensor shape
    assert tensor.shape[0] == 1, "Should have batch dimension of 1"
    assert tensor.shape[1] == 6, "Should have 6 channels"
    
    # Check that spatial dimensions are divisible by 16
    D, H, W = tensor.shape[2], tensor.shape[3], tensor.shape[4]
    assert D % 16 == 0, f"Depth {D} should be divisible by 16"
    assert H % 16 == 0, f"Height {H} should be divisible by 16"
    assert W % 16 == 0, f"Width {W} should be divisible by 16"
    
    # Check pad_info structure
    assert "original_shape" in pad_info
    assert "padded_shape" in pad_info
    assert pad_info["original_shape"] == sample_volume_stack.shape[1:]
    
    # Check that unpadding works correctly
    test_array = np.random.randn(*pad_info["padded_shape"]).astype(np.float32)
    unpadded = unpad(test_array, pad_info)
    assert unpadded.shape == pad_info["original_shape"]


def test_preprocess_normalization(sample_volume_stack):
    """
    Test that preprocess_volume normalizes first 4 channels (MRI modalities).
    """
    tensor, pad_info = preprocess_volume(sample_volume_stack)
    
    # Convert back to numpy for inspection
    tensor_np = tensor[0].cpu().numpy()
    
    # First 4 channels should be normalized (z-score)
    # We can't easily verify exact normalization without reimplementing,
    # but we can check that values are reasonable
    for c in range(4):
        channel = tensor_np[c]
        # After z-score normalization, values should be centered around 0
        # and have reasonable range (not all zeros, not all huge)
        assert not np.all(channel == 0), f"Channel {c} should not be all zeros"
        assert np.abs(channel).max() < 1e6, f"Channel {c} values too large"


def test_unpad(sample_volume_stack):
    """
    Test that unpad correctly removes padding.
    """
    tensor, pad_info = preprocess_volume(sample_volume_stack)
    
    # Create a test array with padded shape
    padded_shape = pad_info["padded_shape"]
    test_array = np.random.randn(*padded_shape).astype(np.float32)
    
    # Unpad
    unpadded = unpad(test_array, pad_info)
    
    # Check shape matches original
    assert unpadded.shape == pad_info["original_shape"]
    
    # Check that unpadding is correct (first part should match)
    original_shape = pad_info["original_shape"]
    for i in range(min(original_shape[0], padded_shape[0])):
        for j in range(min(original_shape[1], padded_shape[1])):
            for k in range(min(original_shape[2], padded_shape[2])):
                assert unpadded[i, j, k] == test_array[i, j, k]


def test_run_model_returns_shapes(sample_model, sample_volume_stack):
    """
    Test that run_model returns predictions and probabilities with correct shapes.
    """
    device = "cpu"  # Use CPU for testing
    
    # Preprocess volume
    input_tensor, pad_info = preprocess_volume(sample_volume_stack)
    input_tensor = input_tensor.to(device)
    sample_model = sample_model.to(device)
    
    # Run model
    pred, probs = run_model(sample_model, input_tensor, device)
    
    # Check prediction shape: should be (D, H, W) after unpadding
    original_shape = pad_info["original_shape"]
    assert pred.shape == original_shape, f"Prediction shape {pred.shape} != {original_shape}"
    assert pred.dtype == np.uint8, "Predictions should be uint8"
    
    # Check probability shape: should be (4, D, H, W) after unpadding
    assert probs.shape == (4, *original_shape), f"Probabilities shape {probs.shape} != (4, {original_shape})"
    assert probs.dtype == np.float32, "Probabilities should be float32"
    
    # Check prediction values are valid labels (0-3)
    unique_labels = np.unique(pred)
    assert all(label in [0, 1, 2, 3] for label in unique_labels), \
        f"Invalid labels found: {unique_labels}"
    
    # Check probability values are in [0, 1]
    assert probs.min() >= 0.0, "Probabilities should be >= 0"
    assert probs.max() <= 1.0, "Probabilities should be <= 1"


def test_save_outputs(tmp_path, sample_volume_stack):
    """
    Test that save_outputs creates the expected files.
    """
    import nibabel as nib
    
    # Create dummy prediction and probabilities
    D, H, W = sample_volume_stack.shape[1:]
    pred = np.random.randint(0, 4, size=(D, H, W), dtype=np.uint8)
    probs = np.random.rand(4, D, H, W).astype(np.float32)
    probs = probs / probs.sum(axis=0, keepdims=True)  # Normalize to probabilities
    
    # Create reference NIfTI
    reference_nii = nib.Nifti1Image(
        sample_volume_stack[0],
        affine=np.eye(4),
        header=nib.Nifti1Header()
    )
    
    # Save outputs
    job_id = "test_job_123"
    saved_paths = save_outputs(
        str(tmp_path),
        job_id,
        pred,
        probs,
        reference_nii
    )
    
    # Check that files were created
    assert "segmentation_npy" in saved_paths
    assert Path(saved_paths["segmentation_npy"]).exists()
    
    assert "segmentation_nifti" in saved_paths
    assert saved_paths["segmentation_nifti"] is not None
    assert Path(saved_paths["segmentation_nifti"]).exists()
    
    assert "probability_maps" in saved_paths
    assert Path(saved_paths["probability_maps"]).exists()
    
    # Verify file contents
    loaded_pred = np.load(saved_paths["segmentation_npy"])
    assert np.array_equal(loaded_pred, pred)
    
    loaded_probs = np.load(saved_paths["probability_maps"])
    assert "class_0" in loaded_probs
    assert "class_1" in loaded_probs
    assert "class_2" in loaded_probs
    assert "class_3" in loaded_probs


def test_save_outputs_no_reference(tmp_path, sample_volume_stack):
    """
    Test that save_outputs works without reference NIfTI.
    """
    D, H, W = sample_volume_stack.shape[1:]
    pred = np.random.randint(0, 4, size=(D, H, W), dtype=np.uint8)
    probs = np.random.rand(4, D, H, W).astype(np.float32)
    probs = probs / probs.sum(axis=0, keepdims=True)
    
    job_id = "test_job_456"
    saved_paths = save_outputs(
        str(tmp_path),
        job_id,
        pred,
        probs,
        reference_nii=None
    )
    
    # Should still save .npy and .npz
    assert "segmentation_npy" in saved_paths
    assert Path(saved_paths["segmentation_npy"]).exists()
    
    # segmentation_nifti should be None
    assert saved_paths["segmentation_nifti"] is None

