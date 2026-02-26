# src/infer.py
import torch
import numpy as np
import torch.nn.functional as F
from monai.networks.nets import UNet
import nibabel as nib
import os

def load_model(device="cuda"):
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16,32,64,128,256),
        strides=(2,2,2,2),
        num_res_units=2,
    ).to(device)
    model.eval()
    return model

def predict(model, image_paths, device="cuda"):
    # Load images as numpy arrays
    imgs = [nib.load(p).get_fdata() for p in image_paths]  # list of [H,W,D]
    img_tensor = torch.tensor(np.stack(imgs, axis=0)[None,...], dtype=torch.float32).to(device)
    
    # Reorder channels: [B,C,H,W,D]
    img_tensor = img_tensor.permute(0,1,2,3,4)
    
    # Pad depth if needed
    D = img_tensor.shape[-1]
    if D % 16 != 0:
        pad_d = 16 - D % 16
        img_tensor = F.pad(img_tensor, (0,pad_d,0,0,0,0))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)
        pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]
    
    return pred  # [H,W,D] label map

def save_seg(pred, out_path, ref_nii_path):
    ref = nib.load(ref_nii_path)
    seg_nii = nib.Nifti1Image(pred.astype(np.uint8), ref.affine, ref.header)
    nib.save(seg_nii, out_path)
    print(f"Segmentation saved to {out_path}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    
    # Example usage
    example_images = [
        "G:/My Drive/brats7d/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii",
        "G:/My Drive/brats7d/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii",
        "G:/My Drive/brats7d/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii",
        "G:/My Drive/brats7d/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t2.nii"
    ]
    pred = predict(model, example_images, device)
    save_seg(pred, "pred_seg.nii.gz", example_images[0])
