import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, binary_erosion

# ---------------------------
# 1️⃣ Utility Functions
# ---------------------------

def generate_distance_map(gt_data):
    """Generate a distance map from a binary ground truth mask."""
    binary_mask = (gt_data > 0).astype(np.uint8)
    dist_map = distance_transform_edt(binary_mask)
    return dist_map


def generate_boundary_map(gt_data):
    """Generate a boundary map from a binary ground truth mask."""
    binary_mask = (gt_data > 0).astype(np.uint8)
    eroded = binary_erosion(binary_mask)
    boundary = binary_mask - eroded
    return boundary


# ---------------------------
# 2️⃣ Subject-Level Processing
# ---------------------------

def process_subject(subject_folder, save_folder):
    """
    Generate distance and boundary maps for a single subject.
    """
    os.makedirs(save_folder, exist_ok=True)

    seg_files = [
        f for f in os.listdir(subject_folder)
        if 'seg' in f and f.endswith(('.nii', '.nii.gz'))
    ]
    if not seg_files:
        print(f"⚠️ No segmentation file found in {subject_folder}")
        return

    gt_path = os.path.join(subject_folder, seg_files[0])
    subject_id = os.path.basename(subject_folder)

    gt_nii = nib.load(gt_path)
    gt_data = gt_nii.get_fdata().astype(np.int64)
    gt_data[gt_data == 4] = 3  # map label 4 → 3

    dist_map = generate_distance_map(gt_data)
    boundary_map = generate_boundary_map(gt_data)

    np.save(os.path.join(save_folder, f"{subject_id}_dist.npy"), dist_map)
    np.save(os.path.join(save_folder, f"{subject_id}_boundary.npy"), boundary_map)
    print(f"✅ Processed: {subject_id}")


# ---------------------------
# 3️⃣ Bulk Processing
# ---------------------------

def process_all(data_root, save_folder):
    """
    Process all subject folders in a dataset root and save generated maps.
    """
    os.makedirs(save_folder, exist_ok=True)

    subject_folders = [
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ]

    for subj_path in subject_folders:
        process_subject(subj_path, save_folder)


# ---------------------------
# 4️⃣ Main Entry Point for Colab
# ---------------------------

if __name__ == "__main__":
    # Define paths (modify only these two if needed)
    data_root = "/content/drive/MyDrive/brats7d/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    save_folder = "/content/drive/MyDrive/brats7d/data/processed_maps"

    print(f"🏁 Starting map generation...")
    print(f"📂 Input folder: {data_root}")
    print(f"💾 Output folder: {save_folder}")

    process_all(data_root, save_folder)

    print("🎯 Done! All subjects processed.")
