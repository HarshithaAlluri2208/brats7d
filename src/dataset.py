# src/dataset.py
import os
import glob
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    """
    BraTS Dataset loader with MRI modalities, segmentation, and generated maps.
    Supports distance + boundary maps stored in separate folders.
    """

    def __init__(
        self,
        data_dir,
        boundary_map_dir=None,
        dist_map_dir=None,
        patch_size=None,
        tumor_patch_prob=0.5,
        transforms=None,
        has_labels=True,
        extra_channels=True
    ):
        """
        Args:
            data_dir (str): Path to BraTS dataset (folder with subject subfolders).
            boundary_map_dir (str): Path to folder with *_boundary.npy files.
            dist_map_dir (str): Path to folder with *_dist.npy files.
            patch_size (tuple[int], optional): e.g. (96,96,96). None = full volume.
            tumor_patch_prob (float): Probability of sampling tumor patches.
            transforms (callable, optional): Optional preprocessing transforms.
            has_labels (bool): Whether to load segmentation labels.
        """
        self.data_dir = data_dir
        self.boundary_map_dir = boundary_map_dir
        self.dist_map_dir = dist_map_dir
        self.has_labels = has_labels
        self.transforms = transforms
        self.patch_size = np.array(patch_size) if patch_size else None
        self.tumor_patch_prob = tumor_patch_prob
        self.extra_channels = extra_channels

        # Detect subjects
        self.subjects = [
            d
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and "BraTS20" in d
        ]
        self.subjects.sort()
        print(f"[BraTSDataset] Found {len(self.subjects)} subjects in {data_dir}")

        # Check for generated map folders
        if boundary_map_dir and dist_map_dir:
            self.extra_channels = True
            print(f"[BraTSDataset] ✅ Using extra channels (boundary + distance maps).")
            print(f"   Boundary dir: {boundary_map_dir}")
            print(f"   Distance dir: {dist_map_dir}")
        else:
            self.extra_channels = False
            print(f"[BraTSDataset] ⚠️ No extra channels — only MRI modalities will be loaded.")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        subject_folder = os.path.join(self.data_dir, subject_id)

        # --- Load 4 MRI modalities (flair, t1, t1ce, t2) ---
        flair = os.path.join(subject_folder, f"{subject_id}_flair.nii")
        t1 = os.path.join(subject_folder, f"{subject_id}_t1.nii")
        t1ce = os.path.join(subject_folder, f"{subject_id}_t1ce.nii")
        t2 = os.path.join(subject_folder, f"{subject_id}_t2.nii")

        # load volumes and ensure float32
        try:
            flair_arr = nib.load(flair).get_fdata(dtype=np.float32)
            t1_arr = nib.load(t1).get_fdata(dtype=np.float32)
            t1ce_arr = nib.load(t1ce).get_fdata(dtype=np.float32)
            t2_arr = nib.load(t2).get_fdata(dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"[BraTSDataset] Failed to load MRI files for {subject_id}: {e}")

        img = np.stack([
            flair_arr,
            t1_arr,
            t1ce_arr,
            t2_arr,
        ]).astype(np.float32)

        # normalize only the MRI channels (first 4)
        img[:4] = self.zscore_normalize(img[:4])

        # --- Load distance and boundary maps (single-channel expected) ---
        dist_path = os.path.join(self.dist_map_dir, f"{subject_id}_dist.npy") if self.dist_map_dir else None
        boundary_path = os.path.join(self.boundary_map_dir, f"{subject_id}_boundary.npy") if self.boundary_map_dir else None

        if dist_path and boundary_path and os.path.exists(dist_path) and os.path.exists(boundary_path):
            dist_map = np.load(dist_path)
            boundary_map = np.load(boundary_path)
        else:
            if not (dist_path and boundary_path):
                print(f"[⚠️ Warning] Map dirs not provided for {subject_id}. Using zero-filled maps.")
            else:
                print(f"[⚠️ Warning] Missing maps for {subject_id}. Using zero-filled maps.")
            dist_map = np.zeros_like(img[0], dtype=np.float32)
            boundary_map = np.zeros_like(img[0], dtype=np.float32)

        # --- Defensive handling: if maps have extra channel dims, pick the first channel and warn ---
        # Acceptable shape for maps: (D, H, W) i.e. 3D. If 4D (C, D, H, W) or (D, H, W, C), try to reduce.
        try:
            # handle dist_map
            if dist_map.ndim == 4:
                # assume (C, D, H, W) -> pick first channel
                print(f"[⚠️ Warning] {subject_id} dist_map has 4 dims {dist_map.shape}. Using first channel.")
                dist_map = dist_map[0]
            elif dist_map.ndim == 3:
                pass
            elif dist_map.ndim == 2:
                # unexpected, expand dims
                dist_map = np.expand_dims(dist_map, axis=0)
                print(f"[⚠️ Warning] {subject_id} dist_map has 2 dims, expanded to 3D: {dist_map.shape}.")
            else:
                # fallback to zeros if strange
                print(f"[⚠️ Warning] {subject_id} dist_map has unexpected shape {getattr(dist_map,'shape',None)}. Using zeros.")
                dist_map = np.zeros_like(img[0], dtype=np.float32)

            # handle boundary_map
            if boundary_map.ndim == 4:
                print(f"[⚠️ Warning] {subject_id} boundary_map has 4 dims {boundary_map.shape}. Using first channel.")
                boundary_map = boundary_map[0]
            elif boundary_map.ndim == 3:
                pass
            elif boundary_map.ndim == 2:
                boundary_map = np.expand_dims(boundary_map, axis=0)
                print(f"[⚠️ Warning] {subject_id} boundary_map has 2 dims, expanded to 3D: {boundary_map.shape}.")
            else:
                print(f"[⚠️ Warning] {subject_id} boundary_map has unexpected shape {getattr(boundary_map,'shape',None)}. Using zeros.")
                boundary_map = np.zeros_like(img[0], dtype=np.float32)
        except Exception as e:
            print(f"[⚠️ Warning] Error while processing maps for {subject_id}: {e}. Using zero maps.")
            dist_map = np.zeros_like(img[0], dtype=np.float32)
            boundary_map = np.zeros_like(img[0], dtype=np.float32)

        # ensure float32 dtype
        dist_map = dist_map.astype(np.float32)
        boundary_map = boundary_map.astype(np.float32)

        # Ensure maps have shape (D, H, W). If they are (H, W, D) or other order, user must ensure consistency.
        # Now stack exactly once: (4 MRI) + (dist) + (boundary) -> 6 channels
        img = np.concatenate([
            img,                         # shape (4, D, H, W)
            dist_map[None, ...],         # shape (1, D, H, W)
            boundary_map[None, ...]      # shape (1, D, H, W)
        ], axis=0).astype(np.float32)

        # Safety check: enforce 6 channels no matter what
        if img.shape[0] < 6:
            print(f"[⚠️ AutoFix] {subject_id} has only {img.shape[0]} channels. Padding with zeros.")
            pad_ch = 6 - img.shape[0]
            pad_maps = np.zeros((pad_ch, *img.shape[1:]), dtype=np.float32)
            img = np.concatenate([img, pad_maps], axis=0)
        elif img.shape[0] > 6:
            print(f"[⚠️ Warning] {subject_id} has {img.shape[0]} channels. Truncating to 6.")
            img = img[:6, ...]

        assert img.shape[0] == 6, f"[❌ ERROR] {subject_id}: final image has {img.shape[0]} channels!"

        # --- Load segmentation if available ---
        seg_path = os.path.join(subject_folder, f"{subject_id}_seg.nii")

        if self.has_labels:
            seg_arr = nib.load(seg_path).get_fdata(dtype=np.float32).astype(np.int64)
            seg_arr[seg_arr == 4] = 3
            return torch.from_numpy(img).float(), torch.from_numpy(seg_arr).long()
        else:
            return torch.from_numpy(img).float()

    def zscore_normalize(self, img):
        for c in range(img.shape[0]):
            channel = img[c]
            nonzero = channel[channel > 0]
            if nonzero.size > 0:
                mean = nonzero.mean()
                std = nonzero.std()
                if std > 0:
                    img[c] = (channel - mean) / std
        return img

    def random_patch(self, img, seg):
        C, D, H, W = img.shape
        pd, ph, pw = self.patch_size

        if (seg > 0).any() and np.random.rand() < self.tumor_patch_prob:
            zs, ys, xs = np.where(seg > 0)
            i = np.random.randint(len(zs))
            cz, cy, cx = zs[i], ys[i], xs[i]
        else:
            cz, cy, cx = np.random.randint(0, D), np.random.randint(0, H), np.random.randint(0, W)

        z0, y0, x0 = max(0, cz - pd // 2), max(0, cy - ph // 2), max(0, cx - pw // 2)
        z1, y1, x1 = z0 + pd, y0 + ph, x0 + pw

        img_patch = img[:, z0:z1, y0:y1, x0:x1]
        seg_patch = seg[z0:z1, y0:y1, x0:x1]
        return img_patch, seg_patch
