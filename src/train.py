
import os
import yaml
import csv
import torch
import torch.nn.functional as F
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from torch.cuda.amp import GradScaler, autocast
from src.dataset import BraTSDataset

def train_model(config_path):
    """Train or resume a 3D U-Net model for BraTS segmentation."""

    # -------------------------------------------------------------------------
    # 1. Load config
    # -------------------------------------------------------------------------
    print(f"[INFO] Loading config from: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Dataset setup
    # -------------------------------------------------------------------------
    data_dir = cfg.get("train_dir") or cfg.get("data_dir")
    boundary_map_dir = cfg.get("boundary_map_dir")
    dist_map_dir = cfg.get("dist_map_dir")

    train_ds = BraTSDataset(
        data_dir=data_dir,
        boundary_map_dir=boundary_map_dir,
        dist_map_dir=dist_map_dir,
        has_labels=True,
        extra_channels=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=True,
    )

    print(f"[INFO] Dataset size: {len(train_ds)} subjects. Dataloader batch_size={cfg.get('batch_size',1)}")

    # -------------------------------------------------------------------------
    # 3. Model setup
    # -------------------------------------------------------------------------
    in_ch = int(cfg.get("in_channels", 6))
    out_ch = int(cfg.get("out_channels", 4))
    model = UNet(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=out_ch,
        channels=tuple(cfg.get("unet_channels", (32, 64, 128, 256, 512))),
        strides=tuple(cfg.get("unet_strides", (2, 2, 2, 2))),
        num_res_units=cfg.get("num_res_units", 2),
    ).to(device)

    # -------------------------------------------------------------------------
    # 4. Numeric-safe hyperparams
    # -------------------------------------------------------------------------
    lr_val = cfg.get("lr", cfg.get("learning_rate", 1e-4))
    wd_val = cfg.get("weight_decay", 1e-5)
    try:
        lr_val = float(lr_val)
    except Exception:
        lr_val = 1e-4
    try:
        wd_val = float(wd_val)
    except Exception:
        wd_val = 1e-5

    print(f"[INFO] Learning rate: {lr_val}, Weight decay: {wd_val}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=wd_val)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    scaler = GradScaler()

    # -------------------------------------------------------------------------
    # 5. Resume checkpoint (if any)
    # -------------------------------------------------------------------------
    start_epoch = 0
    resume_ckpt = cfg.get("resume_checkpoint")
    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"[INFO] Loading checkpoint from {resume_ckpt}")
        checkpoint = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        start_epoch = checkpoint.get("epoch", 0)
        print(f"[INFO] Resumed from epoch {start_epoch}")
    else:
        print("[INFO] No checkpoint found — starting fresh.")

    # -------------------------------------------------------------------------
    # 6. Logging & checkpoint directories
    # -------------------------------------------------------------------------
    models_dir = cfg.get("models_dir", "models")
    logs_dir = cfg.get("logs_dir", "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(log_path, "w", newline="") as f_log:
        writer = csv.writer(f_log)
        writer.writerow(["epoch", "avg_loss", "learning_rate"])
    print(f"[INFO] Logging to: {log_path}")

    # -------------------------------------------------------------------------
    # 7. Training loop
    # -------------------------------------------------------------------------
    max_epochs = int(cfg.get("max_epochs", cfg.get("epochs", 50)))
    print(f"[INFO] Starting training for {max_epochs} epochs...")

    for epoch in range(start_epoch + 1, max_epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}")
        for batch in pbar:
            imgs, segs = batch
            imgs, segs = imgs.to(device), segs.to(device)

            # --- Pad to multiple of 16 for UNet (preserves skip-connection sizes) ---
            B, C, D, H, W = imgs.shape
            multiple = 16
            pd = (multiple - (D % multiple)) % multiple
            ph = (multiple - (H % multiple)) % multiple
            pw = (multiple - (W % multiple)) % multiple
            pad = (0, pw, 0, ph, 0, pd)

            imgs_p = F.pad(imgs, pad)
            segs_p = F.pad(segs.unsqueeze(1), pad).long()

            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs_p)
                loss = criterion(outputs, segs_p)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(train_loader)
        print(f"🧠 Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

        # --- Logging ---
        with open(log_path, "a", newline="") as f_log:
            writer = csv.writer(f_log)
            writer.writerow([epoch, avg_loss, lr_val])
            f_log.flush()
            os.fsync(f_log.fileno())

        # --- Save checkpoint after every epoch ---
        ckpt_path = os.path.join(models_dir, f"checkpoint_epoch{epoch}.pth")
        latest_path = os.path.join(models_dir, "latest.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }, ckpt_path)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }, latest_path)
        print(f"[INFO] Saved checkpoint: {ckpt_path}")

    print("[✅] Training completed successfully!")

if __name__ == "__main__":
    cfg_path = "/content/drive/MyDrive/brats7d/configs/resume_fixed.yaml"
    train_model(cfg_path)
