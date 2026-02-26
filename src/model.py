# src/model.py
import torch
import torch.nn as nn
from monai.networks.nets import UNet


def get_unet_model(in_channels=6, out_channels=4, base_filters=32):
    """
    Returns a 3D U-Net model from MONAI, configured for BraTS + generated channels.
    """
    print(f"[INFO] Initializing 3D U-Net with {in_channels} input channels and {out_channels} output classes")

    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,      # ✅ supports 6 now
        out_channels=out_channels,
        channels=(base_filters, base_filters*2, base_filters*4, base_filters*8, base_filters*16),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1,
    )

    return model


if __name__ == "__main__":
    # Test
    model = get_unet_model()
    x = torch.randn(1, 6, 128, 128, 128)
    y = model(x)
    print("Output shape:", y.shape)
