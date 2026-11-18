"""
frequency_branch.py — Patched Native-Resolution 2D-FFT Spectral Branch
======================================================================
Fixes over v1:
  1. Removes destructive per-sample min-max normalization to [0, 1] inside HighPassFFTLayer
     (which previously collapsed real vs. AI frequency energy to ~0.49 vs ~0.50).
     Instead uses log1p(|F_high|) + global BatchNorm2d normalization to preserve true spectral amplitude.
  2. Extracts frequency residuals on NATIVE-RESOLUTION crops (center or multi-crop) instead of
     bilinear downsampling to 256x256, preserving high-frequency grid/latent artifacts on 1024px FLUX images.
  3. Upgrades compute_fft_power_spectrum(pil_image) to visualize native residual 2D-FFT.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from native_forensics import compute_native_2d_fft


class HighPassFFTLayer(nn.Module):
    """
    Amplitude-preserving 2D-FFT High-Pass Filtering Layer.
    Outputs a 2-channel tensor [B, 2, H, W]:
      - Channel 0: High-pass reconstructed spatial residual (log-compressed signed amplitude)
      - Channel 1: Shifted log1p(2D-FFT high-pass magnitude) preserving absolute spectral energy
    """

    def __init__(self, r0: int = 16):
        super().__init__()
        self.r0 = r0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] float tensor in [0, 1] (at native crop resolution)
        Returns:
            [B, 2, H, W] amplitude-preserving residual + log-spectrum tensor
        """
        B, C, H, W = x.shape

        # Convert RGB -> Luminance
        if C == 3:
            weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            gray = (x * weights).sum(dim=1)  # [B, H, W]
        else:
            gray = x[:, 0]

        # Subtract local mean to suppress DC / scene brightness before FFT
        gray = gray - gray.mean(dim=(-2, -1), keepdim=True)

        # 2D FFT -> shift DC to center
        fft_shifted = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))

        # Radial high-pass mask scaled proportionally if H != 256
        cy, cx = H // 2, W // 2
        r0_scaled = max(4, int(self.r0 * min(H, W) / 256.0))
        yy = torch.arange(H, device=x.device, dtype=torch.float32) - cy
        xx = torch.arange(W, device=x.device, dtype=torch.float32) - cx
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        dist = torch.hypot(grid_x, grid_y)
        mask = (dist >= r0_scaled).float().unsqueeze(0)  # [1, H, W]

        fft_filtered = fft_shifted * mask

        # Channel 0: High-pass spatial residual WITHOUT per-sample min-max collapse
        recon = torch.fft.ifft2(torch.fft.ifftshift(fft_filtered, dim=(-2, -1))).real
        recon_log = torch.sign(recon) * torch.log1p(torch.abs(recon) * 25.0)

        # Channel 1: Global-scale log1p FFT magnitude (preserves real 1/f vs AI spectral energy gap)
        spec_mag = torch.log1p(torch.abs(fft_filtered))

        return torch.stack([recon_log, spec_mag], dim=1)  # [B, 2, H, W]


class FrequencyBranch(nn.Module):
    """
    Native-resolution Frequency Branch:
      HighPassFFTLayer (2ch) -> BatchNorm2d -> 3-layer ConvNet -> AdaptiveAvgPool -> Linear(128, out_features)
    """

    def __init__(self, r0: int = 16, out_features: int = 128, native_crop_size: int = 128):
        super().__init__()
        self.native_crop_size = native_crop_size
        self.hpf = HighPassFFTLayer(r0=r0)

        # Global learned normalization across dataset (replaces per-sample min-max!)
        self.input_bn = nn.BatchNorm2d(2)

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, out_features)
        self.out_features = out_features

    def _center_crop_native(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        cs = self.native_crop_size
        if H == cs and W == cs:
            return x
        if H < cs or W < cs:
            pad_h = max(0, cs - H)
            pad_w = max(0, cs - W)
            x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode="reflect")
            _, _, H, W = x.shape
        y0 = (H - cs) // 2
        x0 = (W - cs) // 2
        return x[:, :, y0 : y0 + cs, x0 : x0 + cs]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] at native image resolution (not bilinear-resized!)
        Returns:
            feat: [B, out_features]
        """
        x_crop = self._center_crop_native(x)
        hpf_2ch = self.hpf(x_crop)              # [B, 2, cs, cs]
        hpf_norm = self.input_bn(hpf_2ch)       # Global BN preserves inter-sample amplitude gap
        feat_map = self.cnn(hpf_norm)           # [B, 128, cs/4, cs/4]
        pooled = self.pool(feat_map).flatten(1) # [B, 128]
        return self.fc(pooled)

    def get_hpf_visualization(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            hpf_2ch = self.hpf(self._center_crop_native(x))
            arr = torch.abs(hpf_2ch[0, 0]).cpu().numpy()
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr


def compute_fft_power_spectrum(pil_image, size: int = 128) -> np.ndarray:
    """
    Computes 2D-FFT log-magnitude spectrum on a high-pass denoised NATIVE patch
    (without bilinear resizing that erases generative upsampling peaks).
    """
    rgb = np.array(pil_image.convert("RGB"), dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    _, log_uint8 = compute_native_2d_fft(gray, max_patch_size=size)
    return log_uint8


if __name__ == "__main__":
    print("Patched FrequencyBranch — standalone check")
    model = FrequencyBranch(r0=16, out_features=128, native_crop_size=128)
    dummy_highres = torch.rand(2, 3, 512, 512)
    out = model(dummy_highres)
    print(f"Input shape: {dummy_highres.shape} -> Output shape: {out.shape}")
    assert out.shape == (2, 128)
    print("OK — amplitude-preserving FrequencyBranch works.")
