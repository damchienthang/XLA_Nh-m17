"""
gradcam.py — Explainable AI Localization (PyTorch Grad-CAM + Native Local Anomaly Map)
======================================================================================
Provides:
  1. GradCAMExplainer: Backward-hook Grad-CAM on VAEReconEncoder.layer2[-1] (when PyTorch is active).
  2. NativeAnomalyLocalizer: Fast gradient-free 16x16 local chroma-residual & uniform-noise map
     that runs on any CPU without needing VAE backward passes.
"""

from typing import Optional
import cv2
import numpy as np
from PIL import Image

from native_forensics import extract_multi_domain_features

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


class NativeAnomalyLocalizer:
    """Fast CPU-friendly forensic heatmap localizer based on local Chroma/Luma & noise uniformity."""

    def explain(
        self,
        pil_original: Image.Image,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.48,
        gain: float = 1.0,
    ) -> Image.Image:
        orig_np = np.array(pil_original.convert("RGB"))
        feat = extract_multi_domain_features(orig_np)
        grid = feat["gradcam_grid_16"]
        h, w, _ = orig_np.shape
        up_grid = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
        up_norm = np.clip(up_grid * gain, 0.0, 1.0)
        heat_bgr = cv2.applyColorMap((up_norm * 255).astype(np.uint8), colormap)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(orig_np, 1.0 - alpha, heat_rgb, alpha, 0)
        return Image.fromarray(blended)


if HAS_TORCH:
    class GradCAMExplainer:
        """Grad-CAM for VAEReconEncoder (ResNet-50 layer2[-1]) in CO-SPY."""

        def __init__(self, artifact_encoder: nn.Module, fc_layer: nn.Module, target_layer: Optional[nn.Module] = None):
            self.encoder = artifact_encoder
            self.fc = fc_layer
            self.target_layer = target_layer if target_layer is not None else artifact_encoder.layer2[-1]
            self._feature_maps = None
            self._gradients = None
            self._hooks = []

        def _register_hooks(self):
            def _fwd_hook(module, _inp, output):
                self._feature_maps = output.detach().clone()

            def _bwd_hook(module, _grad_in, grad_out):
                self._gradients = grad_out[0].detach().clone()

            self._hooks = [
                self.target_layer.register_forward_hook(_fwd_hook),
                self.target_layer.register_full_backward_hook(_bwd_hook),
            ]

        def _remove_hooks(self):
            for h in self._hooks:
                h.remove()
            self._hooks = []

        def get_raw_cam(self, tensor_input: "torch.Tensor", output_size: tuple = (224, 224)) -> np.ndarray:
            self.encoder.eval()
            self.fc.eval()
            self._register_hooks()
            x = tensor_input.clone().requires_grad_(True)
            try:
                feat = self.encoder(x)
                score = self.fc(feat)
                self.encoder.zero_grad()
                if self.fc.weight.grad is not None:
                    self.fc.zero_grad()
                score.backward(torch.ones_like(score))
            finally:
                self._remove_hooks()

            weights = self._gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self._feature_maps).sum(dim=1).squeeze(0)).cpu().numpy()
            if cam.max() > 0:
                cam = cam / cam.max()
            return cv2.resize(cam, output_size, interpolation=cv2.INTER_LINEAR)

        def explain(
            self,
            tensor_input: "torch.Tensor",
            pil_original: Image.Image,
            colormap: int = cv2.COLORMAP_JET,
            alpha: float = 0.45,
        ) -> Image.Image:
            w_orig, h_orig = pil_original.size
            cam = self.get_raw_cam(tensor_input, output_size=(w_orig, h_orig))
            heat_bgr = cv2.applyColorMap((cam * 255).astype(np.uint8), colormap)
            heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
            orig_np = np.array(pil_original.convert("RGB"))
            overlay = cv2.addWeighted(orig_np, 1.0 - alpha, heat_rgb, alpha, 0)
            return Image.fromarray(overlay)
else:
    GradCAMExplainer = NativeAnomalyLocalizer
