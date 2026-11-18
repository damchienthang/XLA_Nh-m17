"""
engine.py — Repaired CO-SPY Forensic Inference Engine (Full PyTorch + Lightweight Standalone)
===========================================================================================
Supports:
  1. Full CO-SPY PyTorch mode (End2EndDetector SigLIP-384 + VAE + trained EnhancedCoSpyHead)
     when Co-Spy weights and PyTorch are installed.
  2. Calibrated Lightweight Native Multi-Domain CAGU mode (NumPy + OpenCV) automatically when
     PyTorch/CompVis weights are absent or when fast CPU inference is preferred.
  3. Side-by-side Baseline (v1 256px FFT heuristic) vs Enhanced v2 comparison for Streamlit UI & CLI.
"""

import importlib
import math
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from native_forensics import predict_multi_domain_cagu

try:
    import torch
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    torch = None
    HAS_TORCH = False
    DEVICE = "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CO_SPY_DIR = os.path.join(BASE_DIR, "Co-Spy")
if os.path.isdir(CO_SPY_DIR) and CO_SPY_DIR not in sys.path:
    sys.path.insert(0, CO_SPY_DIR)


def _sigmoid(z: float) -> float:
    if z > 20.0:
        return 1.0
    if z < -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _compute_legacy_baseline_v1(pil_image: Image.Image) -> dict:
    """
    Reproduces exact engine.py v1 heuristic (256x256 bilinear FFT + blur MSE)
    so the UI and benchmark can compare Baseline v1 vs Enhanced v2 side-by-side.
    """
    rgb = np.array(pil_image.convert("RGB"), dtype=np.float64)
    gray_full = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # Spatial 3x3 blur reconstruction MSE on [0, 1]
    h, w = gray_full.shape
    if h >= 3 and w >= 3:
        blur = (
            gray_full[1:-1, :-2]
            + gray_full[1:-1, 2:]
            + gray_full[:-2, 1:-1]
            + gray_full[2:, 1:-1]
            + 4.0 * gray_full[1:-1, 1:-1]
        ) / 8.0
        mse = float(np.mean(((gray_full[1:-1, 1:-1] - blur) / 255.0) ** 2))
    else:
        mse = 0.001

    np_gray256 = np.array(pil_image.convert("L").resize((256, 256)), dtype=np.float32)
    f_mat = np.fft.fftshift(np.fft.fft2(np_gray256))
    mag_mat = np.abs(f_mat)
    log_mag = np.log(mag_mat + 1e-7)

    cy, cx = 128, 128
    y_grid, x_grid = np.ogrid[-cy : 256 - cy, -cx : 256 - cx]
    r_dist = np.hypot(x_grid, y_grid).astype(int)

    outer_e = float(np.mean(mag_mat[r_dist > 70]))
    inner_e = float(np.mean(mag_mat[(r_dist > 15) & (r_dist <= 70)]))
    f_ratio = outer_e / (inner_e + 1e-6)

    log_no_dc = log_mag.copy()
    log_no_dc[cy - 12 : cy + 12, cx - 12 : cx + 12] = 0.0
    mu_f, sig_f = float(np.mean(log_no_dc)), float(np.std(log_no_dc))
    f_spikes = int(np.sum(log_no_dc > (mu_f + 2.5 * sig_f)))

    spatial_prob = _sigmoid((0.0011 - mse) * 1200.0) * 0.55
    if f_ratio > 0.23:
        boost = min(max((f_ratio - 0.2) * 3.5, 0.4), 0.75)
        prob_ai = min(max(spatial_prob + boost + 0.25, 0.8), 0.95)
        reason = f"engine.py v1 kích hoạt FFT (f_ratio={f_ratio:.3f} > 0.23, spikes={f_spikes})."
    elif mse < 0.00045:
        prob_ai = max(0.58, spatial_prob + 0.22)
        reason = f"engine.py v1 dựa vào sai phân không gian thấp (MSE={mse:.6f}); nhánh FFT không kích hoạt (f_ratio={f_ratio:.3f})."
    else:
        prob_ai = max(0.0, min(1.0, 0.12 + spatial_prob * 0.45))
        reason = f"engine.py v1: f_ratio={f_ratio:.3f} (< 0.23) nên nhánh FFT không kích hoạt; suy luận chỉ dựa vào không gian."

    return {
        "probability_ai": float(prob_ai),
        "f_ratio": float(f_ratio),
        "f_spikes": f_spikes,
        "spatial_mse": float(mse),
        "reasoning": reason,
    }


def _build_analysis_log(res: dict, threshold: float = 0.5) -> list:
    """
    Builds structured forensic evidence log entries for app.py and CLI
    without the false-alarm f_spikes > 100 bug from v1.
    """
    log = []
    p_ai = float(res.get("probability_ai", 0.0))
    is_ai = p_ai >= threshold
    metrics = res.get("metrics", {})
    ensemble = res.get("ensemble_info", {})
    native_feat = res.get("native_features", {})
    cagu_w = res.get("cagu_stats", {})
    sub = res.get("sub_scores", {})
    base_info = res.get("baseline_info", {})

    if base_info:
        f_rat = float(base_info.get("f_ratio", 0.0))
        f_spk = int(base_info.get("f_spikes", 0))
        b_prob = float(base_info.get("probability_ai", 0.0))
        log.append(
            {
                "level": "info" if f_rat > 0.23 else "warning",
                "icon": "◆" if f_rat > 0.23 else "▲",
                "title": f"Kiểm tra Baseline (engine.py v1): f_ratio = {f_rat:.3f} · spikes = {f_spk}",
                "detail": (
                    f"Nhánh FFT cũ kích hoạt ngưỡng (>0.23), cho xác suất Baseline {b_prob * 100:.1f}%."
                    if f_rat > 0.23
                    else f"Nhánh FFT cũ (resize 256px) KHÔNG kích hoạt (cần > 0.23), dù log cũ báo nhầm spikes={f_spk} > 100."
                ),
                "value": f"Base {b_prob * 100:.1f}%",
            }
        )

    if native_feat and cagu_w:
        blk = float(native_feat.get("jpeg_blockiness", 1.0))
        sp_pct = float(cagu_w.get("spatial_pct", 33.3))
        fr_pct = float(cagu_w.get("freq_pct", 33.3))
        ch_pct = float(cagu_w.get("chroma_pct", 33.3))
        mode = cagu_w.get("compression_mode", "uncompressed")
        log.append(
            {
                "level": "warning" if mode == "heavy_jpeg" else "ok",
                "icon": "▲" if mode == "heavy_jpeg" else "●",
                "title": f"Van điều tiết CAGU (JPEG Blockiness = {blk:.2f})",
                "detail": (
                    f"Phát hiện nén JPEG nặng (>1.22): CAGU phân bổ {sp_pct:.1f}% Không gian · "
                    f"{fr_pct:.1f}% Tần số Native · {ch_pct:.1f}% Kênh màu Bayer."
                    if mode == "heavy_jpeg"
                    else f"Ảnh ít/không nén JPEG: CAGU phân bổ {sp_pct:.1f}% Không gian · "
                    f"{fr_pct:.1f}% Tần số · {ch_pct:.1f}% Kênh màu."
                ),
                "value": f"{fr_pct:.0f}% Freq",
            }
        )

    if native_feat:
        hi_mid = float(native_feat.get("native_hi_mid_ratio", 1.0))
        peak_med = float(native_feat.get("native_peak_med_ratio", 1.0))
        fft_sz = int(native_feat.get("native_fft_size", 128))
        fr_score = float(sub.get("freq_score", 0.0))
        log.append(
            {
                "level": "critical" if fr_score >= 0.55 else "ok",
                "icon": "✖" if fr_score >= 0.55 else "●",
                "title": f"Phổ 2D-FFT Native ({fft_sz}×{fft_sz}): hi/mid = {hi_mid:.2f} · peak/med = {peak_med:.2f}",
                "detail": (
                    f"Phổ dư khử nội dung suy giảm bất thường ở cao tần (hi/mid={hi_mid:.2f}) "
                    f"hoặc xuất hiện đỉnh chu kỳ (peak/med={peak_med:.2f})."
                    if fr_score >= 0.55
                    else "Phổ năng lượng 2D-FFT trên patch gốc suy giảm tự nhiên theo quy luật 1/f của quang học."
                ),
                "value": f"Score {fr_score * 100:.0f}%",
            }
        )

        chr_lum = float(native_feat.get("chroma_to_luma", 0.2))
        min_corr = float(native_feat.get("min_channel_corr", 0.9))
        sat = float(native_feat.get("saturation_mean", 0.2))
        ch_score = float(sub.get("chroma_score", 0.0))
        log.append(
            {
                "level": "info" if sat < 0.09 else ("critical" if ch_score >= 0.55 else "ok"),
                "icon": "◆" if sat < 0.09 else ("✖" if ch_score >= 0.55 else "●"),
                "title": f"Tương quan Bayer RGB: Chroma/Luma = {chr_lum:.3f} · min_corr = {min_corr:.3f}",
                "detail": (
                    f"Ảnh đơn sắc / độ bão hòa thấp ({sat:.2f}): nhánh sắc độ được trung hòa tự động."
                    if sat < 0.09
                    else (
                        f"Dư sắc độ Chroma/Luma thấp ({chr_lum:.3f}) và các kênh RGB khóa chặt ({min_corr:.3f}) — "
                        "đặc trưng giải mã từ latent space."
                        if ch_score >= 0.55
                        else "Nhiễu sắc độ độc lập giữa các kênh RGB phù hợp cảm biến Bayer thực tế."
                    )
                ),
                "value": f"Score {ch_score * 100:.0f}%",
            }
        )

        sm_noise = float(native_feat.get("smooth_noise", 12.0))
        kurt = float(native_feat.get("lap_kurtosis", 15.0))
        sp_score = float(sub.get("spatial_score", 0.0))
        log.append(
            {
                "level": "critical" if sp_score >= 0.55 else "ok",
                "icon": "✖" if sp_score >= 0.55 else "●",
                "title": f"Nhiễu vùng phẳng & Kurtosis: smoothNoise = {sm_noise:.1f} · Kurtosis = {kurt:.1f}",
                "detail": (
                    f"Phân bố sai phân Laplacian bất thường (smoothNoise={sm_noise:.1f}, kurtosis={kurt:.1f})."
                    if sp_score >= 0.55
                    else "Độ nhọn và nhiễu nền vùng phẳng nằm trong biên độ tự nhiên của ống kính & cảm biến."
                ),
                "value": f"Score {sp_score * 100:.0f}%",
            }
        )

    if ensemble:
        prob_e2e = ensemble.get("prob_end2end")
        prob_sd = ensemble.get("prob_sd")
        if prob_e2e is not None:
            log.append(
                {
                    "level": "critical" if prob_e2e > 0.65 else ("warning" if prob_e2e > 0.4 else "ok"),
                    "icon": "✖" if prob_e2e > 0.65 else ("▲" if prob_e2e > 0.4 else "●"),
                    "title": f"CO-SPY End2End Backbone (SigLIP-384 + VAE): {prob_e2e * 100:.1f}%",
                    "detail": "Nhánh ngữ nghĩa SigLIP-384 kháng nén JPEG kết hợp tái tạo VAE.",
                    "value": f"{prob_e2e * 100:.1f}%",
                }
            )
        elif prob_sd is not None:
            log.append(
                {
                    "level": "warning" if prob_sd > 0.5 else "ok",
                    "icon": "▲" if prob_sd > 0.5 else "●",
                    "title": f"CO-SPY VAE Artifact Branch (sd-v1_4): {prob_sd * 100:.1f}%",
                    "detail": "Điểm tham chiếu từ nhánh VAE thuần (đã hiệu chuẩn với đa miền CAGU).",
                    "value": f"{prob_sd * 100:.1f}%",
                }
            )

    if is_ai:
        log.append(
            {
                "level": "critical",
                "icon": "✖",
                "title": f"KẾT LUẬN ENHANCED v2: ẢNH DO AI TẠO SINH ({p_ai * 100:.1f}%)",
                "detail": res.get("verdict_reasoning", ""),
                "value": f"{p_ai * 100:.1f}% AI",
            }
        )
    else:
        log.append(
            {
                "level": "ok",
                "icon": "●",
                "title": f"KẾT LUẬN ENHANCED v2: ẢNH THẬT QUANG HỌC ({(1.0 - p_ai) * 100:.1f}% thật)",
                "detail": res.get("verdict_reasoning", ""),
                "value": f"{(1.0 - p_ai) * 100:.1f}% thật",
            }
        )
    return log


class CoSpyEngine:
    """Repaired CO-SPY Forensic Inference Engine (PyTorch optional, standalone-ready)."""

    def __init__(self, use_neural_backbone: bool = True):
        self.device = DEVICE
        self.use_neural_backbone = use_neural_backbone and HAS_TORCH and os.path.isdir(CO_SPY_DIR)
        self._models = {}

    def has_sd_artifact_weights(self) -> bool:
        return self.use_neural_backbone and os.path.exists(
            os.path.join(CO_SPY_DIR, "pretrained", "sd-v1_4", "artifact_weights.pth")
        )

    def has_sd_end2end_weights(self) -> bool:
        return self.use_neural_backbone and os.path.exists(
            os.path.join(CO_SPY_DIR, "pretrained", "sd-v1_4", "end2end_weights.pth")
        )

    def get_sd_artifact_detector(self):
        if not self.use_neural_backbone:
            return None
        if "sd_artifact" not in self._models:
            weights_path = os.path.join(CO_SPY_DIR, "pretrained", "sd-v1_4", "artifact_weights.pth")
            if not os.path.exists(weights_path):
                self._models["sd_artifact"] = None
            else:
                mod = importlib.import_module("detectors.sd-v1_4.artifact")
                ArtifactDetector = getattr(mod, "ArtifactDetector")
                detector = ArtifactDetector()
                detector.load_weights(weights_path)
                detector.to(self.device)
                detector.eval()
                self._models["sd_artifact"] = detector
        return self._models["sd_artifact"]

    def get_sd_end2end_detector(self):
        if not self.use_neural_backbone:
            return None
        if "sd_end2end" not in self._models:
            e2e_path = os.path.join(CO_SPY_DIR, "pretrained", "sd-v1_4", "end2end_weights.pth")
            if not os.path.exists(e2e_path):
                self._models["sd_end2end"] = None
            else:
                mod = importlib.import_module("detectors.sd-v1_4.end2end")
                End2EndDetector = getattr(mod, "End2EndDetector")
                detector = End2EndDetector()
                detector.load_weights(e2e_path)
                detector.to(self.device)
                detector.eval()
                self._models["sd_end2end"] = detector
        return self._models["sd_end2end"]

    def get_enhanced_head(self):
        if not HAS_TORCH:
            return None
        if "enhanced_head" not in self._models:
            ckpt_path = os.path.join(BASE_DIR, "pretrained", "enhanced_head.pth")
            if not os.path.exists(ckpt_path):
                self._models["enhanced_head"] = None
            else:
                from adaptive_fusion import EnhancedCoSpyHead
                from frequency_branch import FrequencyBranch

                ckpt = torch.load(ckpt_path, map_location="cpu")
                freq_branch = FrequencyBranch(
                    r0=ckpt.get("r0", 16),
                    out_features=ckpt.get("freq_dim", 128),
                    native_crop_size=ckpt.get("native_crop_size", 128),
                )
                freq_branch.load_state_dict(ckpt["freq_branch"])
                freq_branch.to(self.device).eval()

                head = EnhancedCoSpyHead(
                    spatial_dim=ckpt.get("spatial_dim", 512),
                    freq_dim=ckpt.get("freq_dim", 128),
                    quality_dim=ckpt.get("quality_dim", 4),
                )
                head.load_state_dict(ckpt["head"])
                head.to(self.device).eval()

                self._models["enhanced_head"] = (freq_branch, head)
        return self._models["enhanced_head"]

    def get_progan_fusion_detector(self):
        if not self.use_neural_backbone:
            return None
        if "progan_fusion" not in self._models:
            fus_path = os.path.join(CO_SPY_DIR, "pretrained", "progan", "fusion_weights.pth")
            if not os.path.exists(fus_path):
                self._models["progan_fusion"] = None
            else:
                from detectors.progan.fusion import CoSpyFusionDetector

                sem_path = os.path.join(CO_SPY_DIR, "pretrained", "progan", "semantic_weights.pth")
                art_path = os.path.join(CO_SPY_DIR, "pretrained", "progan", "artifact_weights.pth")
                detector = CoSpyFusionDetector(sem_path, art_path)
                detector.load_weights(fus_path)
                detector.to(self.device)
                detector.eval()
                self._models["progan_fusion"] = detector
        return self._models["progan_fusion"]

    def compute_fft_spectrum(self, pil_image: Image.Image) -> Image.Image:
        """Computes 2D Fourier Transform Power Spectrum on a native-resolution denoised crop."""
        from native_forensics import compute_native_2d_fft

        rgb = np.array(pil_image.convert("RGB"), dtype=np.float64)
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        _, norm_mag = compute_native_2d_fft(gray, max_patch_size=128)
        spectral_color = cv2.applyColorMap(norm_mag, cv2.COLORMAP_INFERNO)
        return Image.fromarray(cv2.cvtColor(spectral_color, cv2.COLOR_BGR2RGB))

    def _build_native_heatmaps(self, md_res: dict, np_rgb: np.ndarray, cv_cmap: int, gain: float):
        res_map = md_res["features"]["residual_map_01"]
        norm_res = np.clip(res_map * gain, 0.0, 1.0)
        hm_bgr = cv2.applyColorMap((norm_res * 255).astype(np.uint8), cv_cmap)
        res_pil = Image.fromarray(cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB))

        grid = md_res["features"]["gradcam_grid_16"]
        h, w, _ = np_rgb.shape
        up_grid = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
        up_norm = np.clip(up_grid * gain, 0.0, 1.0)
        overlay_bgr = cv2.applyColorMap((up_norm * 255).astype(np.uint8), cv_cmap)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        blended = np.clip(0.55 * np_rgb.astype(np.float32) + 0.45 * overlay_rgb.astype(np.float32), 0, 255).astype(
            np.uint8
        )
        gradcam_pil = Image.fromarray(blended)
        return res_pil, gradcam_pil

    def analyze_image(
        self,
        pil_image: Image.Image,
        model_type: str = "CO-SPY Enhanced v2 (Multi-Domain CAGU)",
        colormap_name: str = "JET",
        gain: float = 1.0,
        **kwargs,
    ):
        pil_image = pil_image.convert("RGB")
        start_time = time.time()

        cmap_dict = {
            "JET": cv2.COLORMAP_JET,
            "TURBO": cv2.COLORMAP_TURBO,
            "INFERNO": cv2.COLORMAP_INFERNO,
            "MAGMA": cv2.COLORMAP_MAGMA,
            "BONE": cv2.COLORMAP_BONE,
        }
        cv_cmap = cmap_dict.get(colormap_name, cv2.COLORMAP_JET)
        np_rgb = np.array(pil_image)
        fft_image = self.compute_fft_spectrum(pil_image)
        baseline_info = _compute_legacy_baseline_v1(pil_image)

        m_lower = model_type.lower()

        # --- MODE 1: BASELINE ONLY ---
        if "gốc" in m_lower or "baseline" in m_lower:
            md_res = predict_multi_domain_cagu(np_rgb)
            res_pil, _ = self._build_native_heatmaps(md_res, np_rgb, cv_cmap, gain)
            prob_base = baseline_info["probability_ai"]
            sd_det = self.get_sd_artifact_detector()
            recon_pil = None
            if sd_det is not None and HAS_TORCH:
                with torch.no_grad():
                    t_in = sd_det.test_transform(pil_image).unsqueeze(0).to(self.device)
                    prob_base = float(sd_det.predict(t_in)[0])
                    recon_t = sd_det.artifact_encoder.reconstruct(t_in)
                    recon_np = np.clip(recon_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
                    recon_pil = Image.fromarray((recon_np * 255).astype(np.uint8))

            latency = time.time() - start_time
            return {
                "model_name": "CO-SPY Baseline (engine.py v1 · FFT 256px)",
                "probability_ai": prob_base,
                "probability_real": 1.0 - prob_base,
                "decision": "AI-Generated" if prob_base >= 0.5 else "Real Image",
                "confidence": max(prob_base, 1.0 - prob_base) * 100.0,
                "reconstructed_image": recon_pil if recon_pil is not None else res_pil,
                "residual_image": res_pil,
                "heatmap_image": res_pil,
                "fft_image": fft_image,
                "latency_sec": latency,
                "baseline_info": baseline_info,
                "metrics": {
                    "Xác suất AI Baseline": f"{prob_base * 100:.1f}%",
                    "f_ratio (256px)": f"{baseline_info['f_ratio']:.3f}",
                    "f_spikes": f"{baseline_info['f_spikes']}",
                    "Spatial MSE": f"{baseline_info['spatial_mse']:.6f}",
                    "Độ trễ": f"{latency * 1000:.1f} ms",
                },
                "verdict_reasoning": baseline_info["reasoning"],
            }

        # --- MODE 2: GANs (progan) ---
        if "gan" in m_lower or "progan" in m_lower:
            detector = self.get_progan_fusion_detector()
            if detector is not None and HAS_TORCH:
                tensor = detector.test_transform(pil_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    x_sem = detector.sem_transform(tensor)
                    x_art = detector.art_transform(tensor)
                    score_sem = float(detector.sem(x_sem).sigmoid().item())
                    score_art = float(detector.art(x_art).sigmoid().item())
                    feat = torch.cat([detector.sem(x_sem), detector.art(x_art)], dim=1)
                    prob = float(detector.fc(feat).sigmoid().item())
            else:
                md_res = predict_multi_domain_cagu(np_rgb)
                prob = float(md_res["probability_ai"])
                score_sem = prob
                score_art = float(md_res["sub_scores"]["freq_score"])

            gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
            lap_abs = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
            norm_lap = np.clip((lap_abs / (np.percentile(lap_abs, 99) + 1e-5)) * gain, 0.0, 1.0)
            heatmap_bgr = cv2.applyColorMap((norm_lap * 255).astype(np.uint8), cv_cmap)
            heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB))
            latency = time.time() - start_time
            return {
                "model_name": "CO-SPY GAN Adaptive Fusion (progan)",
                "probability_ai": prob,
                "probability_real": 1.0 - prob,
                "decision": "AI-Generated" if prob >= 0.5 else "Real Image",
                "confidence": max(prob, 1.0 - prob) * 100.0,
                "semantic_score": score_sem,
                "artifact_score": score_art,
                "residual_image": heatmap_pil,
                "heatmap_image": heatmap_pil,
                "reconstructed_image": None,
                "fft_image": fft_image,
                "latency_sec": latency,
                "baseline_info": baseline_info,
                "metrics": {
                    "Semantic Branch": f"{score_sem:.4f}",
                    "Artifact Branch": f"{score_art:.4f}",
                    "Fusion Probability": f"{prob:.4f}",
                    "Độ trễ": f"{latency * 1000:.1f} ms",
                },
                "verdict_reasoning": (
                    "Phát hiện dấu vết up-sampling lưới GAN và bất thường ngữ nghĩa."
                    if prob >= 0.5
                    else "Gradient và ngữ nghĩa tự nhiên, xác nhận ảnh thật."
                ),
            }

        # --- MODE 3 (DEFAULT): CO-SPY ENHANCED v2 (Native Multi-Domain CAGU + Neural Backbone) ---
        md_res = predict_multi_domain_cagu(np_rgb)
        prob_md = float(md_res["probability_ai"])
        md_feat = md_res["features"]
        cagu_stats = md_res["cagu_stats"]
        sub_scores = md_res["sub_scores"]

        prob_sd: Optional[float] = None
        prob_e2e: Optional[float] = None
        prob_trained_head: Optional[float] = None
        mse: Optional[float] = None
        recon_pil = None

        e2e = self.get_sd_end2end_detector()
        sd_det = self.get_sd_artifact_detector()
        enhanced_head_bundle = self.get_enhanced_head()

        if HAS_TORCH and (sd_det is not None or e2e is not None):
            with torch.no_grad():
                if sd_det is not None:
                    sd_t = sd_det.test_transform(pil_image).unsqueeze(0).to(self.device)
                    sp_feat, art_logit = sd_det(sd_t, return_feat=True)
                    prob_sd = float(torch.sigmoid(art_logit).item())

                    recon_t = sd_det.artifact_encoder.reconstruct(sd_t)
                    diff_t = torch.abs(sd_t - recon_t).squeeze(0).permute(1, 2, 0).cpu().numpy()
                    mse = float(np.mean(diff_t ** 2))
                    recon_np = np.clip(recon_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
                    recon_pil = Image.fromarray((recon_np * 255).astype(np.uint8))

                    if enhanced_head_bundle is not None:
                        freq_branch, head = enhanced_head_bundle
                        nat_t = torch.from_numpy(np_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
                        fr_feat = freq_branch(nat_t)
                        q_feat = torch.tensor([md_res["quality_vector"]], dtype=torch.float32, device=self.device)
                        mse_t = torch.tensor([[mse]], dtype=torch.float32, device=self.device)
                        prob_trained_head = float(head(sp_feat, fr_feat, mse_t, quality_feat=q_feat).item())

                if e2e is not None:
                    e2e_t = e2e.test_transform(pil_image).unsqueeze(0).to(self.device)
                    prob_e2e = float(e2e.predict(e2e_t)[0])

        # Hybrid fusion across calibrated multi-domain physics + CO-SPY neural backbones
        if prob_trained_head is not None and prob_e2e is not None:
            prob = 0.45 * prob_trained_head + 0.30 * prob_e2e + 0.25 * prob_md
        elif prob_e2e is not None:
            prob = 0.55 * prob_md + 0.45 * prob_e2e
        elif prob_trained_head is not None:
            prob = 0.50 * prob_trained_head + 0.50 * prob_md
        elif prob_sd is not None:
            vae_w = 0.10 if cagu_stats["compression_mode"] == "heavy_jpeg" else 0.18
            prob = (1.0 - vae_w) * prob_md + vae_w * prob_sd
        else:
            prob = prob_md

        prob = float(max(0.0, min(1.0, prob)))
        res_pil, gradcam_pil = self._build_native_heatmaps(md_res, np_rgb, cv_cmap, gain)
        latency = time.time() - start_time

        return {
            "model_name": "CO-SPY Enhanced v2 (Multi-Domain CAGU + Native 2D-FFT)",
            "probability_ai": prob,
            "probability_real": 1.0 - prob,
            "decision": "AI-Generated" if prob >= 0.5 else "Real Image",
            "confidence": max(prob, 1.0 - prob) * 100.0,
            "reconstructed_image": recon_pil if recon_pil is not None else res_pil,
            "residual_image": res_pil,
            "heatmap_image": gradcam_pil,
            "fft_image": fft_image,
            "latency_sec": latency,
            "baseline_info": baseline_info,
            "freq_data": {
                "ratio": float(md_feat["native_hi_mid_ratio"]),
                "peak_med": float(md_feat["native_peak_med_ratio"]),
                "spikes": 0,
            },
            "cagu_stats": cagu_stats,
            "sub_scores": sub_scores,
            "native_features": {k: v for k, v in md_feat.items() if isinstance(v, (int, float))},
            "ensemble_info": {
                "prob_sd": prob_sd,
                "prob_end2end": prob_e2e,
                "prob_multidomain": prob_md,
                "primary_lead": md_res["primary_lead"],
            },
            "metrics": {
                "Chroma / Luma": f"{md_feat['chroma_to_luma']:.3f}",
                "Tương quan min RGB": f"{md_feat['min_channel_corr']:.3f}",
                "FFT hi/mid (Native)": f"{md_feat['native_hi_mid_ratio']:.2f}",
                "Nhiễu vùng phẳng": f"{md_feat['smooth_noise']:.1f}",
                "Độ nhọn Kurtosis": f"{md_feat['lap_kurtosis']:.1f}",
                "JPEG Blockiness": f"{md_feat['jpeg_blockiness']:.2f}",
            },
            "verdict_reasoning": md_res["reasoning"],
        }
