"""
engine.py — Repaired CO-SPY Forensic Inference Engine
=====================================================
Upgrades over v1:
  1. Loads full End2EndDetector (Semantic SigLIP-384 + VAE Artifact) when pretrained weights exist,
     falling back cleanly to ArtifactDetector when only artifact_weights.pth is present.
  2. Replaces the broken 256x256 bilinear FFT heuristic (f_ratio > 0.23, f_spikes > 100, mse < 0.0008)
     with calibrated Native-Resolution Multi-Domain CAGU (native_forensics.py) + optional trained
     EnhancedCoSpyHead (pretrained/enhanced_head.pth).
  3. Preserves 100% API compatibility with app.py:
       CoSpyEngine.analyze_image(pil_image, model_type, colormap_name, gain)
       _build_analysis_log(res, threshold)
"""

import importlib
import os
import sys
import time
import cv2
import numpy as np
import torch
from PIL import Image

from native_forensics import predict_multi_domain_cagu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CO_SPY_DIR = os.path.join(BASE_DIR, "Co-Spy")
if os.path.isdir(CO_SPY_DIR) and CO_SPY_DIR not in sys.path:
    sys.path.insert(0, CO_SPY_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _build_analysis_log(res: dict, threshold: float = 0.5) -> list:
    """
    Builds structured forensic evidence log entries for app.py without the
    false-alarm f_spikes > 100 bug from v1.
    """
    log = []
    p_ai = float(res.get("probability_ai", 0.0))
    is_ai = p_ai >= threshold
    metrics = res.get("metrics", {})
    ensemble = res.get("ensemble_info", {})
    native_feat = res.get("native_features", {})
    cagu_w = res.get("cagu_stats", {})
    sub = res.get("sub_scores", {})

    # 1. CAGU Quality & JPEG Blockiness Gate
    if native_feat and cagu_w:
        blk = float(native_feat.get("jpeg_blockiness", 1.0))
        sp_pct = float(cagu_w.get("spatial_pct", 33.3))
        fr_pct = float(cagu_w.get("freq_pct", 33.3))
        ch_pct = float(cagu_w.get("chroma_pct", 33.3))
        mode = cagu_w.get("compression_mode", "uncompressed")
        log.append(
            {
                "level": "warning" if mode == "heavy_jpeg" else "ok",
                "icon": "🎛️",
                "title": f"Van điều tiết CAGU (JPEG Blockiness = {blk:.2f})",
                "detail": (
                    f"Phát hiện nén JPEG nặng (>1.22): CAGU phân bổ {sp_pct:.1f}% Không gian · "
                    f"{fr_pct:.1f}% Phổ tần số Native · {ch_pct:.1f}% Kênh màu Bayer."
                    if mode == "heavy_jpeg"
                    else f"Ảnh ít/không nén JPEG: CAGU phân bổ {sp_pct:.1f}% Không gian · "
                    f"{fr_pct:.1f}% Tần số · {ch_pct:.1f}% Kênh màu."
                ),
                "value": f"{fr_pct:.0f}% Freq",
            }
        )

    # 2. Native 2D-FFT Spectral Branch
    if native_feat:
        hi_mid = float(native_feat.get("native_hi_mid_ratio", 1.0))
        peak_med = float(native_feat.get("native_peak_med_ratio", 1.0))
        fft_sz = int(native_feat.get("native_fft_size", 128))
        fr_score = float(sub.get("freq_score", 0.0))
        log.append(
            {
                "level": "critical" if fr_score >= 0.55 else "ok",
                "icon": "🔴" if fr_score >= 0.55 else "🟢",
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

        # 3. Bayer Chrominance Coherence
        chr_lum = float(native_feat.get("chroma_to_luma", 0.2))
        min_corr = float(native_feat.get("min_channel_corr", 0.9))
        sat = float(native_feat.get("saturation_mean", 0.2))
        ch_score = float(sub.get("chroma_score", 0.0))
        log.append(
            {
                "level": "info" if sat < 0.09 else ("critical" if ch_score >= 0.55 else "ok"),
                "icon": "🔵" if sat < 0.09 else ("🔴" if ch_score >= 0.55 else "🟢"),
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

        # 4. Smooth-Region Sensor Noise & Kurtosis
        sm_noise = float(native_feat.get("smooth_noise", 12.0))
        kurt = float(native_feat.get("lap_kurtosis", 15.0))
        sp_score = float(sub.get("spatial_score", 0.0))
        log.append(
            {
                "level": "critical" if sp_score >= 0.55 else "ok",
                "icon": "🔴" if sp_score >= 0.55 else "🟢",
                "title": f"Nhiễu vùng phẳng & Kurtosis: smoothNoise = {sm_noise:.1f} · Kurtosis = {kurt:.1f}",
                "detail": (
                    f"Phân bố sai phân Laplacian bất thường (smoothNoise={sm_noise:.1f}, kurtosis={kurt:.1f})."
                    if sp_score >= 0.55
                    else "Độ nhọn và nhiễu nền vùng phẳng nằm trong biên độ tự nhiên của ống kính & cảm biến."
                ),
                "value": f"Score {sp_score * 100:.0f}%",
            }
        )

    # 5. CO-SPY Neural Backbone info (End2End Semantic+Artifact or VAE)
    if ensemble:
        prob_e2e = ensemble.get("prob_end2end")
        prob_sd = ensemble.get("prob_sd")
        if prob_e2e is not None:
            log.append(
                {
                    "level": "critical" if prob_e2e > 0.65 else ("warning" if prob_e2e > 0.4 else "ok"),
                    "icon": "🔴" if prob_e2e > 0.65 else ("🟡" if prob_e2e > 0.4 else "🟢"),
                    "title": f"CO-SPY End2End Backbone (SigLIP-384 + VAE): {prob_e2e * 100:.1f}%",
                    "detail": "Nhánh ngữ nghĩa SigLIP-384 kháng nén JPEG kết hợp tái tạo VAE.",
                    "value": f"{prob_e2e * 100:.1f}%",
                }
            )
        elif prob_sd is not None:
            log.append(
                {
                    "level": "warning" if prob_sd > 0.5 else "ok",
                    "icon": "🟡" if prob_sd > 0.5 else "🟢",
                    "title": f"CO-SPY VAE Artifact Branch (sd-v1_4): {prob_sd * 100:.1f}%",
                    "detail": "Điểm tham chiếu từ nhánh VAE thuần (đã hiệu chuẩn với đa miền CAGU).",
                    "value": f"{prob_sd * 100:.1f}%",
                }
            )

    # 6. MSE summary for Baseline compatibility
    mse_val = metrics.get("Reconstruction MSE")
    if mse_val is not None:
        log.append(
            {
                "level": "info",
                "icon": "🔵",
                "title": f"VAE Reconstruction MSE: {mse_val}",
                "detail": "Sai số tái tạo CompVis VAE dùng làm tham chiếu không gian (không dùng ngưỡng cứng MSE < 0.0008).",
                "value": str(mse_val),
            }
        )

    # 7. Final Conclusion
    critical_count = sum(1 for x in log if x["level"] == "critical")
    if is_ai:
        log.append(
            {
                "level": "critical",
                "icon": "⚠️",
                "title": f"KẾT LUẬN: {critical_count} bằng chứng mạnh → ẢNH DO AI TẠO SINH",
                "detail": f"Xác suất AI tổng hợp = {p_ai * 100:.2f}% >= ngưỡng {threshold * 100:.0f}%.",
                "value": f"{p_ai * 100:.2f}%",
            }
        )
    else:
        log.append(
            {
                "level": "ok",
                "icon": "✅",
                "title": "KẾT LUẬN: ĐỒNG THUẬN ĐA MIỀN → ẢNH THẬT QUANG HỌC",
                "detail": f"Xác suất AI = {p_ai * 100:.2f}% < ngưỡng {threshold * 100:.0f}% — xác nhận ảnh thật.",
                "value": f"{(1.0 - p_ai) * 100:.2f}% thật",
            }
        )
    return log


class CoSpyEngine:
    """Repaired CO-SPY Forensic Inference Engine."""

    def __init__(self):
        self.device = DEVICE
        self._models = {}

    def get_sd_artifact_detector(self):
        if "sd_artifact" not in self._models:
            mod = importlib.import_module("detectors.sd-v1_4.artifact")
            ArtifactDetector = getattr(mod, "ArtifactDetector")
            detector = ArtifactDetector()
            weights_path = os.path.join(CO_SPY_DIR, "pretrained", "sd-v1_4", "artifact_weights.pth")
            if os.path.exists(weights_path):
                detector.load_weights(weights_path)
            detector.to(self.device)
            detector.eval()
            self._models["sd_artifact"] = detector
        return self._models["sd_artifact"]

    def get_sd_end2end_detector(self):
        """Loads full End2EndDetector (Semantic SigLIP-384 + Artifact VAE) if weights exist."""
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
        """Loads trained EnhancedCoSpyHead + FrequencyBranch if pretrained/enhanced_head.pth exists."""
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
        if "progan_fusion" not in self._models:
            from detectors.progan.fusion import CoSpyFusionDetector

            sem_path = os.path.join(CO_SPY_DIR, "pretrained", "progan", "semantic_weights.pth")
            art_path = os.path.join(CO_SPY_DIR, "pretrained", "progan", "artifact_weights.pth")
            fus_path = os.path.join(CO_SPY_DIR, "pretrained", "progan", "fusion_weights.pth")
            detector = CoSpyFusionDetector(sem_path, art_path)
            detector.load_weights(fus_path)
            detector.to(self.device)
            detector.eval()
            self._models["progan_fusion"] = detector
        return self._models["progan_fusion"]

    def compute_fft_spectrum(self, pil_image: Image.Image) -> Image.Image:
        """Computes 2D Fourier Transform Power Spectrum on a native-resolution denoised crop."""
        from frequency_branch import compute_fft_power_spectrum

        norm_mag = compute_fft_power_spectrum(pil_image.convert("RGB"), size=128)
        spectral_color = cv2.applyColorMap(norm_mag, cv2.COLORMAP_INFERNO)
        spectral_rgb = cv2.cvtColor(spectral_color, cv2.COLOR_BGR2RGB)
        return Image.fromarray(spectral_rgb)

    def _build_native_heatmaps(self, md_res: dict, np_rgb: np.ndarray, cv_cmap: int, gain: float):
        res_map = md_res["features"]["residual_map_01"]
        norm_res = np.clip(res_map * gain, 0.0, 1.0)
        hm_bgr = cv2.applyColorMap((norm_res * 255).astype(np.uint8), cv_cmap)
        hm_pil = Image.fromarray(cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB))

        # Overlay 16x16 anomaly map cleanly onto resized preview
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
        return hm_pil, gradcam_pil

    def analyze_image(
        self,
        pil_image: Image.Image,
        model_type: str = "Tự động (Auto-Ensemble) — Khuyến nghị",
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

        m_lower = model_type.lower()

        # --- MODE 1: REPAIRED CO-SPY ENHANCED v2 (Native Multi-Domain CAGU + Neural Backbone) ---
        if "cải tiến" in m_lower or "enhanced" in m_lower or "tự động" in m_lower or "auto" in m_lower or "⭐" in model_type:
            md_res = predict_multi_domain_cagu(np_rgb)
            prob_md = float(md_res["probability_ai"])
            md_feat = md_res["features"]
            cagu_stats = md_res["cagu_stats"]
            sub_scores = md_res["sub_scores"]

            # Optional Neural CO-SPY Backbone (End2End Semantic+Artifact or Artifact-only)
            prob_sd = None
            prob_e2e = None
            mse = None
            recon_pil = None
            neural_heatmap = None

            e2e = self.get_sd_end2end_detector()
            sd_det = self.get_sd_artifact_detector()
            enhanced_head_bundle = self.get_enhanced_head()

            with torch.no_grad():
                sd_tensor = sd_det.test_transform(pil_image).unsqueeze(0).to(self.device)
                sp_feat, art_logit = sd_det(sd_tensor, return_feat=True)
                prob_sd = float(torch.sigmoid(art_logit).item())

                recon_tensor = sd_det.artifact_encoder.reconstruct(sd_tensor)
                diff_tensor = torch.abs(sd_tensor - recon_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
                mse = float(np.mean(diff_tensor ** 2))
                recon_np = np.clip(recon_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
                recon_pil = Image.fromarray((recon_np * 255).astype(np.uint8))

                if e2e is not None:
                    e2e_tensor = e2e.test_transform(pil_image).unsqueeze(0).to(self.device)
                    prob_e2e = float(e2e.predict(e2e_tensor)[0])

                prob_trained_head = None
                if enhanced_head_bundle is not None:
                    freq_branch, head = enhanced_head_bundle
                    native_t = torch.from_numpy(np_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
                    fr_feat = freq_branch(native_t)
                    q_feat = torch.tensor([md_res["quality_vector"]], dtype=torch.float32, device=self.device)
                    mse_t = torch.tensor([[mse]], dtype=torch.float32, device=self.device)
                    prob_trained_head = float(head(sp_feat, fr_feat, mse_t, quality_feat=q_feat).item())

            # Calibrated Hybrid Fusion:
            # - If trained EnhancedCoSpyHead exists + End2End exists, blend all three strongly
            # - Otherwise blend calibrated Native Multi-Domain CAGU (0.75) + End2End/Artifact signal
            if prob_trained_head is not None and prob_e2e is not None:
                prob = 0.45 * prob_trained_head + 0.30 * prob_e2e + 0.25 * prob_md
            elif prob_e2e is not None:
                prob = 0.55 * prob_md + 0.45 * prob_e2e
            elif prob_trained_head is not None:
                prob = 0.50 * prob_trained_head + 0.50 * prob_md
            else:
                # Calibrated fallback: rely primarily on verified Multi-Domain CAGU
                # and gently incorporate VAE artifact under uncompressed conditions
                vae_weight = 0.10 if cagu_stats["compression_mode"] == "heavy_jpeg" else 0.18
                prob = (1.0 - vae_weight) * prob_md + vae_weight * prob_sd

            prob = float(max(0.0, min(1.0, prob)))
            res_hm_pil, gradcam_pil = self._build_native_heatmaps(md_res, np_rgb, cv_cmap, gain)
            heatmap_pil = gradcam_pil

            latency = time.time() - start_time
            return {
                "model_name": "CO-SPY Enhanced v2 (Multi-Domain CAGU + Native 2D-FFT)",
                "probability_ai": prob,
                "probability_real": 1.0 - prob,
                "decision": "AI-Generated" if prob > 0.5 else "Real Image",
                "confidence": max(prob, 1.0 - prob) * 100.0,
                "reconstructed_image": recon_pil if recon_pil is not None else res_hm_pil,
                "heatmap_image": heatmap_pil,
                "fft_image": fft_image,
                "latency_sec": latency,
                "freq_data": {
                    "ratio": float(md_feat["native_hi_mid_ratio"]),
                    "peak_med": float(md_feat["native_peak_med_ratio"]),
                    "spikes": 0,
                },
                "cagu_stats": cagu_stats,
                "sub_scores": sub_scores,
                "native_features": {
                    k: v
                    for k, v in md_feat.items()
                    if isinstance(v, (int, float))
                },
                "ensemble_info": {
                    "prob_sd": prob_sd,
                    "prob_end2end": prob_e2e,
                    "prob_multidomain": prob_md,
                    "primary_lead": md_res["primary_lead"],
                },
                "metrics": {
                    "Xác suất AI (Enhanced v2)": f"{prob * 100:.2f}%",
                    "Multi-Domain CAGU": f"{prob_md * 100:.2f}%",
                    "Chroma / Luma": f"{md_feat['chroma_to_luma']:.3f}",
                    "FFT hi/mid (Native)": f"{md_feat['native_hi_mid_ratio']:.2f}",
                    "JPEG Blockiness": f"{md_feat['jpeg_blockiness']:.2f}",
                    "Reconstruction MSE": f"{mse:.6f}" if mse is not None else "N/A",
                    "Độ trễ": f"{latency:.2f}s",
                },
                "verdict_reasoning": md_res["reasoning"],
            }

        # --- MODE 2: BASELINE (Exact original VAE Artifact-only reference) ---
        elif "gốc" in m_lower or "baseline" in m_lower or "diffusion" in m_lower:
            sd_detector = self.get_sd_artifact_detector()
            sd_tensor = sd_detector.test_transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prob_sd = float(sd_detector.predict(sd_tensor)[0])
                recon_tensor = sd_detector.artifact_encoder.reconstruct(sd_tensor)
                diff_tensor = torch.abs(sd_tensor - recon_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
                mse = float(np.mean(diff_tensor ** 2))
                recon_np = np.clip(recon_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
                recon_pil = Image.fromarray((recon_np * 255).astype(np.uint8))

                diff_gray = np.mean(diff_tensor, axis=2)
                p99 = max(float(np.percentile(diff_gray, 99.5)) if np.max(diff_gray) > 0 else 1.0, 1e-5)
                norm_diff = np.clip((diff_gray / p99) * gain, 0.0, 1.0)
                heatmap_bgr = cv2.applyColorMap((norm_diff * 255).astype(np.uint8), cv_cmap)
                heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB))

            latency = time.time() - start_time
            return {
                "model_name": "CO-SPY Baseline (sd-v1_4 VAE Artifact Only)",
                "probability_ai": prob_sd,
                "probability_real": 1.0 - prob_sd,
                "decision": "AI-Generated" if prob_sd > 0.5 else "Real Image",
                "confidence": max(prob_sd, 1.0 - prob_sd) * 100.0,
                "reconstructed_image": recon_pil,
                "heatmap_image": heatmap_pil,
                "fft_image": fft_image,
                "latency_sec": latency,
                "metrics": {
                    "Xác suất AI Baseline": f"{prob_sd * 100:.2f}%",
                    "Reconstruction MSE": f"{mse:.6f}",
                    "Cơ chế": "Spatial VAE thuần",
                    "Độ trễ": f"{latency:.2f}s",
                },
                "verdict_reasoning": (
                    "Mô hình Baseline chỉ đo sai phân VAE (không có CLIP Semantic & Native Multi-Domain CAGU)."
                ),
            }

        # --- MODE 3: GANs (progan) ---
        else:
            detector = self.get_progan_fusion_detector()
            tensor = detector.test_transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                x_sem = detector.sem_transform(tensor)
                x_art = detector.art_transform(tensor)
                score_sem = float(detector.sem(x_sem).sigmoid().item())
                score_art = float(detector.art(x_art).sigmoid().item())
                feat = torch.cat([detector.sem(x_sem), detector.art(x_art)], dim=1)
                prob = float(detector.fc(feat).sigmoid().item())

                gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
                lap_abs = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
                norm_lap = np.clip((lap_abs / (np.percentile(lap_abs, 99) + 1e-5)) * gain, 0.0, 1.0)
                heatmap_bgr = cv2.applyColorMap((norm_lap * 255).astype(np.uint8), cv_cmap)
                heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB))

            latency = time.time() - start_time
            return {
                "model_name": "CO-SPY GAN Adaptive Fusion Engine (progan)",
                "probability_ai": prob,
                "probability_real": 1.0 - prob,
                "decision": "AI-Generated" if prob > 0.5 else "Real Image",
                "confidence": max(prob, 1.0 - prob) * 100.0,
                "semantic_score": score_sem,
                "artifact_score": score_art,
                "heatmap_image": heatmap_pil,
                "reconstructed_image": None,
                "fft_image": fft_image,
                "latency_sec": latency,
                "metrics": {
                    "Semantic Branch": f"{score_sem:.4f}",
                    "Artifact Branch": f"{score_art:.4f}",
                    "Fusion Probability": f"{prob:.4f}",
                    "Độ trễ": f"{latency:.2f}s",
                },
                "verdict_reasoning": (
                    "Phát hiện dấu vết up-sampling GAN và bất thường ngữ nghĩa CLIP."
                    if prob > 0.5
                    else "Gradient và ngữ nghĩa tự nhiên, xác nhận ảnh thật."
                ),
            }
