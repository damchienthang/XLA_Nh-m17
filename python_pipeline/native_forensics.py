"""
native_forensics.py — Calibrated Native-Resolution Multi-Domain Signal Branch
=============================================================================
Extracts physical camera vs. generative-AI artifacts directly from native pixels
without destructive 256x256 bilinear downsampling:
  1. Multi-crop native 2D-FFT spectral roll-off (excluding DC axes & 8x8 JPEG grid harmonics)
  2. Bayer RGB chromatic coherence (Chroma/Luma residual ratio, corr(R,G), corr(G,B))
  3. Non-quantized smooth-region sensor shot noise (nz_smooth_noise) & Laplacian kurtosis
  4. 8x8 JPEG blockiness estimator & Quality-Aware Adaptive Gating (CAGU)
  5. Graphic/poster overlay isolation + heavy-JPEG 4:2:0 chroma de-weighting
"""

import math
from typing import Dict, List, Tuple
import numpy as np


def _sigmoid(z: float) -> float:
    if z > 20.0:
        return 1.0
    if z < -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _compute_single_crop_fft(
    gray: np.ndarray, ox: int, oy: int, size: int
) -> Tuple[float, float, float, np.ndarray]:
    """Computes clean 2D-FFT ratios on a single native crop, excluding DC cross & JPEG 8x8 harmonics."""
    h, w = gray.shape
    if h < size + 2 or w < size + 2:
        padded = np.pad(gray, ((1, max(1, size + 2 - h)), (1, max(1, size + 2 - w))), mode="edge")
        patch_res = (
            padded[1 : 1 + size, 1 : 1 + size]
            - 0.25
            * (
                padded[1 : 1 + size, 0:size]
                + padded[1 : 1 + size, 2 : 2 + size]
                + padded[0:size, 1 : 1 + size]
                + padded[2 : 2 + size, 1 : 1 + size]
            )
        )
    else:
        ox = max(1, min(w - size - 1, ox))
        oy = max(1, min(h - size - 1, oy))
        sub = gray[oy : oy + size, ox : ox + size]
        avg4 = 0.25 * (
            gray[oy : oy + size, ox - 1 : ox + size - 1]
            + gray[oy : oy + size, ox + 1 : ox + size + 1]
            + gray[oy - 1 : oy + size - 1, ox : ox + size]
            + gray[oy + 1 : oy + size + 1, ox : ox + size]
        )
        patch_res = sub - avg4

    fshift = np.fft.fftshift(np.fft.fft2(patch_res))
    mag = np.abs(fshift)

    half = size // 2
    step8 = max(1, size // 8)
    yy, xx = np.ogrid[:size, :size]
    dx = np.abs(xx - half)
    dy = np.abs(yy - half)
    r_norm = np.hypot(dx, dy) / float(half)
    non_axis = (dx > 2) & (dy > 2)

    low_mask = non_axis & (r_norm >= 0.06) & (r_norm < 0.25)
    mid_mask = non_axis & (r_norm >= 0.25) & (r_norm < 0.625)
    high_mask = non_axis & (r_norm >= 0.625) & (r_norm <= 1.0)

    e_low = float(np.mean(mag[low_mask])) if np.any(low_mask) else 1e-6
    e_mid = float(np.mean(mag[mid_mask])) if np.any(mid_mask) else 1e-6
    e_high = float(np.mean(mag[high_mask])) if np.any(high_mask) else 1e-6

    jpeg_harm_x = ((dx % step8) <= 1) | ((step8 - (dx % step8)) <= 1)
    jpeg_harm_y = ((dy % step8) <= 1) | ((step8 - (dy % step8)) <= 1)
    clean_high_mask = high_mask & (~(jpeg_harm_x & jpeg_harm_y))
    high_vals = mag[clean_high_mask]

    if high_vals.size > 0:
        high_med = float(np.median(high_vals))
        high_p99 = float(np.percentile(high_vals, 99))
        peak_med_ratio = high_p99 / (high_med + 1e-6)
    else:
        peak_med_ratio = 1.0

    mid_low_ratio = e_mid / (e_low + 1e-6)
    hi_mid_ratio = e_high / (e_mid + 1e-6)

    log_mag = np.log(mag + 1e-6)
    l_min, l_max = float(log_mag.min()), float(log_mag.max())
    log_uint8 = np.clip((log_mag - l_min) / (l_max - l_min + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return hi_mid_ratio, mid_low_ratio, peak_med_ratio, log_uint8


def compute_native_2d_fft(
    gray: np.ndarray, max_patch_size: int = 128
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Computes 2D-FFT across 5 native-resolution crops (center + 4 quadrant offsets)
    excluding DC cross & JPEG 8x8 grid harmonics to avoid false architectural/JPEG spikes.

    Returns:
        stats: dict with native_fft_size, hi_mid_ratio, mid_low_ratio, peak_med_ratio, peak_med_p75
        fft_log_mag_uint8: [N, N] uint8 log-magnitude spectrum of center crop for visualization
    """
    h, w = gray.shape
    min_dim = max(64, min(h, w))
    max_pow2 = 1 << int(math.floor(math.log2(min_dim)))
    size = max(64, min(max_patch_size, max_pow2))

    crop_coords = [
        ((w - size) // 2, (h - size) // 2),
        (int((w - size) * 0.18), int((h - size) * 0.25)),
        (int((w - size) * 0.82), int((h - size) * 0.25)),
        (int((w - size) * 0.18), int((h - size) * 0.75)),
        (int((w - size) * 0.82), int((h - size) * 0.75)),
    ]

    hm_list: List[float] = []
    ml_list: List[float] = []
    pk_list: List[float] = []
    center_log_uint8 = np.zeros((size, size), dtype=np.uint8)

    for idx, (ox, oy) in enumerate(crop_coords):
        hm, ml, pk, log_u8 = _compute_single_crop_fft(gray, ox, oy, size)
        hm_list.append(hm)
        ml_list.append(ml)
        pk_list.append(pk)
        if idx == 0:
            center_log_uint8 = log_u8

    hm_sorted = sorted(hm_list)
    ml_sorted = sorted(ml_list)
    pk_sorted = sorted(pk_list)

    return (
        {
            "native_fft_size": int(size),
            "hi_mid_ratio": float(hm_sorted[2]),
            "mid_low_ratio": float(ml_sorted[2]),
            "peak_med_ratio": float(pk_sorted[2]),
            "peak_med_p75": float(pk_sorted[3]),
        },
        center_log_uint8,
    )


def extract_multi_domain_features(rgb_uint8: np.ndarray) -> Dict[str, object]:
    """
    Extracts native-resolution spatial, spectral, chrominance, and JPEG quality features
    from an HxWx3 uint8 RGB numpy array.
    """
    rgb = rgb_uint8.astype(np.float64)
    h, w, _ = rgb.shape
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Saturation mean (used to detect monochrome or vivid graphic overlays)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    saturation_mean = float(np.mean((max_c - min_c) / 255.0))

    # JPEG 8x8 block boundary gradient discontinuity ratio
    dx = np.abs(gray[1:-1, 1:-1] - gray[1:-1, :-2])
    dy = np.abs(gray[1:-1, 1:-1] - gray[:-2, 1:-1])
    x_idx = np.arange(1, w - 1)[None, :]
    y_idx = np.arange(1, h - 1)[:, None]
    bx_mask = (x_idx % 8) == 0
    by_mask = (y_idx % 8) == 0

    bnd_sum = float(np.sum(dx[:, (np.arange(1, w - 1) % 8) == 0])) + float(
        np.sum(dy[(np.arange(1, h - 1) % 8) == 0, :])
    )
    int_sum = float(np.sum(dx[:, (np.arange(1, w - 1) % 8) != 0])) + float(
        np.sum(dy[(np.arange(1, h - 1) % 8) != 0, :])
    )
    bnd_cnt = int(bx_mask.sum() * (h - 2) + by_mask.sum() * (w - 2))
    int_cnt = int((w - 2) * (h - 2) * 2 - bnd_cnt)
    jpeg_blockiness = (bnd_sum / max(bnd_cnt, 1)) / (int_sum / max(int_cnt, 1) + 1e-6)

    # 4-neighbor Laplacian residual per RGB channel
    lr = 4.0 * r[1:-1, 1:-1] - r[1:-1, :-2] - r[1:-1, 2:] - r[:-2, 1:-1] - r[2:, 1:-1]
    lg = 4.0 * g[1:-1, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:] - g[:-2, 1:-1] - g[2:, 1:-1]
    lb = 4.0 * b[1:-1, 1:-1] - b[1:-1, :-2] - b[1:-1, 2:] - b[:-2, 1:-1] - b[2:, 1:-1]

    luma = 0.299 * lr + 0.587 * lg + 0.114 * lb
    cb = -0.169 * lr - 0.331 * lg + 0.5 * lb
    cr = 0.5 * lr - 0.419 * lg - 0.081 * lb

    luma_sq = float(np.mean(luma * luma))
    cb_sq = float(np.mean(cb * cb))
    cr_sq = float(np.mean(cr * cr))
    chroma_to_luma = math.sqrt((cb_sq + cr_sq) / (2.0 * luma_sq + 1e-9))

    var_r = float(np.sum(lr * lr))
    var_g = float(np.sum(lg * lg))
    var_b = float(np.sum(lb * lb))
    corr_rg = float(np.sum(lr * lg)) / math.sqrt(var_r * var_g + 1e-9)
    corr_gb = float(np.sum(lg * lb)) / math.sqrt(var_g * var_b + 1e-9)

    lap_fourth = float(np.mean(luma ** 4))
    lap_kurtosis = lap_fourth / (luma_sq * luma_sq + 1e-9)

    # Smooth-region noise + non-quantized smooth noise (excludes grad < 2 JPEG-zeroed blocks)
    gx = np.abs(g[1:-1, 2:] - g[1:-1, :-2])
    gy = np.abs(g[2:, 1:-1] - g[:-2, 1:-1])
    g_mag = gx + gy
    smooth_mask = g_mag < 12.0
    nz_smooth_mask = (g_mag >= 2.0) & (g_mag < 12.0)

    zero_grad_pct = float(np.mean(g_mag < 2.0) * 100.0)
    sharp_edge_pct = float(np.mean(g_mag > 120.0) * 100.0)

    smooth_noise = math.sqrt(float(np.mean(luma[smooth_mask] ** 2))) if np.any(smooth_mask) else 0.0
    nz_smooth_noise = (
        math.sqrt(float(np.mean(luma[nz_smooth_mask] ** 2))) if np.any(nz_smooth_mask) else smooth_noise
    )

    # Multi-crop Native 2D-FFT
    fft_stats, fft_log_uint8 = compute_native_2d_fft(gray, max_patch_size=128)

    # Spatial SRM residual map (|I - avg4| / 255 normalized to p99.5)
    c_gray = gray[1:-1, 1:-1]
    avg4 = 0.25 * (gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1])
    res_raw = np.abs(c_gray - avg4) / 255.0
    p995 = max(float(np.percentile(res_raw, 99.5)) if res_raw.size > 0 else 0.05, 1e-4)
    residual_map_01 = np.clip(res_raw / p995, 0.0, 1.0).astype(np.float32)

    # 16x16 local anomaly heatmap
    grid_n = 16
    grid_score = np.zeros((grid_n, grid_n), dtype=np.float32)
    for gy_i in range(grid_n):
        y0 = max(1, int((gy_i / grid_n) * h))
        y1 = min(h - 1, int(((gy_i + 1) / grid_n) * h))
        for gx_i in range(grid_n):
            x0 = max(1, int((gx_i / grid_n) * w))
            x1 = min(w - 1, int(((gx_i + 1) / grid_n) * w))
            if y1 <= y0 or x1 <= x0:
                continue
            dr = 2.0 * r[y0:y1:2, x0:x1:2] - r[y0:y1:2, x0 - 1 : x1 - 1 : 2] - r[y0 - 1 : y1 - 1 : 2, x0:x1:2]
            dg = 2.0 * g[y0:y1:2, x0:x1:2] - g[y0:y1:2, x0 - 1 : x1 - 1 : 2] - g[y0 - 1 : y1 - 1 : 2, x0:x1:2]
            db = 2.0 * b[y0:y1:2, x0:x1:2] - b[y0:y1:2, x0 - 1 : x1 - 1 : 2] - b[y0 - 1 : y1 - 1 : 2, x0:x1:2]
            pl = 0.299 * dr + 0.587 * dg + 0.114 * db
            pcb = -0.169 * dr - 0.331 * dg + 0.5 * db
            pcr = 0.5 * dr - 0.419 * dg - 0.081 * db
            pl_sq = float(np.sum(pl * pl))
            pc_sq = float(np.sum(0.5 * (pcb * pcb + pcr * pcr)))
            cnt = max(pl.size, 1)
            local_chr_lum = math.sqrt(pc_sq / (pl_sq + 1e-6))
            local_noise = math.sqrt(pl_sq / cnt)
            p_anom = _clamp01(
                0.65 * _sigmoid((0.16 - local_chr_lum) * 18.0)
                + 0.35 * _sigmoid((10.0 - local_noise) * 0.25)
            )
            grid_score[gy_i, gx_i] = p_anom

    patch_coverage_pct = float(np.mean(grid_score > 0.6) * 100.0)

    return {
        "width": w,
        "height": h,
        "jpeg_blockiness": jpeg_blockiness,
        "saturation_mean": saturation_mean,
        "chroma_to_luma": chroma_to_luma,
        "corr_rg": corr_rg,
        "corr_gb": corr_gb,
        "min_channel_corr": min(corr_rg, corr_gb),
        "smooth_noise": smooth_noise,
        "nz_smooth_noise": nz_smooth_noise,
        "zero_grad_pct": zero_grad_pct,
        "sharp_edge_pct": sharp_edge_pct,
        "lap_kurtosis": lap_kurtosis,
        "native_fft_size": fft_stats["native_fft_size"],
        "native_hi_mid_ratio": fft_stats["hi_mid_ratio"],
        "native_mid_low_ratio": fft_stats["mid_low_ratio"],
        "native_peak_med_ratio": fft_stats["peak_med_ratio"],
        "native_peak_med_p75": fft_stats["peak_med_p75"],
        "fft_log_mag_uint8": fft_log_uint8,
        "residual_map_01": residual_map_01,
        "gradcam_grid_16": grid_score,
        "patch_coverage_pct": patch_coverage_pct,
    }


def predict_multi_domain_cagu(rgb_uint8: np.ndarray) -> Dict[str, object]:
    """
    Runs the calibrated Multi-Domain Quality-Aware CAGU detector.
    Returns probabilities, branch sub-scores, dynamic CAGU weights, and forensic features.
    """
    feat = extract_multi_domain_features(rgb_uint8)
    w = int(feat["width"])
    h = int(feat["height"])
    blk = float(feat["jpeg_blockiness"])
    sat = float(feat["saturation_mean"])
    chr_lum = float(feat["chroma_to_luma"])
    corr_rg = float(feat["corr_rg"])
    corr_gb = float(feat["corr_gb"])
    min_corr = float(feat["min_channel_corr"])
    sm_noise = float(feat["smooth_noise"])
    nz_sm_noise = float(feat["nz_smooth_noise"])
    zero_grad_pct = float(feat["zero_grad_pct"])
    sharp_edge_pct = float(feat["sharp_edge_pct"])
    kurt = float(feat["lap_kurtosis"])
    hi_mid = float(feat["native_hi_mid_ratio"])
    mid_low = float(feat["native_mid_low_ratio"])
    peak_med_med = float(feat["native_peak_med_ratio"])
    peak_med_p75 = float(feat["native_peak_med_p75"])

    if blk > 1.22:
        compression_mode = "heavy_jpeg"
    elif blk > 1.08:
        compression_mode = "moderate_jpeg"
    else:
        compression_mode = "uncompressed"

    is_mono_or_low_sat = sat < 0.09
    is_graphic_composite = zero_grad_pct > 25.0 and sharp_edge_pct > 3.5 and sat > 0.25
    is_small_thumbnail = min(w, h) < 360 and compression_mode == "uncompressed"

    peak_med = peak_med_med if compression_mode == "heavy_jpeg" else peak_med_p75
    feat["native_peak_med_ratio"] = peak_med

    # Branch A: Native Multi-Crop Spectral Branch
    rolloff_pivot = 0.62 if is_small_thumbnail else 1.07
    spectral_rolloff = _sigmoid((rolloff_pivot - hi_mid) * 6.0)
    spectral_peak = _sigmoid((peak_med - 4.45) * 1.15)
    if compression_mode == "heavy_jpeg":
        mid_low_anom = _sigmoid((1.48 - mid_low) * 2.8)
        heavy_hi_mid_anom = _sigmoid((0.76 - hi_mid) * 5.0)
        if is_graphic_composite:
            freq_score = _clamp01(
                0.45 * _sigmoid((0.92 - hi_mid) * 6.0)
                + 0.35 * _sigmoid((1.55 - mid_low) * 3.0)
                + 0.20 * _sigmoid((peak_med_p75 - 4.2) * 1.1)
            )
        else:
            freq_score = _clamp01(0.45 * spectral_peak + 0.35 * mid_low_anom + 0.20 * heavy_hi_mid_anom)
    else:
        freq_score = _clamp01(0.65 * spectral_rolloff + 0.35 * spectral_peak)

    # Branch B: Spatial Sensor Noise & Kurtosis (excludes JPEG-zeroed constant blocks under heavy JPEG/thumbnail)
    eff_smooth = nz_sm_noise if (compression_mode == "heavy_jpeg" or is_small_thumbnail) else sm_noise
    smooth_th = 7.5 if compression_mode == "heavy_jpeg" else (6.0 if is_small_thumbnail else 9.8)
    smooth_slope = 0.90 if compression_mode == "heavy_jpeg" else 0.65
    kurt_th = 17.5 if compression_mode == "heavy_jpeg" else 22.0
    flux_smooth_sig = _sigmoid((smooth_th - eff_smooth) * smooth_slope) * _sigmoid((kurt - kurt_th) * 0.16)
    depressed_kurt_sig = _sigmoid((10.5 - kurt) * 0.65) * _sigmoid((sm_noise - 10.5) * 0.45)
    graphic_portrait_smooth = (
        _sigmoid((8.5 - sm_noise) * 0.8) * _sigmoid((kurt - 18.0) * 0.16)
        if is_graphic_composite
        else 0.0
    )
    spatial_score = _clamp01(max(flux_smooth_sig, depressed_kurt_sig * 0.92, graphic_portrait_smooth))

    # Branch C: Chrominance & Bayer Channel Coherence
    mean_corr = 0.5 * (corr_rg + corr_gb)
    raw_chroma_anom = _sigmoid((0.165 - chr_lum) * 26.0)
    lock_metric = (0.3 * mean_corr + 0.7 * min_corr) if chr_lum >= 0.165 else (0.5 * (mean_corr + min_corr))
    raw_lock_anom = _sigmoid((lock_metric - 0.942) * 42.0)
    chroma_neutralized = is_mono_or_low_sat or is_graphic_composite
    chroma_score = (
        0.45
        if chroma_neutralized
        else _clamp01(0.60 * raw_chroma_anom + 0.40 * raw_lock_anom)
    )

    # Quality-Aware CAGU Dynamic Weights
    raw_w_sp = 0.48 if compression_mode == "heavy_jpeg" else 0.32
    raw_w_fr = 0.44 if compression_mode == "heavy_jpeg" else 0.33
    raw_w_ch = 0.05 if chroma_neutralized else (0.08 if compression_mode == "heavy_jpeg" else 0.35)

    if min(w, h) >= 900 and compression_mode == "uncompressed":
        raw_w_sp = 0.36
        raw_w_ch = 0.42
        raw_w_fr = 0.22
    elif is_graphic_composite:
        raw_w_sp = 0.52
        raw_w_fr = 0.43
        raw_w_ch = 0.05

    w_sum = raw_w_sp + raw_w_fr + raw_w_ch
    w_sp = raw_w_sp / w_sum
    w_fr = raw_w_fr / w_sum
    w_ch = raw_w_ch / w_sum

    fused_raw = w_sp * spatial_score + w_fr * freq_score + w_ch * chroma_score

    # Dual-Domain Consistency Regularizer + Optical Bayer Independence Guard
    # Under heavy_jpeg, JPEG 4:2:0 subsampling suppresses chroma -> only spatial+freq vote for highAgree
    active_high = (
        [spatial_score, freq_score]
        if (compression_mode == "heavy_jpeg" or chroma_neutralized)
        else [spatial_score, freq_score, chroma_score]
    )
    active_low = [spatial_score, freq_score] if chroma_neutralized else [spatial_score, freq_score, chroma_score]
    high_agree = sum(1 for s in active_high if s > 0.55)
    low_agree = sum(1 for s in active_low if s < 0.35)

    strong_optical_bayer = (not chroma_neutralized) and (
        (chroma_score < 0.18 and chr_lum > 0.20) or (min_corr < 0.92 and chr_lum > 0.145)
    )
    extreme_latent_chroma = (
        (not chroma_neutralized) and compression_mode == "uncompressed" and chr_lum < 0.06 and min_corr > 0.99
    )

    consistency_factor = 1.0
    if strong_optical_bayer and not (compression_mode == "heavy_jpeg" and spatial_score > 0.75):
        consistency_factor = 0.72
        prob_ai = _clamp01(fused_raw * 0.65)
    elif high_agree >= 2:
        consistency_factor = 1.12
        prob_ai = _clamp01(0.55 + (fused_raw - 0.42) * 1.38)
    elif extreme_latent_chroma and freq_score > 0.35:
        consistency_factor = 1.08
        prob_ai = _clamp01(0.58 + (fused_raw - 0.40) * 1.25)
    elif low_agree >= 2 and fused_raw < 0.6:
        consistency_factor = 0.85
        prob_ai = _clamp01(fused_raw * 0.72)
    else:
        prob_ai = _clamp01(_sigmoid((fused_raw - 0.48) * 5.2))

    contributions = sorted(
        [
            ("Chroma & Bayer Coherence", w_ch * chroma_score),
            ("Native 2D-FFT Spectral", w_fr * freq_score),
            ("Sensor Noise & Kurtosis", w_sp * spatial_score),
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    primary_lead = contributions[0][0]

    if prob_ai >= 0.5:
        if is_graphic_composite:
            reasoning = (
                f"Phát hiện ảnh AI kết hợp chữ/đồ họa vector (mép chữ {sharp_edge_pct:.1f}%): CAGU cô lập nhiễu mép "
                f"kênh màu còn {w_ch * 100:.0f}% và xác nhận đặc trưng AI qua độ trơn nền & phổ 2D-FFT "
                f"({w_sp * 100:.0f}% + {w_fr * 100:.0f}%)."
            )
        elif compression_mode == "heavy_jpeg":
            reasoning = (
                f"CAGU phát hiện nén JPEG mạnh (blockiness={blk:.2f}), hạ trọng số sắc độ 4:2:0 xuống "
                f"{w_ch * 100:.0f}%, dồn {w_sp * 100:.0f}% Không gian + {w_fr * 100:.0f}% phổ 2D-FFT đa vùng và xác nhận ảnh AI."
            )
        elif flux_smooth_sig > 0.6:
            reasoning = (
                f"Vùng phẳng siêu mịn (smoothNoise={sm_noise:.1f}, kurtosis={kurt:.1f}) kết hợp khóa kênh RGB "
                f"(min_corr={min_corr:.3f}) — dấu hiệu đặc trưng của Latent Diffusion / FLUX."
            )
        else:
            reasoning = (
                f"Đồng thuận đa miền: phổ 2D-FFT native suy giảm bất thường (hi/mid={hi_mid:.2f}) "
                f"và dư sắc độ Chroma/Luma thấp ({chr_lum:.3f})."
            )
    else:
        if is_mono_or_low_sat:
            reasoning = (
                f"Ảnh có độ bão hòa thấp ({sat:.2f}); CAGU hạ trọng số kênh màu còn {w_ch * 100:.1f}% "
                "để tránh báo nhầm, xác nhận nhiễu và phổ quang học tự nhiên."
            )
        elif compression_mode == "heavy_jpeg":
            reasoning = (
                f"Ảnh nén JPEG mạng xã hội (blockiness={blk:.2f}): CAGU loại bỏ bội số lưới JPEG 8×8 & vùng phẳng "
                f"bị lượng tử hóa (nzNoise={nz_sm_noise:.1f}), hạ kênh sắc độ 4:2:0 còn {w_ch * 100:.0f}% và xác nhận ảnh chụp tự nhiên."
            )
        else:
            reasoning = (
                f"Đồng thuận quang học: nhiễu sắc độ Chroma/Luma ({chr_lum:.3f}, min_corr={min_corr:.3f}) "
                f"và phổ năng lượng native (hi/mid={hi_mid:.2f}) phù hợp cảm biến Bayer thực tế."
            )

    return {
        "probability_ai": prob_ai,
        "sub_scores": {
            "spatial_score": spatial_score,
            "freq_score": freq_score,
            "chroma_score": chroma_score,
            "consistency_factor": consistency_factor,
        },
        "cagu_stats": {
            "spatial_pct": w_sp * 100.0,
            "freq_pct": w_fr * 100.0,
            "chroma_pct": w_ch * 100.0,
            "compression_mode": compression_mode,
        },
        "features": feat,
        "primary_lead": primary_lead,
        "reasoning": reasoning,
        "quality_vector": [blk, sat, chr_lum, min_corr],
    }
