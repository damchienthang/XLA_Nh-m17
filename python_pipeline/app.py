"""
app.py — CO-SPY Forensic Studio (Full Python Streamlit Edition)
===============================================================
Clean dark-obsidian forensic workbench in Python matching the React Studio:
  - Tab 1: Bàn Khám Nghiệm Đa Miền (Baseline v1 vs Enhanced v2, 4 viewports, CAGU bar, 6 telemetry metrics, Decision Log)
  - Tab 2: Benchmark Thực Nghiệm (Live batch evaluation on test_images/manifest.csv + samples/, 6 metrics, Confusion Matrices, ROC Curve)
  - Tab 3: Chẩn Đoán Pipeline & Kiến Trúc (Root-cause diagnostics & Python pipeline configuration)

Run:
  streamlit run app.py
"""

import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from engine import HAS_TORCH, CoSpyEngine, _build_analysis_log
from train_head import _compute_metrics

st.set_page_config(
    page_title="CO-SPY Forensic Studio (Python)",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean Dark Obsidian Theme (#070B14 / #0D1322) — zero pill clutter or pulsing dots
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    background-color: #070B14;
    color: #F1F5F9;
}
.stApp { background-color: #070B14; }
section[data-testid="stSidebar"] {
    background-color: #0B1120;
    border-right: 1px solid #1E293B;
}
.mono-num {
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
}
.card-box {
    background-color: #0D1322;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 16px 20px;
}
.card-box-accent {
    background-color: #0D1322;
    border: 1px solid rgba(6, 182, 212, 0.35);
    border-radius: 8px;
    padding: 16px 20px;
}
.telemetry-cell {
    background-color: #0D1322;
    border: 1px solid #1E293B;
    border-radius: 6px;
    padding: 12px 14px;
}
.telemetry-label {
    font-size: 0.72rem;
    color: #94A3B8;
    margin-bottom: 4px;
}
.telemetry-val {
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
    font-size: 1.05rem;
    font-weight: 600;
    color: #F8FAFC;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engine() -> CoSpyEngine:
    return CoSpyEngine()


def _discover_specimens() -> List[Dict[str, object]]:
    """Looks for test_images/manifest.csv or ../public/dataset/manifest.csv."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "test_images"),
        os.path.abspath(os.path.join(base_dir, "..", "public", "dataset")),
    ]
    specimens: List[Dict[str, object]] = []
    for root in candidates:
        manifest = os.path.join(root, "manifest.csv")
        if os.path.exists(manifest):
            with open(manifest, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    full = os.path.join(root, row["file"])
                    if os.path.exists(full):
                        specimens.append(
                            {
                                "id": row["file"],
                                "filename": os.path.basename(row["file"]),
                                "group": row.get("group", "other"),
                                "label": 1 if row["label"].strip().lower() in ("fake", "ai", "1") else 0,
                                "path": full,
                            }
                        )
            # Also include samples/
            samples_dir = os.path.join(root, "samples")
            if not os.path.isdir(samples_dir):
                samples_dir = os.path.join(os.path.dirname(root), "samples")
            if os.path.isdir(samples_dir):
                for fname in sorted(os.listdir(samples_dir)):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        specimens.append(
                            {
                                "id": f"samples/{fname}",
                                "filename": fname,
                                "group": "samples",
                                "label": 1 if ("ai" in fname.lower() or "fake" in fname.lower()) else 0,
                                "path": os.path.join(samples_dir, fname),
                            }
                        )
            break
    return specimens


engine = load_engine()
specimens = _discover_specimens()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### CO-SPY Forensic Studio")
    st.caption("Python Multi-Domain CAGU Edition")
    st.divider()

    uploaded = st.file_uploader(
        "Tải ảnh từ máy tính",
        type=["png", "jpg", "jpeg", "webp"],
    )

    selected_spec = None
    if specimens and uploaded is None:
        groups = ["Tất cả"] + sorted({s["group"] for s in specimens})
        grp = st.selectbox("Lọc nhóm tiêu bản:", groups, index=0)
        filtered = specimens if grp == "Tất cả" else [s for s in specimens if s["group"] == grp]
        default_idx = next((i for i, s in enumerate(filtered) if "degraded_04" in s["filename"]), 0)
        selected_idx = st.selectbox(
            f"Chọn tiêu bản ({len(filtered)} ảnh):",
            range(len(filtered)),
            index=default_idx,
            format_func=lambda i: f"{filtered[i]['filename']} [{'AI' if filtered[i]['label']==1 else 'REAL'}]",
        )
        selected_spec = filtered[selected_idx]

    st.divider()
    st.markdown("**Hiệu chuẩn tham số**")
    threshold = st.slider("Ngưỡng phân loại AI:", 0.10, 0.90, 0.50, 0.05)
    colormap = st.selectbox("Bảng màu sai phân quang phổ:", ["JET", "TURBO", "INFERNO", "MAGMA", "BONE"], index=0)
    gain = st.slider("Khuếch đại sai phân (Gain):", 0.50, 3.00, 1.00, 0.25)

    st.divider()
    backend_label = (
        f"PyTorch ({engine.device.upper()})"
        + (" + End2End" if engine.has_sd_end2end_weights() else " + Multi-Domain CAGU" if engine.has_sd_artifact_weights() else "")
        if HAS_TORCH
        else "NumPy + OpenCV Multi-Domain CAGU"
    )
    st.caption(f"Engine: {backend_label}")


# --- TOP TABS ---
tab_work, tab_bench, tab_diag = st.tabs([
    "Bàn Khám Nghiệm",
    "Benchmark Thực Nghiệm",
    "Chẩn Đoán Pipeline",
])

with tab_work:
    current_image = None
    active_name = "Chưa có ảnh"
    ground_truth = None

    if uploaded is not None:
        current_image = Image.open(uploaded).convert("RGB")
        active_name = uploaded.name
    elif selected_spec is not None:
        current_image = Image.open(selected_spec["path"]).convert("RGB")
        active_name = selected_spec["filename"]
        ground_truth = "AI-Generated" if selected_spec["label"] == 1 else "Real Image"

    if current_image is None:
        st.info("Hãy tải ảnh lên ở thanh bên hoặc đặt thư mục `test_images/` cạnh `app.py`.")
    else:
        res = engine.analyze_image(current_image, colormap_name=colormap, gain=gain)
        base = res["baseline_info"]
        enh_prob = float(res["probability_ai"])
        base_prob = float(base["probability_ai"])
        enh_is_ai = enh_prob >= threshold
        base_is_ai = base_prob >= threshold

        meta_parts = [f"`{active_name}`", f"`{current_image.width}×{current_image.height} px`", f"`{res['latency_sec']*1000:.1f} ms`"]
        if ground_truth:
            meta_parts.insert(1, f"Nhãn gốc: **{ground_truth}**")
        st.markdown("### Bàn Khám Nghiệm Đa Miền: Baseline vs Enhanced v2")
        st.markdown(" · ".join(meta_parts))

        # Side-by-side comparison cards
        c_base, c_enh = st.columns(2)
        with c_base:
            b_color = "#F43F5E" if base_is_ai else "#10B981"
            b_txt = "✖ Dự đoán: Ảnh AI" if base_is_ai else "● Dự đoán: Ảnh Thật"
            st.markdown(
                f"""
            <div class="card-box">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <div style="font-size:0.78rem; color:#94A3B8;">CO-SPY Baseline (engine.py v1 · FFT 256px)</div>
                        <div style="font-size:1.15rem; font-weight:600; color:{b_color}; margin-top:4px;">{b_txt}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.75rem; color:#94A3B8;">Xác suất AI</div>
                        <div class="mono-num" style="font-size:1.55rem; font-weight:700; color:{b_color};">{base_prob*100:.1f}%</div>
                    </div>
                </div>
                <div style="margin-top:10px; padding-top:10px; border-top:1px solid #1E293B; font-size:0.8rem; color:#CBD5E1;">
                    {base['reasoning']}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with c_enh:
            e_color = "#F43F5E" if enh_is_ai else "#10B981"
            e_txt = "✖ Phát hiện: Ảnh AI" if enh_is_ai else "● Xác nhận: Ảnh Thật"
            st.markdown(
                f"""
            <div class="card-box-accent">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <div style="font-size:0.78rem; color:#22D3EE;">CO-SPY Enhanced v2 (Multi-Domain CAGU)</div>
                        <div style="font-size:1.15rem; font-weight:600; color:{e_color}; margin-top:4px;">{e_txt}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.75rem; color:#94A3B8;">Xác suất AI</div>
                        <div class="mono-num" style="font-size:1.55rem; font-weight:700; color:{e_color};">{enh_prob*100:.1f}%</div>
                    </div>
                </div>
                <div style="margin-top:10px; padding-top:10px; border-top:1px solid #1E293B; font-size:0.8rem; color:#E2E8F0;">
                    {res['verdict_reasoning']}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # 4 Synchronized Viewports
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            st.markdown("**01. Ảnh gốc đầu vào**")
            st.image(current_image, use_container_width=True)
            st.caption("Pixel nguyên bản không nén lại.")
        with v2:
            cov = res["native_features"].get("patch_coverage_pct", 0.0)
            st.markdown(f"**02. Bản đồ vùng nghi vấn** (`{cov:.0f}%`)")
            st.image(res["heatmap_image"], use_container_width=True)
            st.caption("Định vị cục bộ vùng trơn sắc độ / nhiễu.")
        with v3:
            st.markdown(f"**03. Sai phân SRM cao tần** (`{colormap}`)")
            st.image(res["residual_image"], use_container_width=True)
            st.caption("Dư thông cao |I − blur(I)|.")
        with v4:
            sz = int(res["native_features"].get("native_fft_size", 128))
            st.markdown(f"**04. Phổ 2D-FFT Native** (`{sz}×{sz}`)")
            st.image(res["fft_image"], use_container_width=True)
            st.caption("Log-magnitude trên patch gốc.")

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # CAGU Proportional Bar
        cagu = res["cagu_stats"]
        sp_w = cagu["spatial_pct"]
        fr_w = cagu["freq_pct"]
        ch_w = cagu["chroma_pct"]
        lead = res["ensemble_info"]["primary_lead"]
        mode_txt = (
            "Nén JPEG nặng (>1.22)"
            if cagu["compression_mode"] == "heavy_jpeg"
            else ("Nén JPEG vừa" if cagu["compression_mode"] == "moderate_jpeg" else "Không nén / PNG gốc")
        )
        st.markdown(
            f"""
        <div class="card-box">
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;">
                <div>
                    <div style="font-size:0.9rem; font-weight:600;">Van điều tiết thích ứng CAGU (Quality-Aware Adaptive Gating)</div>
                    <div style="font-size:0.78rem; color:#94A3B8; margin-top:2px;">
                        Trạng thái nén: <span style="color:#E2E8F0;">{mode_txt}</span> · Nhánh dẫn dắt: <span style="color:#22D3EE;">{lead}</span>
                    </div>
                </div>
                <div class="mono-num" style="font-size:0.8rem;">
                    <span style="color:#22D3EE;">Không gian {sp_w:.1f}%</span> &nbsp;·&nbsp;
                    <span style="color:#F59E0B;">Tần số {fr_w:.1f}%</span> &nbsp;·&nbsp;
                    <span style="color:#10B981;">Kênh màu {ch_w:.1f}%</span>
                </div>
            </div>
            <div style="margin-top:10px; display:flex; height:10px; border-radius:4px; overflow:hidden; background:#1E293B;">
                <div style="width:{sp_w}%; background:#06B6D4;"></div>
                <div style="width:{fr_w}%; background:#F59E0B;"></div>
                <div style="width:{ch_w}%; background:#10B981;"></div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        # 6 Precision Telemetry Cells
        m_items = list(res["metrics"].items())
        t_cols = st.columns(len(m_items))
        for idx, (k, v) in enumerate(m_items):
            with t_cols[idx]:
                st.markdown(
                    f"""
                <div class="telemetry-cell">
                    <div class="telemetry-label">{k}</div>
                    <div class="telemetry-val">{v}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Decision Log
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Nhật ký bằng chứng phân tích (Forensic Decision Log)")
        for entry in _build_analysis_log(res, threshold=threshold):
            lvl = entry["level"]
            t_col = "#F43F5E" if lvl == "critical" else ("#F59E0B" if lvl == "warning" else ("#22D3EE" if lvl == "info" else "#10B981"))
            st.markdown(
                f"""
            <div style="padding:10px 14px; border-bottom:1px solid #1E293B; display:flex; justify-content:space-between; align-items:flex-start; gap:12px;">
                <div>
                    <div style="font-size:0.82rem; font-weight:600; color:{t_col};">{entry['icon']} {entry['title']}</div>
                    <div style="font-size:0.78rem; color:#94A3B8; margin-top:2px;">{entry['detail']}</div>
                </div>
                <div class="mono-num" style="font-size:0.8rem; color:#E2E8F0; white-space:nowrap;">{entry['value']}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


with tab_bench:
    st.markdown("### Benchmark Thực Nghiệm trên Bộ Dữ Liệu Thực Tế")
    if not specimens:
        st.warning("Chưa tìm thấy `test_images/manifest.csv` hoặc `../public/dataset/manifest.csv`.")
    else:
        if st.button(f"Chạy Benchmark Toàn Bộ {len(specimens)} Ảnh", type="primary"):
            prog = st.progress(0.0)
            records = []
            for i, s in enumerate(specimens):
                img = Image.open(s["path"]).convert("RGB")
                r = engine.analyze_image(img)
                records.append(
                    {
                        "Tệp ảnh": s["filename"],
                        "Nhóm": s["group"],
                        "label": s["label"],
                        "Nhãn gốc": "AI" if s["label"] == 1 else "REAL",
                        "base_prob": r["baseline_info"]["probability_ai"],
                        "enh_prob": r["probability_ai"],
                        "Chroma/Luma": round(r["native_features"]["chroma_to_luma"], 3),
                        "FFT hi/mid": round(r["native_features"]["native_hi_mid_ratio"], 2),
                        "JPEG Blk": round(r["native_features"]["jpeg_blockiness"], 2),
                        "Đúng?": "● Đúng" if (r["probability_ai"] >= threshold) == (s["label"] == 1) else "✖ Sai",
                    }
                )
                prog.progress((i + 1) / len(specimens))
            st.session_state["py_bench_rows"] = records

        rows = st.session_state.get("py_bench_rows", [])
        if rows:
            labels = [r["label"] for r in rows]
            b_probs = [r["base_prob"] for r in rows]
            e_probs = [r["enh_prob"] for r in rows]
            bm = _compute_metrics(labels, b_probs, threshold)
            em = _compute_metrics(labels, e_probs, threshold)

            col_tbl, col_roc = st.columns([2, 1])
            with col_tbl:
                df_summary = pd.DataFrame(
                    [
                        {
                            "Mô hình": "Baseline (engine.py v1)",
                            "Accuracy": f"{bm['accuracy']:.1f}%",
                            "Precision": f"{bm['precision']:.1f}%",
                            "Recall": f"{bm['recall']:.1f}%",
                            "Specificity": f"{bm['specificity']:.1f}%",
                            "F1-Score": f"{bm['f1']:.4f}",
                            "AUC-ROC": f"{bm['auc_roc']:.4f}",
                        },
                        {
                            "Mô hình": "Enhanced v2 (Multi-Domain CAGU)",
                            "Accuracy": f"{em['accuracy']:.1f}%",
                            "Precision": f"{em['precision']:.1f}%",
                            "Recall": f"{em['recall']:.1f}%",
                            "Specificity": f"{em['specificity']:.1f}%",
                            "F1-Score": f"{em['f1']:.4f}",
                            "AUC-ROC": f"{em['auc_roc']:.4f}",
                        },
                    ]
                )
                st.dataframe(df_summary, use_container_width=True, hide_index=True)
                st.markdown(
                    f"**Confusion Matrix — Baseline:** `TN={bm['tn']}, FP={bm['fp']}, FN={bm['fn']}, TP={bm['tp']}` &nbsp;|&nbsp; "
                    f"**Enhanced v2:** `TN={em['tn']}, FP={em['fp']}, FN={em['fn']}, TP={em['tp']}`"
                )

            with col_roc:
                def _roc_pts(probs_list):
                    ths = sorted(set([1.01, -0.01] + probs_list), reverse=True)
                    pos = max(sum(labels), 1)
                    neg = max(len(labels) - sum(labels), 1)
                    xs, ys = [], []
                    for th in ths:
                        xs.append(sum(1 for y, p in zip(labels, probs_list) if y == 0 and p >= th) / neg)
                        ys.append(sum(1 for y, p in zip(labels, probs_list) if y == 1 and p >= th) / pos)
                    return xs, ys

                bx, by = _roc_pts(b_probs)
                ex, ey = _roc_pts(e_probs)
                fig, ax = plt.subplots(figsize=(4.2, 3.6), facecolor="#0D1322")
                ax.set_facecolor("#070B14")
                ax.plot([0, 1], [0, 1], "--", color="#475569", lw=1)
                ax.plot(bx, by, "--", color="#F59E0B", lw=1.8, label=f"Baseline ({bm['auc_roc']:.3f})")
                ax.plot(ex, ey, "-", color="#06B6D4", lw=2.4, label=f"Enhanced ({em['auc_roc']:.3f})")
                ax.set_xlabel("FPR", color="#94A3B8", fontsize=9)
                ax.set_ylabel("TPR", color="#94A3B8", fontsize=9)
                ax.tick_params(colors="#94A3B8", labelsize=8)
                ax.legend(facecolor="#0D1322", edgecolor="#1E293B", labelcolor="#E2E8F0", fontsize=8)
                st.pyplot(fig)

            st.markdown("#### Chi tiết từng tiêu bản")
            df_detail = pd.DataFrame(
                [
                    {
                        "Tệp ảnh": r["Tệp ảnh"],
                        "Nhóm": r["Nhóm"],
                        "Nhãn gốc": r["Nhãn gốc"],
                        "Base Prob": f"{r['base_prob']*100:.1f}%",
                        "Enh v2 Prob": f"{r['enh_prob']*100:.1f}%",
                        "Chroma/Luma": r["Chroma/Luma"],
                        "FFT hi/mid": r["FFT hi/mid"],
                        "JPEG Blk": r["JPEG Blk"],
                        "Kết quả": r["Đúng?"],
                    }
                    for r in rows
                ]
            )
            st.dataframe(df_detail, use_container_width=True, hide_index=True)


with tab_diag:
    st.markdown("### Chẩn Đoán Nguyên Nhân & Bản Vá Pipeline Python")
    st.markdown(
        """
| Module | Vấn đề ở bản v1 | Cách đã sửa ở bản v2 |
|---|---|---|
| `frequency_branch.py` | Chuẩn hóa min-max `[0, 1]` theo từng sample làm mất chênh lệch năng lượng `FreqE` (`0.49` vs `0.50`); resize bilinear `256×256` xóa dấu vết FLUX `1024px`. | Giữ nguyên biên độ `log1p(\|F\|)` + `BatchNorm2d` toàn cục; chạy 2D-FFT trên patch **độ phân giải gốc**. |
| `native_forensics.py` | Thiếu tín hiệu vật lý của cảm biến quang học thực tế. | Bổ sung `Chroma/Luma`, `min(corr(R,G), corr(G,B))`, `Smooth Noise`, `Laplacian Kurtosis`, `JPEG Blockiness 8×8` & màng lọc xóa phông nông. |
| `adaptive_fusion.py` | `EnhancedCoSpyHead` không nhận biết được mức nén JPEG nên CAGU kẹt ở `50.2% / 49.8%`. | Truyền trực tiếp vector `[jpeg_blockiness, saturation, chroma_to_luma, min_corr]` vào van điều tiết CAGU. |
| `engine.py` | Chỉ nạp `ArtifactDetector`, bỏ sót `End2EndDetector` (SigLIP-384), ngưỡng `f_ratio > 0.23` kích hoạt 0/29 ảnh và log `f_spikes > 100` báo nhầm 100% ảnh thật. | Nạp `End2EndDetector` nếu có, kết hợp `Native Multi-Domain CAGU` và sửa `_build_analysis_log`. |
| `train_head.py` | Notebook loop qua CLIP/VAE mỗi epoch tốn `41,357 ms/ảnh`. | Đóng băng CLIP/VAE, cache feature ra RAM/đĩa và train `EnhancedCoSpyHead` trong vài giây. |
"""
    )
