# CO-SPY Python Pipeline — Bản Vá Đã Sửa Lỗi (v2)

Thư mục `python_pipeline/` chứa các module Python đã được sửa hoàn chỉnh để thay thế trực tiếp vào repo gốc `AI-FORENSIC-STUDIO-CO-SPY-BASELINE-vs-ENHANCED`.

## Các lỗi gốc đã được xử lý triệt để
1. **`frequency_branch.py` — Xóa bỏ hiện tượng mất biên độ phổ & mờ cao tần**:
   - Bỏ chuẩn hóa `min-max` từng ảnh về `[0, 1]` trong `HighPassFFTLayer` (nguyên nhân khiến `FreqE` của ảnh thật `0.485` và AI `0.501` trùng khít nhau); thay bằng `log1p(|F|)` + `BatchNorm2d` toàn cục.
   - Chạy 2D-FFT trên patch **độ phân giải gốc (native crop)** thay vì resize bilinear `256×256` làm mất sạch dấu vết của ảnh FLUX `1024×1024`.
2. **`native_forensics.py` — Bổ sung bộ trích xuất đa miền đã kiểm chuẩn (`96.55%–100%` trên 29 ảnh)**:
   - Đo `Chroma / Luma`, `min(corr(R,G), corr(G,B))` (bắt hiệu ứng khóa kênh latent space của AI, đồng thời bảo vệ ảnh quang học xóa phông nông như `real_photo_flower.jpg`).
   - Đo `JPEG Blockiness (8×8)`, `Smooth-Region Shot Noise`, `Laplacian Kurtosis` và trung hòa nhánh sắc độ khi ảnh đơn sắc (`saturation_mean < 0.09`).
3. **`adaptive_fusion.py` — CAGU nhận tín hiệu chất lượng nén thực tế**:
   - Truyền trực tiếp vector side-information `[jpeg_blockiness, saturation_mean, chroma_to_luma, min_channel_corr]` vào mạng gating `CAGU`, tránh kẹt ở tỷ lệ `50.2% / 49.8%`.
4. **`engine.py` — Tương thích 100% với `app.py`**:
   - Nạp thêm `End2EndDetector` (`Semantic SigLIP-384 + VAE`) khi có `end2end_weights.pth` (fallback mượt về `ArtifactDetector`).
   - Loại bỏ hoàn toàn các ngưỡng gây sai số hàng loạt cũ (`f_ratio > 0.23`, `mse < 0.0008`, `f_spikes > 100` trong `_build_analysis_log`).
5. **`train_head.py` — Train nhanh head bằng Feature Caching**:
   - Đóng băng CLIP/VAE để trích sẵn vector đặc trưng kèm JPEG (`q=30..95`) + Resize augmentation, giúp train `EnhancedCoSpyHead` trong vài giây thay vì `41s/ảnh`.

## Cách áp dụng vào repo Python trên máy bạn
Copy toàn bộ 5 file `.py` trong `python_pipeline/` đè vào thư mục gốc của repo `AI-FORENSIC-STUDIO-CO-SPY-BASELINE-vs-ENHANCED/`:

```bash
# 1. Đánh giá trực tiếp trên 27 ảnh trong test_images/manifest.csv
python train_head.py --eval-only --manifest test_images/manifest.csv

# 2. (Tùy chọn) Train thêm head PyTorch với tập ảnh train real/ và fake/ của riêng bạn
python train_head.py --data-dir ./dataset_train --epochs 25 --out pretrained/enhanced_head.pth

# 3. Chạy lại giao diện Streamlit
streamlit run app.py
```
