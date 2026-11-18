# CO-SPY Forensic Studio — Full Python Project (`python_pipeline/`)

Toàn bộ dự án **CO-SPY Forensic Studio** đã được viết hoàn chỉnh bằng Python (gồm cả engine suy luận lai PyTorch/NumPy, giao diện Streamlit 3 tab, công cụ CLI, XAI Grad-CAM và script train/benchmark).

## Cấu trúc mã nguồn Python
- `native_forensics.py`: Bộ phân tích tín hiệu vật lý đa miền ở độ phân giải gốc (`Native 2D-FFT`, `Chroma/Luma`, `min RGB corr`, `Smooth Noise`, `Laplacian Kurtosis`, `JPEG 8×8 Blockiness` và van điều tiết CAGU).
- `frequency_branch.py`: Mạng `HighPassFFTLayer` + `FrequencyBranch` giữ nguyên biên độ phổ `log1p(|F|)` trên crop độ phân giải gốc (không chuẩn hóa min-max per-sample).
- `adaptive_fusion.py`: `CAGU` điều kiện hóa theo chất lượng nén + `DualDomainConsistency` + `EnhancedCoSpyHead`.
- `engine.py`: `CoSpyEngine` hỗ trợ cả chế độ PyTorch (`End2EndDetector` SigLIP-384 + VAE + `EnhancedCoSpyHead`) lẫn chế độ nhẹ CPU thuần NumPy/OpenCV, kèm đối chiếu trực tiếp Baseline v1 vs Enhanced v2.
- `gradcam.py`: `GradCAMExplainer` (PyTorch ResNet hook) và `NativeAnomalyLocalizer` (CPU gradient-free).
- `app.py`: Giao diện web Streamlit đầy đủ 3 tab (`Bàn Khám Nghiệm`, `Benchmark Thực Nghiệm`, `Chẩn Đoán Pipeline`).
- `cli.py`: Công cụ dòng lệnh kiểm tra 1 ảnh hoặc chạy benchmark toàn bộ `manifest.csv`.
- `train_head.py`: Script đóng băng CLIP/VAE, cache feature và train `FrequencyBranch` + `EnhancedCoSpyHead` kèm augmentation JPEG (`q=30..95`) & Resize.
- `requirements.txt`: Danh sách thư viện cài đặt.

## Cách chạy trên máy cá nhân
```bash
cd python_pipeline
pip install -r requirements.txt

# 1. Chạy giao diện Streamlit (tự động đọc ảnh mẫu tại ../public/dataset/ nếu chạy trong repo này, hoặc ./test_images/)
streamlit run app.py

# 2. Hoặc chạy benchmark nhanh bằng CLI (không cần GPU)
python cli.py --benchmark --manifest ../public/dataset/manifest.csv

# 3. Kiểm tra 1 ảnh cụ thể
python cli.py --image ../public/dataset/degraded/degraded_04.jpg

# 4. (Tùy chọn) Train head PyTorch nếu có tập ảnh real/ và fake/ riêng
python train_head.py --data-dir path/to/train_data --epochs 25 --out pretrained/enhanced_head.pth
```
