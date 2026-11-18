import React from 'react';

export const DiagnosticsView: React.FC = () => {
  const issues = [
    {
      severity: 'Cao',
      module: 'engine.py · get_sd_artifact_detector()',
      symptom: 'Bỏ sót nhánh Semantic (CLIP ViT-SO400M-14-SigLIP-384) của CO-SPY gốc',
      detail:
        'Repo hiện tại chỉ load ArtifactDetector (VAE + ResNet-50), không load SemanticDetector hay End2EndDetector trong Co-Spy/detectors/sd-v1_4/. Theo Table 10 của paper CO-SPY, Only-Artifact đạt 76.10% trong khi kết hợp Semantic đạt 87.67% và kháng nén JPEG vượt trội.',
      fix: 'Load End2EndDetector (semantic + artifact + 2 regulators) từ Co-Spy/detectors/sd-v1_4/end2end.py làm backbone chính.',
    },
    {
      severity: 'Cao',
      module: 'engine.py · compute_fft_spectrum / analyze_image (dòng 373–399)',
      symptom: 'Ngưỡng f_ratio > 0.23 kích hoạt 0/29 ảnh thật/AI/degraded trong bộ test',
      detail:
        'Resize ảnh về 256×256 bằng bilinear trước khi tính FFT xóa mất cao tần của ảnh 1024px (FLUX). Đồng thời JPEG nén 2 lần làm f_ratio tụt xuống 0.077–0.138, trong khi log cũ báo sai f_spikes > 100 trên 100% ảnh thật.',
      fix: 'Tính 2D-FFT trên patch crop độ phân giải gốc (không resize) của ảnh dư khử nội dung (I − blur), đo Peak-to-Median cục bộ và phổ High/Mid.',
    },
    {
      severity: 'Cao',
      module: 'frequency_branch.py · HighPassFFTLayer (dòng 60–63)',
      symptom: 'Chuẩn hóa min-max từng ảnh về [0, 1] làm mất độ lớn năng lượng tần số',
      detail:
        'Ép recon_norm về [0, 1] theo từng sample khiến FreqE của ảnh thật (0.485–0.494) và ảnh AI (0.501–0.503) trùng khít nhau hoàn toàn trong notebook.',
      fix: 'Bỏ min-max per-sample; dùng log1p(|F|) chuẩn hóa theo BatchNorm/LayerNorm toàn cục để bảo toàn biên độ phổ.',
    },
    {
      severity: 'Trung bình',
      module: 'adaptive_fusion.py & evaluation_metrics.ipynb',
      symptom: 'EnhancedCoSpyHead & FrequencyBranch chạy với trọng số ngẫu nhiên chưa train',
      detail:
        'Trong Cell 6 của notebook, head chưa được fine-tune nên CAGU luôn chia 50.2% / 49.8% và Accuracy trên 20 ảnh test bằng 50.0% (Recall = 0%).',
      fix: 'Đóng băng CLIP & VAE, trích xuất sẵn feature vector ra đĩa rồi train riêng FrequencyBranch + CAGU trên DRCT + JPEG/Resize augmentation (quality 30–95).',
    },
  ];

  return (
    <div className="mx-auto max-w-[1440px] space-y-6 px-6 py-6">
      <div className="border-b border-slate-800 pb-5">
        <h1 className="text-xl font-semibold tracking-tight text-slate-100">
          Chẩn Đoán Nguyên Nhân & Lộ Trình Sửa Pipeline Python
        </h1>
        <p className="mt-1 text-xs text-slate-400">
          Đối chiếu trực tiếp giữa bài báo CO-SPY (arXiv:2503.18286), mã nguồn hiện tại trong repo và các chỉ số đo thực tế trên 29 ảnh kiểm thử.
        </p>
      </div>

      {/* 4 Root-Cause Diagnostic Cards */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {issues.map((item, idx) => (
          <div key={idx} className="rounded-lg border border-slate-800 bg-[#0D1322] p-5">
            <div className="flex items-center justify-between gap-3">
              <span className="font-mono text-xs text-cyan-400">{item.module}</span>
              <span
                className={`text-xs font-semibold ${
                  item.severity === 'Cao' ? 'text-rose-400' : 'text-amber-400'
                }`}
              >
                Mức độ: {item.severity}
              </span>
            </div>
            <h2 className="mt-2 text-sm font-semibold text-slate-100">{item.symptom}</h2>
            <p className="mt-2 text-xs leading-relaxed text-slate-400">{item.detail}</p>
            <div className="mt-3 border-t border-slate-800 pt-3 text-xs text-emerald-300">
              <span className="font-semibold">Hướng sửa: </span>
              {item.fix}
            </div>
          </div>
        ))}
      </div>

      {/* Recommended Python Architecture Snippet */}
      <div className="rounded-lg border border-slate-800 bg-[#0D1322] p-5">
        <h2 className="text-sm font-semibold text-slate-100">
          Kiến trúc đề xuất khi sửa lại repo Python (sau khi kiểm thử xong tại đây)
        </h2>
        <p className="mt-1 text-xs text-slate-400">
          Kết hợp 3 nguồn tín hiệu bổ trợ lẫn nhau: (1) CLIP SigLIP-384 (Semantic kháng JPEG) + (2) SD-v1.4 VAE + SRM/Native 2D-FFT (Artifact không gian & tần số) + (3) JPEG Blockiness-Conditioned CAGU.
        </p>

        <pre className="mt-4 overflow-x-auto rounded border border-slate-800 bg-[#070B14] p-4 font-mono text-xs leading-relaxed text-slate-200">
{`# 1. Load đầy đủ End2EndDetector (Semantic SigLIP-384 + VAE Artifact)
from detectors.sd_v1_4.end2end import End2EndDetector
detector = End2EndDetector()
detector.load_weights("Co-Spy/pretrained/sd-v1_4/end2end_weights.pth")

# 2. Nhánh Tần số trên Native Crop (KHÔNG resize bilinear về 256px)
#    và bỏ min-max normalization theo từng sample trong HighPassFFTLayer
residual = x_native_crop - F.avg_pool2d(x_native_crop, 3, stride=1, padding=1)
fft_mag = torch.log1p(torch.abs(torch.fft.fftshift(torch.fft.fft2(residual))))

# 3. Đưa JPEG Blockiness + Chroma-to-Luma Ratio làm side-information cho CAGU
#    để tự động tăng trọng số CLIP Semantic + Mid-band FFT khi ảnh bị nén mạng xã hội`}
        </pre>
      </div>
    </div>
  );
};
