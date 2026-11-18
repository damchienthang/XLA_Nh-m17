import React, { useEffect, useRef, useState } from 'react';
import {
  DATASET_SPECIMENS,
  DatasetGroup,
  DatasetSpecimen,
  GROUP_LABELS,
} from '../forensics/dataset';
import { ColormapName } from '../forensics/colormaps';
import {
  ForensicLogEntry,
  ForensicResult,
  buildForensicLog,
} from '../forensics/analyzer';
import {
  paintFftSpectrumToCanvas,
  paintGradcamOverlayToCanvas,
  paintScalarHeatmapToCanvas,
} from '../forensics/browserRunner';
import { Play, SlidersHorizontal, Upload } from 'lucide-react';

interface WorkbenchViewProps {
  selectedSpecimen: DatasetSpecimen | null;
  customImage: { filename: string; url: string; width: number; height: number } | null;
  onSelectSpecimen: (spec: DatasetSpecimen) => void;
  onTriggerUpload: () => void;
  threshold: number;
  onThresholdChange: (val: number) => void;
  colormap: ColormapName;
  onColormapChange: (cm: ColormapName) => void;
  gain: number;
  onGainChange: (val: number) => void;
  result: ForensicResult | null;
  isAnalyzing: boolean;
  errorMsg: string | null;
  onRunAnalysis: () => void;
}

const GROUPS: DatasetGroup[] = ['degraded', 'ai_flux', 'ai_sd15', 'real', 'samples'];

export const WorkbenchView: React.FC<WorkbenchViewProps> = ({
  selectedSpecimen,
  customImage,
  onSelectSpecimen,
  onTriggerUpload,
  threshold,
  onThresholdChange,
  colormap,
  onColormapChange,
  gain,
  onGainChange,
  result,
  isAnalyzing,
  errorMsg,
  onRunAnalysis,
}) => {
  const [groupFilter, setGroupFilter] = useState<DatasetGroup | 'all'>('all');
  const [probeCoord, setProbeCoord] = useState<{ x: number; y: number } | null>(null);

  const gradcamCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const residualCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fftCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const activeUrl = customImage ? customImage.url : selectedSpecimen?.url ?? '';
  const activeName = customImage
    ? customImage.filename
    : selectedSpecimen?.filename ?? 'Chưa chọn tiêu bản';

  useEffect(() => {
    if (!result || !activeUrl) return;
    if (residualCanvasRef.current) {
      paintScalarHeatmapToCanvas(
        residualCanvasRef.current,
        result.residualMap01,
        result.visWidth,
        result.visHeight,
        colormap,
        gain
      );
    }
    if (gradcamCanvasRef.current) {
      paintGradcamOverlayToCanvas(
        gradcamCanvasRef.current,
        activeUrl,
        result.gradcamMap01,
        result.visWidth,
        result.visHeight,
        colormap,
        gain,
        0.48
      );
    }
    if (fftCanvasRef.current) {
      paintFftSpectrumToCanvas(
        fftCanvasRef.current,
        result.nativeFftLogMag,
        result.nativeFftSize,
        'INFERNO'
      );
    }
  }, [result, activeUrl, colormap, gain]);

  const filteredSpecimens =
    groupFilter === 'all'
      ? DATASET_SPECIMENS
      : DATASET_SPECIMENS.filter((s) => s.group === groupFilter);

  const enhIsAi = result ? result.enhancedProbAi >= threshold : false;
  const baseIsAi = result ? result.baselineProbAi >= threshold : false;
  const logEntries: ForensicLogEntry[] = result ? buildForensicLog(result, threshold) : [];

  const handleViewportMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const rx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const ry = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    const w = result?.width ?? customImage?.width ?? 256;
    const h = result?.height ?? customImage?.height ?? 256;
    setProbeCoord({
      x: Math.round(rx * w),
      y: Math.round(ry * h),
    });
  };

  return (
    <div className="mx-auto max-w-[1440px] px-4 py-5 sm:px-6 sm:py-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[330px_1fr]">
        {/* Left Control & Specimen Parameter Column */}
        <aside className="flex flex-col gap-5 rounded-lg border border-slate-800 bg-[#0D1322] p-5 self-start">
          <div>
            <h2 className="text-sm font-semibold text-slate-100">
              Chọn tiêu bản kiểm thử (29 ảnh)
            </h2>
            <p className="mt-1 text-xs text-slate-400">
              Chạy thuật toán pixel thực tế ngay trên ảnh thật COCO, SD-v1.5, FLUX.1-dev và ảnh nén mạng xã hội.
            </p>
          </div>

          <div className="flex flex-wrap gap-1.5 border-b border-slate-800 pb-3">
            <button
              type="button"
              onClick={() => setGroupFilter('all')}
              className={`rounded px-2.5 py-1 text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                groupFilter === 'all'
                  ? 'bg-cyan-500/20 text-cyan-300'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
            >
              Tất cả (29)
            </button>
            {GROUPS.map((grp) => (
              <button
                key={grp}
                type="button"
                onClick={() => setGroupFilter(grp)}
                className={`rounded px-2.5 py-1 text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                  groupFilter === grp
                    ? 'bg-cyan-500/20 text-cyan-300'
                    : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }`}
              >
                {grp === 'real'
                  ? 'Real'
                  : grp === 'ai_sd15'
                  ? 'SD1.5'
                  : grp === 'ai_flux'
                  ? 'FLUX'
                  : grp === 'degraded'
                  ? 'JPEG×2'
                  : 'Mẫu'}
              </button>
            ))}
          </div>

          <div className="max-h-[260px] space-y-1 overflow-y-auto pr-1">
            {filteredSpecimens.map((spec) => {
              const isSelected = !customImage && selectedSpecimen?.id === spec.id;
              return (
                <button
                  key={spec.id}
                  type="button"
                  onClick={() => onSelectSpecimen(spec)}
                  className={`flex w-full items-center justify-between gap-2 rounded px-2.5 py-2 text-left text-xs transition-colors cursor-pointer ${
                    isSelected
                      ? 'bg-cyan-500/15 text-cyan-200'
                      : 'text-slate-300 hover:bg-slate-800/70'
                  }`}
                >
                  <span className="truncate font-mono">{spec.filename}</span>
                  <span
                    className={`shrink-0 font-mono text-[11px] ${
                      spec.label === 1 ? 'text-rose-400' : 'text-emerald-400'
                    }`}
                  >
                    {spec.label === 1 ? 'AI' : 'REAL'}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="border-t border-slate-800 pt-4">
            <div className="mb-3 flex items-center gap-2 text-xs font-semibold text-slate-200">
              <SlidersHorizontal className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
              <span>Hiệu chuẩn tham số</span>
            </div>

            <div className="space-y-4">
              <div>
                <div className="mb-1.5 flex items-center justify-between text-xs">
                  <label htmlFor="threshold-slider" className="text-slate-400">
                    Ngưỡng phân loại AI
                  </label>
                  <span className="font-mono font-semibold text-cyan-400 tabular-nums">
                    {(threshold * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  id="threshold-slider"
                  type="range"
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  value={threshold}
                  onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
              </div>

              <div>
                <label htmlFor="colormap-select" className="mb-1.5 block text-xs text-slate-400">
                  Bảng màu sai phân quang phổ
                </label>
                <select
                  id="colormap-select"
                  value={colormap}
                  onChange={(e) => onColormapChange(e.target.value as ColormapName)}
                  className="w-full rounded border border-slate-700 bg-[#070B14] px-3 py-2 text-xs font-mono text-slate-200 focus:border-cyan-400 focus:outline-none"
                >
                  <option value="JET">JET (Chuẩn OpenCV)</option>
                  <option value="TURBO">TURBO (Tương phản cao)</option>
                  <option value="INFERNO">INFERNO (Nhiệt học)</option>
                  <option value="MAGMA">MAGMA</option>
                  <option value="BONE">BONE (Đơn sắc)</option>
                </select>
              </div>

              <div>
                <div className="mb-1.5 flex items-center justify-between text-xs">
                  <label htmlFor="gain-slider" className="text-slate-400">
                    Khuếch đại sai phân (Gain)
                  </label>
                  <span className="font-mono font-semibold text-cyan-400 tabular-nums">
                    {gain.toFixed(2)}×
                  </span>
                </div>
                <input
                  id="gain-slider"
                  type="range"
                  min={0.5}
                  max={3.0}
                  step={0.25}
                  value={gain}
                  onChange={(e) => onGainChange(parseFloat(e.target.value))}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-2 border-t border-slate-800 pt-4">
            <button
              type="button"
              onClick={onRunAnalysis}
              disabled={isAnalyzing || !activeUrl}
              className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-cyan-500 px-4 py-2.5 text-xs font-semibold text-slate-950 transition-colors hover:bg-cyan-400 disabled:opacity-50 cursor-pointer"
            >
              <Play className="h-3.5 w-3.5" aria-hidden="true" />
              <span>{isAnalyzing ? 'Đang tính toán đa miền...' : 'Phân tích lại tiêu bản'}</span>
            </button>
            <button
              type="button"
              onClick={onTriggerUpload}
              className="inline-flex w-full items-center justify-center gap-2 rounded-lg border border-slate-700 bg-transparent px-4 py-2 text-xs font-medium text-slate-300 transition-colors hover:border-slate-500 hover:text-slate-100 cursor-pointer"
            >
              <Upload className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Tải ảnh từ máy tính</span>
            </button>
          </div>
        </aside>

        {/* Main Forensic Inspection Stage */}
        <main className="min-w-0 space-y-6">
          {/* Specimen Header Metadata */}
          <div className="flex flex-wrap items-baseline justify-between gap-4 border-b border-slate-800 pb-4">
            <div>
              <h1 className="text-xl font-semibold tracking-tight text-slate-100">
                Bàn Khám Nghiệm Đa Miền: Baseline vs Enhanced v2
              </h1>
              <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-slate-400">
                <span className="font-mono text-slate-200">{activeName}</span>
                {selectedSpecimen && !customImage && (
                  <>
                    <span aria-hidden="true">·</span>
                    <span>{GROUP_LABELS[selectedSpecimen.group]}</span>
                    <span aria-hidden="true">·</span>
                    <span
                      className={`font-semibold ${
                        selectedSpecimen.label === 1 ? 'text-rose-400' : 'text-emerald-400'
                      }`}
                    >
                      Nhãn gốc: {selectedSpecimen.label === 1 ? 'AI-Generated' : 'Real Image'}
                    </span>
                  </>
                )}
                {result && (
                  <>
                    <span aria-hidden="true">·</span>
                    <span className="font-mono tabular-nums">
                      {result.width}×{result.height} px
                    </span>
                    <span aria-hidden="true">·</span>
                    <span className="font-mono tabular-nums">
                      {result.latencyMs.toFixed(1)} ms
                    </span>
                  </>
                )}
              </div>
            </div>

            {probeCoord && (
              <div className="font-mono text-xs text-cyan-400 tabular-nums">
                Tọa độ soi: x={probeCoord.x}, y={probeCoord.y}
              </div>
            )}
          </div>

          {errorMsg && (
            <div
              role="alert"
              className="rounded-lg border border-rose-500/40 bg-rose-950/30 px-4 py-3 text-xs text-rose-200"
            >
              {errorMsg}
            </div>
          )}

          {/* Side-by-Side Verdict Comparison (Baseline v1 vs Enhanced v2) */}
          {result && (
            <section
              aria-label="So sánh kết luận Baseline và Enhanced"
              className="grid grid-cols-1 gap-4 md:grid-cols-2"
            >
              {/* Baseline v1 */}
              <div className="rounded-lg border border-slate-800 bg-[#0D1322] p-5">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="text-xs font-medium text-slate-400">
                      CO-SPY Baseline (engine.py v1 · FFT 256px)
                    </div>
                    <div
                      className={`mt-1.5 text-lg font-semibold ${
                        baseIsAi ? 'text-rose-400' : 'text-emerald-400'
                      }`}
                    >
                      {baseIsAi ? '✖ Dự đoán: Ảnh AI' : '● Dự đoán: Ảnh Thật'}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-slate-400">Xác suất AI</div>
                    <div
                      data-testid="baseline-prob"
                      className={`mt-0.5 font-mono text-2xl font-bold tabular-nums ${
                        baseIsAi ? 'text-rose-400' : 'text-emerald-400'
                      }`}
                    >
                      {(result.baselineProbAi * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
                <p className="mt-3 border-t border-slate-800/80 pt-3 text-xs leading-relaxed text-slate-300">
                  {result.baselineReasoning}
                </p>
              </div>

              {/* Enhanced v2 */}
              <div className="rounded-lg border border-cyan-500/30 bg-[#0D1322] p-5">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="text-xs font-medium text-cyan-400">
                      CO-SPY Enhanced v2 (Multi-Domain CAGU)
                    </div>
                    <div
                      className={`mt-1.5 text-lg font-semibold ${
                        enhIsAi ? 'text-rose-400' : 'text-emerald-400'
                      }`}
                    >
                      {enhIsAi ? '✖ Phát hiện: Ảnh AI' : '● Xác nhận: Ảnh Thật'}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-slate-400">Xác suất AI</div>
                    <div
                      data-testid="enhanced-prob"
                      className={`mt-0.5 font-mono text-2xl font-bold tabular-nums ${
                        enhIsAi ? 'text-rose-400' : 'text-emerald-400'
                      }`}
                    >
                      {(result.enhancedProbAi * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
                <p className="mt-3 border-t border-slate-800/80 pt-3 text-xs leading-relaxed text-slate-200">
                  {result.enhancedReasoning}
                </p>
              </div>
            </section>
          )}

          {/* 4 Synchronized Forensic Viewports */}
          <section aria-label="Bốn khung quan sát pháp y số" className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-lg border border-slate-800 bg-[#0D1322] p-3">
              <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
                <span className="font-medium text-slate-200">01. Ảnh gốc đầu vào</span>
                <span className="font-mono tabular-nums">
                  {result ? `${result.width}×${result.height}` : '—'}
                </span>
              </div>
              <div
                onMouseMove={handleViewportMouseMove}
                onMouseLeave={() => setProbeCoord(null)}
                className="relative aspect-square w-full overflow-hidden rounded border border-slate-800 bg-[#070B14] cursor-crosshair"
              >
                {activeUrl ? (
                  <img
                    src={activeUrl}
                    alt={activeName}
                    referrerPolicy="no-referrer"
                    className="h-full w-full object-contain"
                  />
                ) : (
                  <div className="flex h-full items-center justify-center text-xs text-slate-500">
                    Chưa có ảnh
                  </div>
                )}
              </div>
              <p className="mt-2 text-[11px] text-slate-400">
                Soi trực tiếp pixel nguyên bản không qua nén lại.
              </p>
            </div>

            <div className="rounded-lg border border-slate-800 bg-[#0D1322] p-3">
              <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
                <span className="font-medium text-slate-200">02. Bản đồ vùng nghi vấn</span>
                <span className="font-mono tabular-nums">
                  {result ? `${result.features.patchAnomalyCoveragePct.toFixed(0)}% vùng` : '—'}
                </span>
              </div>
              <div
                onMouseMove={handleViewportMouseMove}
                onMouseLeave={() => setProbeCoord(null)}
                className="relative aspect-square w-full overflow-hidden rounded border border-slate-800 bg-[#070B14] cursor-crosshair"
              >
                <canvas
                  ref={gradcamCanvasRef}
                  data-testid="canvas-gradcam"
                  className="h-full w-full object-contain"
                />
              </div>
              <p className="mt-2 text-[11px] text-slate-400">
                Định vị cục bộ vùng có độ trơn sắc độ / nhiễu bất thường.
              </p>
            </div>

            <div className="rounded-lg border border-slate-800 bg-[#0D1322] p-3">
              <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
                <span className="font-medium text-slate-200">03. Sai phân SRM cao tần</span>
                <span className="font-mono tabular-nums">
                  {colormap} · {gain.toFixed(1)}×
                </span>
              </div>
              <div
                onMouseMove={handleViewportMouseMove}
                onMouseLeave={() => setProbeCoord(null)}
                className="relative aspect-square w-full overflow-hidden rounded border border-slate-800 bg-[#070B14] cursor-crosshair"
              >
                <canvas
                  ref={residualCanvasRef}
                  data-testid="canvas-residual"
                  className="h-full w-full object-contain"
                />
              </div>
              <p className="mt-2 text-[11px] text-slate-400">
                Bóc tách dư thông cao |I − blur(I)| loại bỏ nội dung cảnh.
              </p>
            </div>

            <div className="rounded-lg border border-slate-800 bg-[#0D1322] p-3">
              <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
                <span className="font-medium text-slate-200">04. Phổ 2D-FFT Native</span>
                <span className="font-mono tabular-nums">
                  {result ? `${result.nativeFftSize}×${result.nativeFftSize}` : '—'}
                </span>
              </div>
              <div
                onMouseMove={handleViewportMouseMove}
                onMouseLeave={() => setProbeCoord(null)}
                className="relative aspect-square w-full overflow-hidden rounded border border-slate-800 bg-[#070B14] cursor-crosshair"
              >
                <canvas
                  ref={fftCanvasRef}
                  data-testid="canvas-fft"
                  className="h-full w-full object-contain"
                />
              </div>
              <p className="mt-2 text-[11px] text-slate-400">
                Phổ log-magnitude trên patch gốc không resize bilinear.
              </p>
            </div>
          </section>

          {/* CAGU Dynamic Gating & Telemetry Strip */}
          {result && (
            <section className="rounded-lg border border-slate-800 bg-[#0D1322] p-5">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="text-sm font-semibold text-slate-100">
                    Van điều tiết thích ứng CAGU (Quality-Aware Adaptive Gating)
                  </h2>
                  <p className="mt-0.5 text-xs text-slate-400">
                    Trạng thái nén:{' '}
                    <span className="font-mono text-slate-200">
                      {result.cagu.compressionMode === 'heavy_jpeg'
                        ? 'Nén JPEG nặng (blockiness > 1.22)'
                        : result.cagu.compressionMode === 'moderate_jpeg'
                        ? 'Nén JPEG vừa'
                        : 'Không nén / PNG gốc'}
                    </span>{' '}
                    · Nhánh dẫn dắt: <span className="text-cyan-400">{result.primaryLead}</span>
                  </p>
                </div>
                <div className="flex items-center gap-4 text-xs font-mono tabular-nums">
                  <span className="text-cyan-400">
                    Không gian {result.cagu.spatialPct.toFixed(1)}%
                  </span>
                  <span className="text-amber-400">
                    Tần số {result.cagu.freqPct.toFixed(1)}%
                  </span>
                  <span className="text-emerald-400">
                    Kênh màu {result.cagu.chromaPct.toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* CAGU stacked proportional bar */}
              <div className="mt-3 flex h-2.5 w-full overflow-hidden rounded bg-slate-800">
                <div
                  style={{ width: `${result.cagu.spatialPct}%` }}
                  className="bg-cyan-500 transition-all duration-200"
                  title={`Không gian ${result.cagu.spatialPct.toFixed(1)}%`}
                />
                <div
                  style={{ width: `${result.cagu.freqPct}%` }}
                  className="bg-amber-500 transition-all duration-200"
                  title={`Tần số ${result.cagu.freqPct.toFixed(1)}%`}
                />
                <div
                  style={{ width: `${result.cagu.chromaPct}%` }}
                  className="bg-emerald-500 transition-all duration-200"
                  title={`Kênh màu ${result.cagu.chromaPct.toFixed(1)}%`}
                />
              </div>

              {/* 6-cell Precision Telemetry Grid */}
              <div className="mt-5 grid grid-cols-2 gap-3 border-t border-slate-800 pt-4 sm:grid-cols-3 lg:grid-cols-6">
                <div>
                  <div className="text-[11px] text-slate-400">Chroma / Luma</div>
                  <div className="mt-0.5 font-mono text-base font-semibold text-slate-100 tabular-nums">
                    {result.features.chromaToLuma.toFixed(3)}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-slate-400">Tương quan R–G</div>
                  <div className="mt-0.5 font-mono text-base font-semibold text-slate-100 tabular-nums">
                    {result.features.corrRG.toFixed(3)}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-slate-400">FFT hi/mid (Native)</div>
                  <div className="mt-0.5 font-mono text-base font-semibold text-slate-100 tabular-nums">
                    {result.features.nativeHighMidRatio.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-slate-400">Nhiễu vùng phẳng</div>
                  <div className="mt-0.5 font-mono text-base font-semibold text-slate-100 tabular-nums">
                    {result.features.smoothNoise.toFixed(1)}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-slate-400">Độ nhọn Kurtosis</div>
                  <div className="mt-0.5 font-mono text-base font-semibold text-slate-100 tabular-nums">
                    {result.features.lapKurtosis.toFixed(1)}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-slate-400">JPEG Blockiness</div>
                  <div className="mt-0.5 font-mono text-base font-semibold text-slate-100 tabular-nums">
                    {result.features.jpegBlockiness.toFixed(2)}
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* Analysis Decision Log */}
          {result && (
            <section className="rounded-lg border border-slate-800 bg-[#0D1322] p-5">
              <h2 className="text-sm font-semibold text-slate-100">
                Nhật ký bằng chứng phân tích (Forensic Decision Log)
              </h2>
              <div className="mt-3 divide-y divide-slate-800/80">
                {logEntries.map((item, idx) => {
                  const statusTone =
                    item.level === 'critical'
                      ? 'text-rose-400'
                      : item.level === 'warning'
                      ? 'text-amber-400'
                      : item.level === 'info'
                      ? 'text-cyan-400'
                      : 'text-emerald-400';
                  const statusGlyph =
                    item.level === 'critical'
                      ? '✖'
                      : item.level === 'warning'
                      ? '▲'
                      : item.level === 'info'
                      ? '◆'
                      : '●';
                  return (
                    <div
                      key={idx}
                      className="flex flex-col justify-between gap-2 py-3 sm:flex-row sm:items-start"
                    >
                      <div className="min-w-0">
                        <div className={`text-xs font-semibold ${statusTone}`}>
                          <span aria-hidden="true" className="mr-1.5">
                            {statusGlyph}
                          </span>
                          {item.title}
                        </div>
                        <p className="mt-1 text-xs leading-relaxed text-slate-400">
                          {item.detail}
                        </p>
                      </div>
                      <div className="shrink-0 font-mono text-xs font-medium text-slate-200 tabular-nums">
                        {item.value}
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  );
};
