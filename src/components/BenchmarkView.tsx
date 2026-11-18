import React, { useState } from 'react';
import {
  DATASET_SPECIMENS,
  DatasetGroup,
  DatasetSpecimen,
} from '../forensics/dataset';
import {
  BinaryMetrics,
  ForensicResult,
  RocPoint,
  analyzeImagePixels,
  computeBinaryMetrics,
  computeRocCurve,
} from '../forensics/analyzer';
import { decodeImageUrlToPixels } from '../forensics/browserRunner';
import { Play } from 'lucide-react';

export interface BenchmarkRow {
  specimen: DatasetSpecimen;
  result: ForensicResult;
}

interface BenchmarkViewProps {
  rows: BenchmarkRow[];
  onBenchmarkComplete: (rows: BenchmarkRow[]) => void;
  threshold: number;
  onInspectRow: (specimen: DatasetSpecimen) => void;
}

export const BenchmarkView: React.FC<BenchmarkViewProps> = ({
  rows,
  onBenchmarkComplete,
  threshold,
  onInspectRow,
}) => {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [groupFilter, setGroupFilter] = useState<DatasetGroup | 'all'>('all');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const runFullBenchmark = async () => {
    setIsRunning(true);
    setProgress(0);
    setErrorMsg(null);
    try {
      const collected: BenchmarkRow[] = [];
      for (let i = 0; i < DATASET_SPECIMENS.length; i++) {
        const spec = DATASET_SPECIMENS[i];
        const { rgba, width, height } = await decodeImageUrlToPixels(spec.url);
        const result = analyzeImagePixels(rgba, width, height);
        collected.push({ specimen: spec, result });
        setProgress(i + 1);
      }
      onBenchmarkComplete(collected);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Lỗi khi chạy benchmark');
    } finally {
      setIsRunning(false);
    }
  };

  const labels = rows.map((r) => r.specimen.label);
  const baseProbs = rows.map((r) => r.result.baselineProbAi);
  const enhProbs = rows.map((r) => r.result.enhancedProbAi);

  const baseMetrics: BinaryMetrics | null =
    rows.length > 0 ? computeBinaryMetrics(labels, baseProbs, threshold) : null;
  const enhMetrics: BinaryMetrics | null =
    rows.length > 0 ? computeBinaryMetrics(labels, enhProbs, threshold) : null;

  const rocBase = rows.length > 0 ? computeRocCurve(labels, baseProbs) : null;
  const rocEnh = rows.length > 0 ? computeRocCurve(labels, enhProbs) : null;

  const avgLatency =
    rows.length > 0
      ? rows.reduce((acc, r) => acc + r.result.latencyMs, 0) / rows.length
      : 0;

  const visibleRows =
    groupFilter === 'all'
      ? rows
      : rows.filter((r) => r.specimen.group === groupFilter);

  const toPolyline = (pts: RocPoint[]) =>
    pts
      .map((p) => {
        const x = 32 + p.fpr * 240;
        const y = 264 - p.tpr * 240;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

  return (
    <div className="mx-auto max-w-[1440px] space-y-6 px-6 py-6">
      {/* Header & Runner Trigger */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-slate-100">
            Benchmark Thực Nghiệm trên 29 Ảnh Thực Tế
          </h1>
          <p className="mt-1 text-xs text-slate-400">
            Chạy suy luận trực tiếp trên 7 ảnh COCO thật, 7 ảnh SD-v1.5, 7 ảnh FLUX.1-dev, 5 ảnh nén JPEG ×2 và 3 ảnh mẫu.
          </p>
        </div>

        <button
          type="button"
          onClick={runFullBenchmark}
          disabled={isRunning}
          className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2.5 text-xs font-semibold text-slate-950 transition-colors hover:bg-cyan-400 disabled:opacity-60 whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Play className="h-3.5 w-3.5" aria-hidden="true" />
          <span>
            {isRunning
              ? `Đang chạy (${progress}/${DATASET_SPECIMENS.length})...`
              : rows.length > 0
              ? 'Chạy lại Benchmark (29 ảnh)'
              : 'Chạy Benchmark Toàn Bộ 29 Ảnh'}
          </span>
        </button>
      </div>

      {errorMsg && (
        <div
          role="alert"
          className="rounded-lg border border-rose-500/40 bg-rose-950/30 px-4 py-3 text-xs text-rose-200"
        >
          {errorMsg}
        </div>
      )}

      {rows.length === 0 ? (
        <div className="rounded-lg border border-slate-800 bg-[#0D1322] p-10 text-center">
          <h2 className="text-sm font-semibold text-slate-200">
            Chưa có dữ liệu benchmark trong phiên này
          </h2>
          <p className="mx-auto mt-1.5 max-w-md text-xs text-slate-400">
            Bấm nút &ldquo;Chạy Benchmark Toàn Bộ 29 Ảnh&rdquo; ở góc trên bên phải để đo trực tiếp Accuracy, Recall, Specificity, F1-Score và AUC-ROC giữa Baseline v1 và Enhanced v2.
          </p>
        </div>
      ) : (
        <>
          {/* 7-Metric Comparison Table + ROC Curve + Confusion Matrix */}
          {baseMetrics && enhMetrics && (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              {/* Metrics Table */}
              <div className="rounded-lg border border-slate-800 bg-[#0D1322] p-5 lg:col-span-2">
                <div className="mb-3 flex items-baseline justify-between">
                  <h2 className="text-sm font-semibold text-slate-100">
                    Bảng So Sánh 6 Chỉ Số (Threshold = {(threshold * 100).toFixed(0)}%)
                  </h2>
                  <span className="font-mono text-xs text-slate-400 tabular-nums">
                    Độ trễ TB: {avgLatency.toFixed(1)} ms/ảnh
                  </span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400">
                        <th className="py-2.5 pr-3 font-medium">Mô hình</th>
                        <th className="py-2.5 px-2 text-right font-medium">Accuracy</th>
                        <th className="py-2.5 px-2 text-right font-medium">Precision</th>
                        <th className="py-2.5 px-2 text-right font-medium">Recall</th>
                        <th className="py-2.5 px-2 text-right font-medium">Specificity</th>
                        <th className="py-2.5 px-2 text-right font-medium">F1-Score</th>
                        <th className="py-2.5 pl-2 text-right font-medium">AUC-ROC</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/70 font-mono tabular-nums">
                      <tr>
                        <td className="py-3 pr-3 font-sans font-medium text-slate-300">
                          Baseline (engine.py v1)
                        </td>
                        <td className="px-2 py-3 text-right text-slate-300">
                          {baseMetrics.accuracy.toFixed(1)}%
                        </td>
                        <td className="px-2 py-3 text-right text-slate-300">
                          {baseMetrics.precision.toFixed(1)}%
                        </td>
                        <td className="px-2 py-3 text-right text-slate-300">
                          {baseMetrics.recall.toFixed(1)}%
                        </td>
                        <td className="px-2 py-3 text-right text-slate-300">
                          {baseMetrics.specificity.toFixed(1)}%
                        </td>
                        <td className="px-2 py-3 text-right text-slate-300">
                          {baseMetrics.f1.toFixed(4)}
                        </td>
                        <td className="py-3 pl-2 text-right text-slate-300">
                          {baseMetrics.aucRoc.toFixed(4)}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-3 pr-3 font-sans font-semibold text-cyan-400">
                          Enhanced v2 (Multi-Domain CAGU)
                        </td>
                        <td className="px-2 py-3 text-right font-semibold text-emerald-400">
                          {enhMetrics.accuracy.toFixed(1)}%
                        </td>
                        <td className="px-2 py-3 text-right font-semibold text-slate-100">
                          {enhMetrics.precision.toFixed(1)}%
                        </td>
                        <td className="px-2 py-3 text-right font-semibold text-emerald-400">
                          {enhMetrics.recall.toFixed(1)}%
                        </td>
                        <td className="px-2 py-3 text-right font-semibold text-slate-100">
                          {enhMetrics.specificity.toFixed(1)}%
                        </td>
                        <td className="px-2 py-3 text-right font-semibold text-cyan-400">
                          {enhMetrics.f1.toFixed(4)}
                        </td>
                        <td className="py-3 pl-2 text-right font-semibold text-cyan-400">
                          {enhMetrics.aucRoc.toFixed(4)}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-3 pr-3 font-sans text-slate-400">
                          Chênh lệch (Delta)
                        </td>
                        <td className="px-2 py-3 text-right text-cyan-300">
                          {enhMetrics.accuracy - baseMetrics.accuracy >= 0 ? '+' : ''}
                          {(enhMetrics.accuracy - baseMetrics.accuracy).toFixed(1)}%
                        </td>
                        <td className="px-2 py-3 text-right text-slate-400">
                          {enhMetrics.precision - baseMetrics.precision >= 0 ? '+' : ''}
                          {(enhMetrics.precision - baseMetrics.precision).toFixed(1)}%
                        </td>
                        <td className="px-2 py-3 text-right text-cyan-300">
                          {enhMetrics.recall - baseMetrics.recall >= 0 ? '+' : ''}
                          {(enhMetrics.recall - baseMetrics.recall).toFixed(1)}%
                        </td>
                        <td className="px-2 py-3 text-right text-cyan-300">
                          {enhMetrics.specificity - baseMetrics.specificity >= 0 ? '+' : ''}
                          {(enhMetrics.specificity - baseMetrics.specificity).toFixed(1)}%
                        </td>
                        <td className="px-2 py-3 text-right text-cyan-300">
                          {enhMetrics.f1 - baseMetrics.f1 >= 0 ? '+' : ''}
                          {(enhMetrics.f1 - baseMetrics.f1).toFixed(4)}
                        </td>
                        <td className="py-3 pl-2 text-right text-cyan-300">
                          {enhMetrics.aucRoc - baseMetrics.aucRoc >= 0 ? '+' : ''}
                          {(enhMetrics.aucRoc - baseMetrics.aucRoc).toFixed(4)}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                {/* Confusion Matrix comparison */}
                <div className="mt-5 grid grid-cols-2 gap-4 border-t border-slate-800 pt-4 text-xs">
                  <div>
                    <div className="mb-2 font-medium text-slate-400">
                      Confusion Matrix — Baseline v1
                    </div>
                    <div className="grid grid-cols-2 gap-2 font-mono tabular-nums">
                      <div className="rounded border border-slate-800 bg-[#070B14] px-3 py-2">
                        <div className="text-[10px] text-slate-500">TN (Thật đúng)</div>
                        <div className="text-base font-semibold text-emerald-400">
                          {baseMetrics.tn}
                        </div>
                      </div>
                      <div className="rounded border border-slate-800 bg-[#070B14] px-3 py-2">
                        <div className="text-[10px] text-slate-500">FP (Báo nhầm AI)</div>
                        <div className="text-base font-semibold text-amber-400">
                          {baseMetrics.fp}
                        </div>
                      </div>
                      <div className="rounded border border-slate-800 bg-[#070B14] px-3 py-2">
                        <div className="text-[10px] text-slate-500">FN (Bỏ lọt AI)</div>
                        <div className="text-base font-semibold text-rose-400">
                          {baseMetrics.fn}
                        </div>
                      </div>
                      <div className="rounded border border-slate-800 bg-[#070B14] px-3 py-2">
                        <div className="text-[10px] text-slate-500">TP (Bắt đúng AI)</div>
                        <div className="text-base font-semibold text-cyan-400">
                          {baseMetrics.tp}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="mb-2 font-medium text-cyan-400">
                      Confusion Matrix — Enhanced v2
                    </div>
                    <div className="grid grid-cols-2 gap-2 font-mono tabular-nums">
                      <div className="rounded border border-slate-800 bg-[#070B14] px-3 py-2">
                        <div className="text-[10px] text-slate-500">TN (Thật đúng)</div>
                        <div className="text-base font-semibold text-emerald-400">
                          {enhMetrics.tn}
                        </div>
                      </div>
                      <div className="rounded border border-slate-800 bg-[#070B14] px-3 py-2">
                        <div className="text-[10px] text-slate-500">FP (Báo nhầm AI)</div>
                        <div className="text-base font-semibold text-amber-400">
                          {enhMetrics.fp}
                        </div>
                      </div>
                      <div className="rounded border border-slate-800 bg-[#070B14] px-3 py-2">
                        <div className="text-[10px] text-slate-500">FN (Bỏ lọt AI)</div>
                        <div className="text-base font-semibold text-rose-400">
                          {enhMetrics.fn}
                        </div>
                      </div>
                      <div className="rounded border border-slate-800 bg-[#070B14] px-3 py-2">
                        <div className="text-[10px] text-slate-500">TP (Bắt đúng AI)</div>
                        <div className="text-base font-semibold text-cyan-400">
                          {enhMetrics.tp}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* SVG ROC Curve */}
              <div className="rounded-lg border border-slate-800 bg-[#0D1322] p-5">
                <h2 className="text-sm font-semibold text-slate-100">
                  Đường Cong ROC (TPR vs FPR)
                </h2>
                <div className="mt-1 flex items-center gap-4 text-[11px] font-mono">
                  <span className="text-cyan-400">
                    — Enhanced ({enhMetrics.aucRoc.toFixed(3)})
                  </span>
                  <span className="text-amber-400">
                    -- Baseline ({baseMetrics.aucRoc.toFixed(3)})
                  </span>
                </div>

                <svg
                  viewBox="0 0 292 292"
                  className="mt-3 w-full max-w-[320px]"
                  role="img"
                  aria-label="Biểu đồ đường cong ROC so sánh Baseline và Enhanced"
                >
                  {/* Grid */}
                  {[0, 0.25, 0.5, 0.75, 1].map((t) => {
                    const x = 32 + t * 240;
                    const y = 264 - t * 240;
                    return (
                      <g key={t}>
                        <line
                          x1={32}
                          y1={y}
                          x2={272}
                          y2={y}
                          stroke="#1E293B"
                          strokeWidth="1"
                        />
                        <line
                          x1={x}
                          y1={24}
                          x2={x}
                          y2={264}
                          stroke="#1E293B"
                          strokeWidth="1"
                        />
                      </g>
                    );
                  })}
                  {/* Diagonal random */}
                  <line
                    x1={32}
                    y1={264}
                    x2={272}
                    y2={24}
                    stroke="#475569"
                    strokeDasharray="4 4"
                    strokeWidth="1"
                  />
                  {/* Baseline ROC */}
                  {rocBase && (
                    <polyline
                      fill="none"
                      stroke="#F59E0B"
                      strokeWidth="2"
                      strokeDasharray="5 3"
                      points={toPolyline(rocBase.points)}
                    />
                  )}
                  {/* Enhanced ROC */}
                  {rocEnh && (
                    <polyline
                      fill="none"
                      stroke="#06B6D4"
                      strokeWidth="2.5"
                      points={toPolyline(rocEnh.points)}
                    />
                  )}
                  <text x="152" y="284" textAnchor="middle" fill="#94A3B8" fontSize="10">
                    False Positive Rate (FPR)
                  </text>
                  <text
                    x="12"
                    y="144"
                    textAnchor="middle"
                    fill="#94A3B8"
                    fontSize="10"
                    transform="rotate(-90 12 144)"
                  >
                    True Positive Rate (Recall)
                  </text>
                </svg>
              </div>
            </div>
          )}

          {/* Per-image filterable diagnostic table */}
          <div className="rounded-lg border border-slate-800 bg-[#0D1322] p-5">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-sm font-semibold text-slate-100">
                Chi tiết từng tiêu bản ({visibleRows.length} ảnh)
              </h2>

              <div className="flex flex-wrap gap-1.5">
                {(['all', 'real', 'ai_sd15', 'ai_flux', 'degraded', 'samples'] as const).map(
                  (grp) => (
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
                      {grp === 'all' ? 'Tất cả' : grp}
                    </button>
                  )
                )}
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2 pr-3 font-medium">Tệp ảnh</th>
                    <th className="py-2 px-2 font-medium">Nhóm</th>
                    <th className="py-2 px-2 font-medium">Nhãn gốc</th>
                    <th className="py-2 px-2 text-right font-medium">Base Prob</th>
                    <th className="py-2 px-2 text-right font-medium">Enh v2 Prob</th>
                    <th className="py-2 px-2 text-right font-medium">Chroma/Luma</th>
                    <th className="py-2 px-2 text-right font-medium">FFT hi/mid</th>
                    <th className="py-2 px-2 text-right font-medium">JPEG Blk</th>
                    <th className="py-2 pl-2 text-right font-medium">Kết quả</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono tabular-nums">
                  {visibleRows.map(({ specimen, result }) => {
                    const pred = result.enhancedProbAi >= threshold ? 1 : 0;
                    const correct = pred === specimen.label;
                    return (
                      <tr
                        key={specimen.id}
                        className="transition-colors hover:bg-slate-800/40"
                      >
                        <td className="py-2.5 pr-3">
                          <button
                            type="button"
                            onClick={() => onInspectRow(specimen)}
                            className="text-left text-cyan-300 hover:underline cursor-pointer"
                          >
                            {specimen.filename}
                          </button>
                        </td>
                        <td className="px-2 py-2.5 font-sans text-slate-400">
                          {specimen.group}
                        </td>
                        <td
                          className={`px-2 py-2.5 ${
                            specimen.label === 1 ? 'text-rose-400' : 'text-emerald-400'
                          }`}
                        >
                          {specimen.label === 1 ? 'AI' : 'REAL'}
                        </td>
                        <td className="px-2 py-2.5 text-right text-slate-300">
                          {(result.baselineProbAi * 100).toFixed(1)}%
                        </td>
                        <td className="px-2 py-2.5 text-right font-semibold text-slate-100">
                          {(result.enhancedProbAi * 100).toFixed(1)}%
                        </td>
                        <td className="px-2 py-2.5 text-right text-slate-300">
                          {result.features.chromaToLuma.toFixed(3)}
                        </td>
                        <td className="px-2 py-2.5 text-right text-slate-300">
                          {result.features.nativeHighMidRatio.toFixed(2)}
                        </td>
                        <td className="px-2 py-2.5 text-right text-slate-300">
                          {result.features.jpegBlockiness.toFixed(2)}
                        </td>
                        <td
                          className={`py-2.5 pl-2 text-right font-sans font-semibold ${
                            correct ? 'text-emerald-400' : 'text-rose-400'
                          }`}
                        >
                          {correct ? '● Đúng' : '✖ Sai'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
