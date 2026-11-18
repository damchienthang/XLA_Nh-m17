import { computeFft2dMagnitude, computeLegacyEngineFft, LegacyFftMetrics } from './fft';

export type ForensicLevel = 'critical' | 'warning' | 'info' | 'ok';

export interface ForensicLogEntry {
  level: ForensicLevel;
  title: string;
  detail: string;
  value: string;
}

export interface CaguWeights {
  spatialPct: number;
  freqPct: number;
  chromaPct: number;
  compressionMode: 'uncompressed' | 'moderate_jpeg' | 'heavy_jpeg';
}

export interface ForensicFeatures {
  width: number;
  height: number;
  jpegBlockiness: number;
  saturationMean: number;
  // Legacy engine.py (256x256 bilinear FFT)
  legacyFft: LegacyFftMetrics;
  spatialResidualMse: number;
  // Enhanced native-resolution multi-domain features
  nativeMidLowRatio: number;
  nativeHighMidRatio: number;
  nativePeakMedRatio: number;
  chromaToLuma: number;
  corrRG: number;
  corrGB: number;
  smoothNoise: number;
  lapKurtosis: number;
  patchAnomalyCoveragePct: number;
}

export interface ForensicResult {
  width: number;
  height: number;
  latencyMs: number;
  features: ForensicFeatures;
  // Baseline CO-SPY (engine.py v1 heuristic)
  baselineProbAi: number;
  baselineReasoning: string;
  // Enhanced v2 (Multi-Domain CAGU)
  enhancedProbAi: number;
  enhancedReasoning: string;
  primaryLead: string;
  cagu: CaguWeights;
  subScores: {
    spatialScore: number;
    freqScore: number;
    chromaScore: number;
    consistencyFactor: number;
  };
  // Visualization buffers
  visWidth: number;
  visHeight: number;
  residualMap01: Float32Array;
  gradcamMap01: Float32Array;
  nativeFftSize: number;
  nativeFftLogMag: Uint8Array;
  legacyFftLogMag: Uint8Array;
}

function sigmoid(z: number): number {
  if (z > 20) return 1;
  if (z < -20) return 0;
  return 1 / (1 + Math.exp(-z));
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

/**
 * Bilinear resize of RGBA buffer into a 256x256 grayscale Float64Array (0..255),
 * matching PIL Image.convert("L").resize((256, 256)) in engine.py.
 */
export function resizeToGray256(rgba: Uint8Array | Uint8ClampedArray, width: number, height: number): Float64Array {
  const N = 256;
  const out = new Float64Array(N * N);
  const scaleX = width / N;
  const scaleY = height / N;

  if (scaleX > 1.2 || scaleY > 1.2) {
    // Area / box-averaged downsampling (matches PIL Image.resize anti-aliasing)
    for (let y = 0; y < N; y++) {
      const y0 = Math.floor(y * scaleY);
      const y1 = Math.min(height, Math.max(y0 + 1, Math.ceil((y + 1) * scaleY)));
      for (let x = 0; x < N; x++) {
        const x0 = Math.floor(x * scaleX);
        const x1 = Math.min(width, Math.max(x0 + 1, Math.ceil((x + 1) * scaleX)));
        let sum = 0;
        let cnt = 0;
        for (let sy = y0; sy < y1; sy++) {
          for (let sx = x0; sx < x1; sx++) {
            const p = (sy * width + sx) * 4;
            sum += 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
            cnt++;
          }
        }
        out[y * N + x] = sum / Math.max(cnt, 1);
      }
    }
    return out;
  }

  for (let y = 0; y < N; y++) {
    const sy = Math.min(height - 1, Math.max(0, (y + 0.5) * scaleY - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(height - 1, y0 + 1);
    const fy = sy - y0;

    for (let x = 0; x < N; x++) {
      const sx = Math.min(width - 1, Math.max(0, (x + 0.5) * scaleX - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(width - 1, x0 + 1);
      const fx = sx - x0;

      const i00 = (y0 * width + x0) * 4;
      const i10 = (y0 * width + x1) * 4;
      const i01 = (y1 * width + x0) * 4;
      const i11 = (y1 * width + x1) * 4;

      const g00 = 0.299 * rgba[i00] + 0.587 * rgba[i00 + 1] + 0.114 * rgba[i00 + 2];
      const g10 = 0.299 * rgba[i10] + 0.587 * rgba[i10 + 1] + 0.114 * rgba[i10 + 2];
      const g01 = 0.299 * rgba[i01] + 0.587 * rgba[i01 + 1] + 0.114 * rgba[i01 + 2];
      const g11 = 0.299 * rgba[i11] + 0.587 * rgba[i11 + 1] + 0.114 * rgba[i11 + 2];

      const top = g00 + (g10 - g00) * fx;
      const bot = g01 + (g11 - g01) * fx;
      out[y * N + x] = top + (bot - top) * fy;
    }
  }
  return out;
}

/**
 * Pure TypeScript multi-domain pixel forensic analyzer.
 * Runs both Baseline (engine.py v1) and Enhanced v2 (Native 2D-FFT + Bayer Chroma + Quality-Aware CAGU).
 */
export function analyzeImagePixels(
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number
): ForensicResult {
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const totalPixels = width * height;

  // 1. Legacy 256x256 bilinear FFT (exact engine.py v1 logic)
  const gray256 = resizeToGray256(rgba, width, height);
  const legacyFft = computeLegacyEngineFft(gray256);

  // 2. Native-resolution grayscale, saturation, JPEG 8x8 blockiness, RGB residual stats
  const gray = new Float32Array(totalPixels);
  let satSum = 0;
  for (let i = 0; i < totalPixels; i++) {
    const p = i * 4;
    const r = rgba[p];
    const g = rgba[p + 1];
    const b = rgba[p + 2];
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    const maxC = r > g ? (r > b ? r : b) : g > b ? g : b;
    const minC = r < g ? (r < b ? r : b) : g < b ? g : b;
    satSum += (maxC - minC) / 255;
  }
  const saturationMean = satSum / Math.max(totalPixels, 1);

  // JPEG 8x8 blockiness
  let bndGrad = 0;
  let bndCount = 0;
  let intGrad = 0;
  let intCount = 0;

  // Cross-channel RGB & Luminance/Chrominance Laplacian residuals
  let lumaResSq = 0;
  let cbResSq = 0;
  let crResSq = 0;
  let varR = 0;
  let varG = 0;
  let varB = 0;
  let covRG = 0;
  let covGB = 0;
  let lapSq = 0;
  let lapFourth = 0;
  let lapCount = 0;
  let smoothSq = 0;
  let smoothCount = 0;

  // Spatial VAE-proxy residual MSE (3x3 Gaussian blur reconstruction diff on [0,1])
  let blurReconMse = 0;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const dx = Math.abs(gray[idx] - gray[idx - 1]);
      const dy = Math.abs(gray[idx] - gray[idx - width]);
      if (x % 8 === 0) {
        bndGrad += dx;
        bndCount++;
      } else {
        intGrad += dx;
        intCount++;
      }
      if (y % 8 === 0) {
        bndGrad += dy;
        bndCount++;
      } else {
        intGrad += dy;
        intCount++;
      }

      const p = idx * 4;
      const pL = (idx - 1) * 4;
      const pR = (idx + 1) * 4;
      const pU = (idx - width) * 4;
      const pD = (idx + width) * 4;

      const lr = 4 * rgba[p] - rgba[pL] - rgba[pR] - rgba[pU] - rgba[pD];
      const lg = 4 * rgba[p + 1] - rgba[pL + 1] - rgba[pR + 1] - rgba[pU + 1] - rgba[pD + 1];
      const lb = 4 * rgba[p + 2] - rgba[pL + 2] - rgba[pR + 2] - rgba[pU + 2] - rgba[pD + 2];

      const luma = 0.299 * lr + 0.587 * lg + 0.114 * lb;
      const cb = -0.169 * lr - 0.331 * lg + 0.5 * lb;
      const cr = 0.5 * lr - 0.419 * lg - 0.081 * lb;

      lumaResSq += luma * luma;
      cbResSq += cb * cb;
      crResSq += cr * cr;

      varR += lr * lr;
      varG += lg * lg;
      varB += lb * lb;
      covRG += lr * lg;
      covGB += lg * lb;

      lapSq += luma * luma;
      lapFourth += luma * luma * luma * luma;
      lapCount++;

      // Normalized [0,1] blur reconstruction diff
      const blurG = (gray[idx - 1] + gray[idx + 1] + gray[idx - width] + gray[idx + width] + 4 * gray[idx]) / 8;
      const d01 = (gray[idx] - blurG) / 255;
      blurReconMse += d01 * d01;

      const gx = Math.abs(rgba[pR + 1] - rgba[pL + 1]);
      const gy = Math.abs(rgba[pD + 1] - rgba[pU + 1]);
      if (gx + gy < 12) {
        smoothSq += luma * luma;
        smoothCount++;
      }
    }
  }

  const jpegBlockiness = (bndGrad / Math.max(bndCount, 1)) / (intGrad / Math.max(intCount, 1) + 1e-6);
  const chromaToLuma = Math.sqrt((cbResSq + crResSq) / (2 * lumaResSq + 1e-9));
  const corrRG = covRG / Math.sqrt(varR * varG + 1e-9);
  const corrGB = covGB / Math.sqrt(varG * varB + 1e-9);
  const lapMeanSq = lapSq / Math.max(lapCount, 1);
  const lapKurtosis = lapFourth / Math.max(lapCount, 1) / (lapMeanSq * lapMeanSq + 1e-9);
  const smoothNoise = smoothCount > 0 ? Math.sqrt(smoothSq / smoothCount) : 0;
  const spatialResidualMse = blurReconMse / Math.max(lapCount, 1);

  // 3. Native-crop denoised residual 2D-FFT (no bilinear downsampling!)
  const maxPow2 = Math.pow(2, Math.floor(Math.log2(Math.min(width, height))));
  const nativeFftSize = Math.max(64, Math.min(128, maxPow2));
  const ox = Math.floor((width - nativeFftSize) / 2);
  const oy = Math.floor((height - nativeFftSize) / 2);
  const patchRes = new Float64Array(nativeFftSize * nativeFftSize);

  for (let py = 0; py < nativeFftSize; py++) {
    const gy = oy + py;
    for (let px = 0; px < nativeFftSize; px++) {
      const gx = ox + px;
      if (gx > 0 && gx < width - 1 && gy > 0 && gy < height - 1) {
        const c = gray[gy * width + gx];
        const avg4 =
          0.25 *
          (gray[gy * width + gx - 1] +
            gray[gy * width + gx + 1] +
            gray[(gy - 1) * width + gx] +
            gray[(gy + 1) * width + gx]);
        patchRes[py * nativeFftSize + px] = c - avg4;
      }
    }
  }

  const nativeMag = computeFft2dMagnitude(patchRes, nativeFftSize);
  const half = nativeFftSize >> 1;
  let eLow = 0;
  let cLow = 0;
  let eMid = 0;
  let cMid = 0;
  let eHigh = 0;
  let cHigh = 0;
  const highVals: number[] = [];
  const nativeLog = new Float64Array(nativeFftSize * nativeFftSize);
  let nMinLog = Infinity;
  let nMaxLog = -Infinity;

  for (let y = 0; y < nativeFftSize; y++) {
    for (let x = 0; x < nativeFftSize; x++) {
      const idx = y * nativeFftSize + x;
      const v = nativeMag[idx];
      const lm = Math.log(v + 1e-6);
      nativeLog[idx] = lm;
      if (lm < nMinLog) nMinLog = lm;
      if (lm > nMaxLog) nMaxLog = lm;

      const r = Math.hypot(x - half, y - half);
      const rNorm = r / half; // 0..1.41
      if (rNorm >= 0.06 && rNorm < 0.25) {
        eLow += v;
        cLow++;
      } else if (rNorm >= 0.25 && rNorm < 0.625) {
        eMid += v;
        cMid++;
      } else if (rNorm >= 0.625 && rNorm <= 1.0) {
        eHigh += v;
        cHigh++;
        highVals.push(v);
      }
    }
  }

  highVals.sort((a, b) => a - b);
  const highMed = highVals[Math.floor(highVals.length / 2)] || 1;
  const highP99 = highVals[Math.floor(highVals.length * 0.99)] || 1;
  const nativePeakMedRatio = highP99 / (highMed + 1e-6);
  const nativeMidLowRatio = (eMid / Math.max(cMid, 1)) / (eLow / Math.max(cLow, 1) + 1e-6);
  const nativeHighMidRatio = (eHigh / Math.max(cHigh, 1)) / (eMid / Math.max(cMid, 1) + 1e-6);

  const nativeFftLogMag = new Uint8Array(nativeFftSize * nativeFftSize);
  const nSpan = nMaxLog - nMinLog + 1e-6;
  for (let i = 0; i < nativeFftSize * nativeFftSize; i++) {
    nativeFftLogMag[i] = Math.max(0, Math.min(255, Math.round(((nativeLog[i] - nMinLog) / nSpan) * 255)));
  }

  // 4. Visualization buffers (256x256 downsampled residual & patch-level spatial Grad-CAM proxy)
  const visWidth = 256;
  const visHeight = 256;
  const residualRaw = new Float32Array(visWidth * visHeight);
  for (let vy = 0; vy < visHeight; vy++) {
    const sy = Math.min(height - 2, Math.max(1, Math.floor((vy / visHeight) * height)));
    for (let vx = 0; vx < visWidth; vx++) {
      const sx = Math.min(width - 2, Math.max(1, Math.floor((vx / visWidth) * width)));
      const idx = sy * width + sx;
      const avg4 = 0.25 * (gray[idx - 1] + gray[idx + 1] + gray[idx - width] + gray[idx + width]);
      residualRaw[vy * visWidth + vx] = Math.abs(gray[idx] - avg4) / 255;
    }
  }

  const sortedRes = Float32Array.from(residualRaw).sort();
  const p995 = Math.max(sortedRes[Math.floor(sortedRes.length * 0.995)] || 0.05, 1e-4);
  const residualMap01 = new Float32Array(visWidth * visHeight);
  for (let i = 0; i < residualMap01.length; i++) {
    residualMap01[i] = clamp01(residualRaw[i] / p995);
  }

  // Compute 16x16 grid local anomaly map (low chroma variance + tight channel correlation + smooth residual)
  const gridN = 16;
  const gridScore = new Float32Array(gridN * gridN);
  let anomalyPatches = 0;
  for (let gy = 0; gy < gridN; gy++) {
    const y0 = Math.max(1, Math.floor((gy / gridN) * height));
    const y1 = Math.min(height - 1, Math.floor(((gy + 1) / gridN) * height));
    for (let gx = 0; gx < gridN; gx++) {
      const x0 = Math.max(1, Math.floor((gx / gridN) * width));
      const x1 = Math.min(width - 1, Math.floor(((gx + 1) / gridN) * width));
      let pLumaSq = 0;
      let pChromaSq = 0;
      let pCnt = 0;
      for (let y = y0; y < y1; y += 2) {
        for (let x = x0; x < x1; x += 2) {
          const p = (y * width + x) * 4;
          const pL = (y * width + x - 1) * 4;
          const pU = ((y - 1) * width + x) * 4;
          const dr = 2 * rgba[p] - rgba[pL] - rgba[pU];
          const dg = 2 * rgba[p + 1] - rgba[pL + 1] - rgba[pU + 1];
          const db = 2 * rgba[p + 2] - rgba[pL + 2] - rgba[pU + 2];
          const l = 0.299 * dr + 0.587 * dg + 0.114 * db;
          const cb = -0.169 * dr - 0.331 * dg + 0.5 * db;
          const cr = 0.5 * dr - 0.419 * dg - 0.081 * db;
          pLumaSq += l * l;
          pChromaSq += 0.5 * (cb * cb + cr * cr);
          pCnt++;
        }
      }
      const localChrLum = Math.sqrt(pChromaSq / (pLumaSq + 1e-6));
      const localNoise = Math.sqrt(pLumaSq / Math.max(pCnt, 1));
      // Patch is suspicious when chroma residual is suppressed relative to luma or noise is overly uniform
      const pAnom = clamp01(
        0.65 * sigmoid((0.16 - localChrLum) * 18) + 0.35 * sigmoid((10 - localNoise) * 0.25)
      );
      gridScore[gy * gridN + gx] = pAnom;
      if (pAnom > 0.6) anomalyPatches++;
    }
  }
  const patchAnomalyCoveragePct = (anomalyPatches / (gridN * gridN)) * 100;

  // Bilinear upsample 16x16 gridScore -> 256x256 gradcamMap01
  const gradcamMap01 = new Float32Array(visWidth * visHeight);
  for (let vy = 0; vy < visHeight; vy++) {
    const sy = (vy / visHeight) * (gridN - 1);
    const y0 = Math.floor(sy);
    const y1 = Math.min(gridN - 1, y0 + 1);
    const fy = sy - y0;
    for (let vx = 0; vx < visWidth; vx++) {
      const sx = (vx / visWidth) * (gridN - 1);
      const x0 = Math.floor(sx);
      const x1 = Math.min(gridN - 1, x0 + 1);
      const fx = sx - x0;
      const v00 = gridScore[y0 * gridN + x0];
      const v10 = gridScore[y0 * gridN + x1];
      const v01 = gridScore[y1 * gridN + x0];
      const v11 = gridScore[y1 * gridN + x1];
      gradcamMap01[vy * visWidth + vx] =
        v00 + (v10 - v00) * fx + (v01 - v00) * fy + (v00 - v10 - v01 + v11) * fx * fy;
    }
  }

  // 5. Baseline Model (Exact engine.py v1 heuristic logic)
  let baselineProbAi: number;
  let baselineReasoning: string;
  const baselineSpatialProb = sigmoid((0.0011 - spatialResidualMse) * 1200) * 0.55;
  if (legacyFft.isAnomalyFlagged) {
    const boost = Math.min(Math.max((legacyFft.fRatio - 0.2) * 3.5, 0.4), 0.75);
    baselineProbAi = Math.min(Math.max(baselineSpatialProb + boost + 0.25, 0.8), 0.95);
    baselineReasoning = `engine.py v1 kích hoạt FFT (f_ratio=${legacyFft.fRatio.toFixed(3)} > 0.23, spikes=${legacyFft.fSpikes}).`;
  } else if (spatialResidualMse < 0.00045) {
    baselineProbAi = Math.max(0.58, baselineSpatialProb + 0.22);
    baselineReasoning = `engine.py v1 dựa vào sai phân không gian thấp (MSE=${spatialResidualMse.toFixed(6)}); nhánh FFT không kích hoạt (f_ratio=${legacyFft.fRatio.toFixed(3)}).`;
  } else {
    baselineProbAi = clamp01(0.12 + baselineSpatialProb * 0.45);
    baselineReasoning =
      jpegBlockiness > 1.18
        ? `CẢNH BÁO BỊ LỪA: Nén JPEG làm f_ratio tụt xuống ${legacyFft.fRatio.toFixed(3)} (< 0.23) và mờ vi nhiễu — engine.py v1 không kích hoạt được nhánh tần số.`
        : `engine.py v1: f_ratio=${legacyFft.fRatio.toFixed(3)} (< 0.23) nên nhánh FFT không kích hoạt; suy luận chỉ dựa vào không gian.`;
  }

  // 6. Enhanced v2 Model (Quality-Aware CAGU Multi-Domain Fusion)
  const compressionMode: CaguWeights['compressionMode'] =
    jpegBlockiness > 1.22 ? 'heavy_jpeg' : jpegBlockiness > 1.08 ? 'moderate_jpeg' : 'uncompressed';
  const isMonoOrLowSat = saturationMean < 0.09;

  // Branch A: Native Spectral Branch (high/mid roll-off + periodic peak-to-median)
  // On uncompressed images, real photos maintain hi/mid ~1.18-1.38 whereas SD/diffusion drops ~0.63-1.01
  const spectralRolloffAnom = sigmoid((1.09 - nativeHighMidRatio) * 6.0);
  const spectralPeakAnom = sigmoid((nativePeakMedRatio - 4.35) * 1.1);
  const freqScore =
    compressionMode === 'heavy_jpeg'
      ? clamp01(0.65 * spectralPeakAnom + 0.35 * sigmoid((1.55 - nativeMidLowRatio) * 2.5))
      : clamp01(0.65 * spectralRolloffAnom + 0.35 * spectralPeakAnom);

  // Branch B: Spatial / Sensor Noise & Kurtosis Branch
  // FLUX has ultra-low smooth noise (< 9.5) + high kurtosis (> 40)
  // SD1.5 / Degraded AI has depressed kurtosis (< 11) with elevated mid-band noise
  const fluxSmoothSignal = sigmoid((9.8 - smoothNoise) * 0.65) * sigmoid((lapKurtosis - 22) * 0.14);
  const depressedKurtSignal = sigmoid((10.5 - lapKurtosis) * 0.65) * sigmoid((smoothNoise - 10.5) * 0.45);
  const spatialScore = clamp01(Math.max(fluxSmoothSignal, depressedKurtSignal * 0.92));

  // Branch C: Chrominance & Bayer Channel Coherence Branch
  // Real camera Bayer demosaicing yields chroma/luma >= 0.145 & chromatic independence in at least one pair (min corr < 0.925)
  // AI latent decoding yields suppressed chroma residual (0.035..0.157) & all RGB pairs tightly locked (>= 0.948)
  const meanChannelCorr = 0.5 * (corrRG + corrGB);
  const minChannelCorr = Math.min(corrRG, corrGB);
  const rawChromaRatioAnom = sigmoid((0.165 - chromaToLuma) * 26);
  const rawChannelLockAnom = sigmoid((0.5 * (meanChannelCorr + minChannelCorr) - 0.94) * 42);
  const chromaScore = isMonoOrLowSat
    ? 0.45 // Neutralize chrominance cue on grayscale / low-saturation real photos
    : clamp01(0.60 * rawChromaRatioAnom + 0.40 * rawChannelLockAnom);

  // Quality-Aware CAGU Dynamic Weights
  let rawW_Spatial = compressionMode === 'heavy_jpeg' ? 0.42 : 0.32;
  let rawW_Freq = compressionMode === 'heavy_jpeg' ? 0.36 : 0.33;
  let rawW_Chroma = isMonoOrLowSat ? 0.06 : compressionMode === 'heavy_jpeg' ? 0.22 : 0.35;

  // If image is high-res (>= 900px, e.g., FLUX 1024x1024 or high-res camera), boost spatial+chroma precision
  if (Math.min(width, height) >= 900 && compressionMode === 'uncompressed') {
    rawW_Spatial = 0.36;
    rawW_Chroma = 0.42;
    rawW_Freq = 0.22;
  }

  const wSum = rawW_Spatial + rawW_Freq + rawW_Chroma;
  const wSpatial = rawW_Spatial / wSum;
  const wFreq = rawW_Freq / wSum;
  const wChroma = rawW_Chroma / wSum;

  const fusedRaw = wSpatial * spatialScore + wFreq * freqScore + wChroma * chromaScore;

  // Dual-Domain Consistency Regularizer:
  // Checks how many active branches agree (> 0.58) vs strongly oppose (< 0.35),
  // plus optical Bayer chromatic independence check (real sensors break RGB lock in >=1 pair)
  const activeSignals = [spatialScore, freqScore, ...(isMonoOrLowSat ? [] : [chromaScore])];
  const highAgree = activeSignals.filter((s) => s > 0.58).length;
  const lowAgree = activeSignals.filter((s) => s < 0.35).length;
  const strongOpticalBayer =
    !isMonoOrLowSat &&
    ((chromaScore < 0.18 && chromaToLuma > 0.20) || (minChannelCorr < 0.92 && chromaToLuma > 0.145));

  let consistencyFactor = 1.0;
  let enhancedProbAi = fusedRaw;
  if (strongOpticalBayer && !(compressionMode === 'heavy_jpeg' && spatialScore > 0.75)) {
    consistencyFactor = 0.72;
    enhancedProbAi = clamp01(fusedRaw * 0.65);
  } else if (highAgree >= 2) {
    consistencyFactor = 1.12;
    enhancedProbAi = clamp01(0.55 + (fusedRaw - 0.42) * 1.38);
  } else if (lowAgree >= 2 && fusedRaw < 0.6) {
    consistencyFactor = 0.85;
    enhancedProbAi = clamp01(fusedRaw * 0.72);
  } else {
    enhancedProbAi = clamp01(sigmoid((fusedRaw - 0.48) * 5.2));
  }

  // Determine primary lead branch
  const contrib = [
    { name: 'Chroma & Bayer Coherence', val: wChroma * chromaScore },
    { name: 'Native 2D-FFT Spectral', val: wFreq * freqScore },
    { name: 'Sensor Noise & Kurtosis', val: wSpatial * spatialScore },
  ].sort((a, b) => b.val - a.val);

  const primaryLead = contrib[0].name;

  let enhancedReasoning = '';
  if (enhancedProbAi >= 0.5) {
    if (compressionMode === 'heavy_jpeg') {
      enhancedReasoning = `CAGU phát hiện nén JPEG mạnh (blockiness=${jpegBlockiness.toFixed(2)}), tự động chuyển ${(wFreq * 100).toFixed(0)}% trọng số sang phổ tần số native + ${(wChroma * 100).toFixed(0)}% sang tương quan kênh màu và xác nhận ảnh AI.`;
    } else if (fluxSmoothSignal > 0.6) {
      enhancedReasoning = `Vùng phẳng siêu mịn (smoothNoise=${smoothNoise.toFixed(1)}, kurtosis=${lapKurtosis.toFixed(1)}) kết hợp khóa kênh RGB (${(0.5 * (corrRG + corrGB)).toFixed(3)}) — dấu hiệu đặc trưng của Latent Diffusion / FLUX.`;
    } else {
      enhancedReasoning = `Đồng thuận đa miền: phổ 2D-FFT native suy giảm bất thường (hi/mid=${nativeHighMidRatio.toFixed(2)}) và dư màu Chroma/Luma thấp (${chromaToLuma.toFixed(3)}).`;
    }
  } else {
    if (isMonoOrLowSat) {
      enhancedReasoning = `Ảnh có độ bão hòa thấp (${saturationMean.toFixed(2)}); CAGU tự động hạ trọng số kênh màu xuống ${(wChroma * 100).toFixed(1)}% để tránh báo nhầm, xác nhận nhiễu & phổ tự nhiên.`;
    } else {
      enhancedReasoning = `Đồng thuận hai miền: tỷ lệ nhiễu sắc độ Chroma/Luma (${chromaToLuma.toFixed(3)}) và phổ năng lượng native (hi/mid=${nativeHighMidRatio.toFixed(2)}) phù hợp cảm biến quang học thực.`;
    }
  }

  const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();

  return {
    width,
    height,
    latencyMs: Math.max(1, t1 - t0),
    features: {
      width,
      height,
      jpegBlockiness,
      saturationMean,
      legacyFft,
      spatialResidualMse,
      nativeMidLowRatio,
      nativeHighMidRatio,
      nativePeakMedRatio,
      chromaToLuma,
      corrRG,
      corrGB,
      smoothNoise,
      lapKurtosis,
      patchAnomalyCoveragePct,
    },
    baselineProbAi,
    baselineReasoning,
    enhancedProbAi,
    enhancedReasoning,
    primaryLead,
    cagu: {
      spatialPct: wSpatial * 100,
      freqPct: wFreq * 100,
      chromaPct: wChroma * 100,
      compressionMode,
    },
    subScores: {
      spatialScore,
      freqScore,
      chromaScore,
      consistencyFactor,
    },
    visWidth,
    visHeight,
    residualMap01,
    gradcamMap01,
    nativeFftSize,
    nativeFftLogMag,
    legacyFftLogMag: legacyFft.logMagNormalized,
  };
}

export function buildForensicLog(result: ForensicResult, threshold = 0.5): ForensicLogEntry[] {
  const { features, cagu, subScores, enhancedProbAi, baselineProbAi } = result;
  const isAi = enhancedProbAi >= threshold;
  const log: ForensicLogEntry[] = [];

  // 1. Legacy vs Enhanced FFT comparison
  log.push({
    level: features.legacyFft.isAnomalyFlagged ? 'info' : 'warning',
    title: `Kiểm tra Baseline (engine.py v1): f_ratio = ${features.legacyFft.fRatio.toFixed(3)} · spikes = ${features.legacyFft.fSpikes}`,
    detail: features.legacyFft.isAnomalyFlagged
      ? `Nhánh FFT cũ kích hoạt ngưỡng (>0.23), cho xác suất Baseline ${(baselineProbAi * 100).toFixed(1)}%.`
      : `Nhánh FFT cũ (resize 256px) KHÔNG kích hoạt (cần f_ratio > 0.23), dù log cũ báo nhầm spikes=${features.legacyFft.fSpikes} > 100.`,
    value: `Base ${(baselineProbAi * 100).toFixed(1)}%`,
  });

  // 2. JPEG Blockiness & CAGU Gating
  log.push({
    level: cagu.compressionMode === 'heavy_jpeg' ? 'warning' : 'ok',
    title: `Van điều tiết CAGU (Nén JPEG blockiness = ${features.jpegBlockiness.toFixed(2)})`,
    detail:
      cagu.compressionMode === 'heavy_jpeg'
        ? `Phát hiện nén JPEG nặng (>1.22): CAGU giảm nhánh pixel còn ${cagu.spatialPct.toFixed(1)}%, tăng Tần số (${cagu.freqPct.toFixed(1)}%) & Kênh màu (${cagu.chromaPct.toFixed(1)}%).`
        : `Ảnh ít/không nén JPEG: CAGU phân bổ ${cagu.spatialPct.toFixed(1)}% Không gian · ${cagu.freqPct.toFixed(1)}% Tần số · ${cagu.chromaPct.toFixed(1)}% Kênh màu.`,
    value: `${cagu.freqPct.toFixed(0)}% Freq`,
  });

  // 3. Native 2D-FFT Spectral Branch
  log.push({
    level: subScores.freqScore >= 0.55 ? 'critical' : 'ok',
    title: `Phổ 2D-FFT Native (${result.nativeFftSize}×${result.nativeFftSize}): hi/mid = ${features.nativeHighMidRatio.toFixed(2)} · peak/med = ${features.nativePeakMedRatio.toFixed(2)}`,
    detail:
      subScores.freqScore >= 0.55
        ? `Phổ dư khử nội dung suy giảm bất thường ở cao tần (hi/mid=${features.nativeHighMidRatio.toFixed(2)}) hoặc xuất hiện đỉnh chu kỳ (peak/med=${features.nativePeakMedRatio.toFixed(2)}).`
        : `Phổ năng lượng 2D-FFT suy giảm tự nhiên theo quy luật 1/f của quang học thực tế.`,
    value: `Score ${(subScores.freqScore * 100).toFixed(0)}%`,
  });

  // 4. Chrominance & Bayer Coherence
  log.push({
    level: features.saturationMean < 0.09 ? 'info' : subScores.chromaScore >= 0.55 ? 'critical' : 'ok',
    title: `Tương quan Bayer RGB: Chroma/Luma = ${features.chromaToLuma.toFixed(3)} · corr(R,G) = ${features.corrRG.toFixed(3)}`,
    detail:
      features.saturationMean < 0.09
        ? `Ảnh đơn sắc / độ bão hòa thấp (${features.saturationMean.toFixed(2)}): nhánh sắc độ được tự động trung hòa để bảo toàn Specificity.`
        : subScores.chromaScore >= 0.55
        ? `Dư sắc độ Chroma/Luma thấp (${features.chromaToLuma.toFixed(3)}) và các kênh RGB khóa chặt (${features.corrRG.toFixed(3)}) — đặc trưng giải mã từ latent space.`
        : `Nhiễu sắc độ độc lập giữa các kênh RGB phù hợp cảm biến Bayer thực tế.`,
    value: `Score ${(subScores.chromaScore * 100).toFixed(0)}%`,
  });

  // 5. Spatial Sensor Noise & Kurtosis
  log.push({
    level: subScores.spatialScore >= 0.55 ? 'critical' : 'ok',
    title: `Nhiễu vùng phẳng & Kurtosis: smoothNoise = ${features.smoothNoise.toFixed(1)} · Kurtosis = ${features.lapKurtosis.toFixed(1)}`,
    detail:
      subScores.spatialScore >= 0.55
        ? `Phân bố sai phân Laplacian bất thường (smoothNoise=${features.smoothNoise.toFixed(1)}, kurtosis=${features.lapKurtosis.toFixed(1)}, vùng nghi vấn ${features.patchAnomalyCoveragePct.toFixed(1)}%).`
        : `Độ nhọn và nhiễu nền vùng phẳng nằm trong biên độ tự nhiên của ống kính & cảm biến.`,
    value: `Score ${(subScores.spatialScore * 100).toFixed(0)}%`,
  });

  // 6. Final Verdict
  log.push({
    level: isAi ? 'critical' : 'ok',
    title: isAi
      ? `KẾT LUẬN ENHANCED v2: ẢNH DO AI TẠO SINH (${(enhancedProbAi * 100).toFixed(1)}%)`
      : `KẾT LUẬN ENHANCED v2: ẢNH THẬT QUANG HỌC (${((1 - enhancedProbAi) * 100).toFixed(1)}% thật)`,
    detail: result.enhancedReasoning,
    value: `${(enhancedProbAi * 100).toFixed(1)}% AI`,
  });

  return log;
}

export interface BinaryMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1: number;
  aucRoc: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface RocPoint {
  fpr: number;
  tpr: number;
  threshold: number;
}

export function computeRocCurve(labels: (0 | 1)[], probs: number[]): { points: RocPoint[]; auc: number } {
  if (labels.length === 0) return { points: [], auc: 0 };
  const totalPos = labels.filter((l) => l === 1).length;
  const totalNeg = labels.length - totalPos;
  if (totalPos === 0 || totalNeg === 0) {
    return {
      points: [
        { fpr: 0, tpr: 0, threshold: 1 },
        { fpr: 1, tpr: 1, threshold: 0 },
      ],
      auc: 0.5,
    };
  }

  const thresholds = Array.from(new Set([1.01, ...probs, -0.01])).sort((a, b) => b - a);
  const points: RocPoint[] = thresholds.map((th) => {
    let tp = 0;
    let fp = 0;
    for (let i = 0; i < labels.length; i++) {
      if (probs[i] >= th) {
        if (labels[i] === 1) tp++;
        else fp++;
      }
    }
    return { fpr: fp / totalNeg, tpr: tp / totalPos, threshold: th };
  });

  let auc = 0;
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].fpr - points[i - 1].fpr;
    const avgY = 0.5 * (points[i].tpr + points[i - 1].tpr);
    auc += dx * avgY;
  }
  return { points, auc };
}

export function computeBinaryMetrics(
  labels: (0 | 1)[],
  probs: number[],
  threshold = 0.5
): BinaryMetrics {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;

  for (let i = 0; i < labels.length; i++) {
    const pred = probs[i] >= threshold ? 1 : 0;
    const actual = labels[i];
    if (actual === 1 && pred === 1) tp++;
    else if (actual === 0 && pred === 1) fp++;
    else if (actual === 0 && pred === 0) tn++;
    else fn++;
  }

  const total = Math.max(labels.length, 1);
  const accuracy = ((tp + tn) / total) * 100;
  const precision = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
  const recall = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
  const specificity = tn + fp > 0 ? (tn / (tn + fp)) * 100 : 0;
  const f1 = precision + recall > 0 ? (2 * (precision / 100) * (recall / 100)) / (precision / 100 + recall / 100) : 0;
  const { auc } = computeRocCurve(labels, probs);

  return {
    accuracy,
    precision,
    recall,
    specificity,
    f1,
    aucRoc: auc,
    tp,
    fp,
    tn,
    fn,
  };
}
