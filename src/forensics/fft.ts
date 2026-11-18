export function fft1d(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wlenRe = Math.cos(ang);
    const wlenIm = Math.sin(ang);
    const halfLen = len >> 1;
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let j = 0; j < halfLen; j++) {
        const idx1 = i + j;
        const idx2 = idx1 + halfLen;
        const uRe = re[idx1];
        const uIm = im[idx1];
        const vRe = re[idx2] * wRe - im[idx2] * wIm;
        const vIm = re[idx2] * wIm + im[idx2] * wRe;
        re[idx1] = uRe + vRe;
        im[idx1] = uIm + vIm;
        re[idx2] = uRe - vRe;
        im[idx2] = uIm - vIm;
        const nextRe = wRe * wlenRe - wIm * wlenIm;
        wIm = wRe * wlenIm + wIm * wlenRe;
        wRe = nextRe;
      }
    }
  }
}

/**
 * Computes shifted 2D-FFT magnitude spectrum for a square power-of-2 grayscale buffer.
 */
export function computeFft2dMagnitude(gray: Float64Array, size: number): Float64Array {
  const re = new Float64Array(size * size);
  const im = new Float64Array(size * size);
  re.set(gray);

  const rowRe = new Float64Array(size);
  const rowIm = new Float64Array(size);
  for (let y = 0; y < size; y++) {
    const offset = y * size;
    rowRe.set(re.subarray(offset, offset + size));
    rowIm.fill(0);
    fft1d(rowRe, rowIm);
    re.set(rowRe, offset);
    im.set(rowIm, offset);
  }

  const colRe = new Float64Array(size);
  const colIm = new Float64Array(size);
  for (let x = 0; x < size; x++) {
    for (let y = 0; y < size; y++) {
      const idx = y * size + x;
      colRe[y] = re[idx];
      colIm[y] = im[idx];
    }
    fft1d(colRe, colIm);
    for (let y = 0; y < size; y++) {
      const idx = y * size + x;
      re[idx] = colRe[y];
      im[idx] = colIm[y];
    }
  }

  const mag = new Float64Array(size * size);
  const half = size >> 1;
  for (let y = 0; y < size; y++) {
    const sy = (y + half) % size;
    for (let x = 0; x < size; x++) {
      const sx = (x + half) % size;
      const idx = y * size + x;
      mag[sy * size + sx] = Math.hypot(re[idx], im[idx]);
    }
  }
  return mag;
}

export interface LegacyFftMetrics {
  fRatio: number;
  fSpikes: number;
  isAnomalyFlagged: boolean;
  isLogCriticalFlagged: boolean;
  logMagNormalized: Uint8Array; // 256x256 ∈ [0, 255]
}

/**
 * Exact reproduction of the legacy engine.py (lines 373-399) 2D-FFT heuristic
 * on a 256x256 bilinear-resized grayscale image.
 */
export function computeLegacyEngineFft(gray256: Float64Array): LegacyFftMetrics {
  const N = 256;
  const mag = computeFft2dMagnitude(gray256, N);
  const logMag = new Float64Array(N * N);
  let minLog = Infinity;
  let maxLog = -Infinity;

  for (let i = 0; i < N * N; i++) {
    const lm = Math.log(mag[i] + 1e-7);
    logMag[i] = lm;
    if (lm < minLog) minLog = lm;
    if (lm > maxLog) maxLog = lm;
  }

  const cy = 128;
  const cx = 128;
  let outerSum = 0;
  let outerCount = 0;
  let innerSum = 0;
  let innerCount = 0;

  const logNoDc = new Float64Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const idx = y * N + x;
      const r = Math.floor(Math.hypot(x - cx, y - cy));
      if (r > 70) {
        outerSum += mag[idx];
        outerCount++;
      } else if (r > 15 && r <= 70) {
        innerSum += mag[idx];
        innerCount++;
      }
      if (y >= cy - 12 && y < cy + 12 && x >= cx - 12 && x < cx + 12) {
        logNoDc[idx] = 0;
      } else {
        logNoDc[idx] = logMag[idx];
      }
    }
  }

  const fRatio = (outerSum / Math.max(outerCount, 1)) / (innerSum / Math.max(innerCount, 1) + 1e-6);

  let mean = 0;
  for (let i = 0; i < N * N; i++) mean += logNoDc[i];
  mean /= N * N;

  let varSum = 0;
  for (let i = 0; i < N * N; i++) {
    const d = logNoDc[i] - mean;
    varSum += d * d;
  }
  const std = Math.sqrt(varSum / (N * N));

  let fSpikes = 0;
  for (let i = 0; i < N * N; i++) {
    if (logNoDc[i] > mean + 2.5 * std) fSpikes++;
  }

  const logMagNormalized = new Uint8Array(N * N);
  const span = maxLog - minLog + 1e-7;
  for (let i = 0; i < N * N; i++) {
    logMagNormalized[i] = Math.max(0, Math.min(255, Math.round(((logMag[i] - minLog) / span) * 255)));
  }

  return {
    fRatio,
    fSpikes,
    isAnomalyFlagged: (fRatio > 0.23 && fSpikes > 50) || fRatio > 0.27,
    isLogCriticalFlagged: fRatio > 0.23 || fSpikes > 100,
    logMagNormalized,
  };
}
