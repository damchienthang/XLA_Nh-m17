import { ColormapName, mapScalarToRgba } from './colormaps';
import { ForensicResult } from './analyzer';

export interface DecodedPixels {
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
}

export async function decodeImageUrlToPixels(url: string): Promise<DecodedPixels> {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error(`Không thể tải ảnh: ${url}`));
    img.src = url;
  });

  const width = img.naturalWidth || img.width;
  const height = img.naturalHeight || img.height;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Không thể khởi tạo Canvas 2D context');
  ctx.drawImage(img, 0, 0, width, height);
  const imageData = ctx.getImageData(0, 0, width, height);
  return {
    rgba: imageData.data,
    width,
    height,
  };
}

export function paintScalarHeatmapToCanvas(
  canvas: HTMLCanvasElement,
  scalar01: Float32Array,
  width: number,
  height: number,
  colormap: ColormapName,
  gain = 1.0
): void {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const gained = new Float32Array(scalar01.length);
  for (let i = 0; i < scalar01.length; i++) {
    gained[i] = Math.max(0, Math.min(1, scalar01[i] * gain));
  }
  const rgba = mapScalarToRgba(gained, width, height, colormap, false);
  const imgData = ctx.createImageData(width, height);
  imgData.data.set(rgba);
  ctx.putImageData(imgData, 0, 0);
}

export function paintGradcamOverlayToCanvas(
  canvas: HTMLCanvasElement,
  sourceImageUrl: string,
  gradcam01: Float32Array,
  mapWidth: number,
  mapHeight: number,
  colormap: ColormapName,
  gain = 1.0,
  alpha = 0.48
): void {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    const targetW = 256;
    const targetH = 256;
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(img, 0, 0, targetW, targetH);

    // Paint heatmap onto an offscreen canvas then blend
    const heatCanvas = document.createElement('canvas');
    heatCanvas.width = mapWidth;
    heatCanvas.height = mapHeight;
    const heatCtx = heatCanvas.getContext('2d');
    if (!heatCtx) return;

    const gained = new Float32Array(gradcam01.length);
    for (let i = 0; i < gradcam01.length; i++) {
      gained[i] = Math.max(0, Math.min(1, gradcam01[i] * gain));
    }
    const rgba = mapScalarToRgba(gained, mapWidth, mapHeight, colormap, false);
    const heatData = heatCtx.createImageData(mapWidth, mapHeight);
    heatData.data.set(rgba);
    heatCtx.putImageData(heatData, 0, 0);

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.drawImage(heatCanvas, 0, 0, targetW, targetH);
    ctx.restore();
  };
  img.src = sourceImageUrl;
}

export function paintFftSpectrumToCanvas(
  canvas: HTMLCanvasElement,
  fftLogMag: Uint8Array,
  size: number,
  colormap: ColormapName = 'INFERNO'
): void {
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const rgba = mapScalarToRgba(fftLogMag, size, size, colormap, true);
  const imgData = ctx.createImageData(size, size);
  imgData.data.set(rgba);
  ctx.putImageData(imgData, 0, 0);
}

export function getResultScalarSummary(res: ForensicResult) {
  return {
    fRatio: res.features.legacyFft.fRatio,
    fSpikes: res.features.legacyFft.fSpikes,
    blockiness: res.features.jpegBlockiness,
    chromaToLuma: res.features.chromaToLuma,
    corrRG: res.features.corrRG,
    smoothNoise: res.features.smoothNoise,
    kurtosis: res.features.lapKurtosis,
    highMidRatio: res.features.nativeHighMidRatio,
    peakMedRatio: res.features.nativePeakMedRatio,
  };
}
