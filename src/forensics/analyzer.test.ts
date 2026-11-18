import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import {
  analyzeImagePixels,
  buildForensicLog,
  computeBinaryMetrics,
  computeRocCurve,
} from './analyzer';
import { DATASET_SPECIMENS } from './dataset';
import { getColormapLut, mapScalarToRgba } from './colormaps';
import { computeFft2dMagnitude } from './fft';

function decodeImageRgbaViaFfmpeg(filePath: string): { rgba: Uint8Array; width: number; height: number } {
  const dimOut = execFileSync('ffprobe', [
    '-v',
    'error',
    '-select_streams',
    'v:0',
    '-show_entries',
    'stream=width,height',
    '-of',
    'csv=p=0',
    filePath,
  ])
    .toString()
    .trim();
  const [width, height] = dimOut.split(',').map(Number);
  const rgba = execFileSync(
    'ffmpeg',
    ['-v', 'error', '-i', filePath, '-f', 'rawvideo', '-pix_fmt', 'rgba', '-'],
    { maxBuffer: 60 * 1024 * 1024 }
  );
  return { rgba: new Uint8Array(rgba), width, height };
}

test('2D-FFT Parseval/DC impulse sanity check', () => {
  const size = 64;
  const constant = new Float64Array(size * size).fill(10);
  const mag = computeFft2dMagnitude(constant, size);
  const centerIdx = (size / 2) * size + size / 2;
  assert.ok(Math.abs(mag[centerIdx] - 10 * size * size) < 1e-6);
  assert.ok(mag[0] < 1e-6);
});

test('Colormap LUT produces valid 256-entry RGBA mappings', () => {
  const lut = getColormapLut('INFERNO');
  assert.equal(lut.length, 256 * 3);
  const scalar = new Float32Array([0, 0.5, 1, 0.25]);
  const rgba = mapScalarToRgba(scalar, 2, 2, 'TURBO');
  assert.equal(rgba.length, 16);
  assert.equal(rgba[3], 255);
});

test('Binary metrics and ROC AUC compute accurately', () => {
  const labels: (0 | 1)[] = [0, 0, 1, 1];
  const probs = [0.1, 0.2, 0.8, 0.9];
  const metrics = computeBinaryMetrics(labels, probs, 0.5);
  assert.equal(metrics.accuracy, 100);
  assert.equal(metrics.recall, 100);
  assert.equal(metrics.specificity, 100);
  assert.equal(metrics.f1, 1);
  assert.equal(metrics.aucRoc, 1);

  const roc = computeRocCurve(labels, probs);
  assert.equal(roc.auc, 1);
});

test('Baseline vs Enhanced v2 evaluation across all 29 real dataset images', () => {
  const labels: (0 | 1)[] = [];
  const baseProbs: number[] = [];
  const enhProbs: number[] = [];

  for (const spec of DATASET_SPECIMENS) {
    const absPath = path.resolve(process.cwd(), 'public', spec.url.replace(/^\//, ''));
    const { rgba, width, height } = decodeImageRgbaViaFfmpeg(absPath);
    const res = analyzeImagePixels(rgba, width, height);

    assert.equal(res.residualMap01.length, 256 * 256);
    assert.equal(res.gradcamMap01.length, 256 * 256);
    assert.ok(res.enhancedProbAi >= 0 && res.enhancedProbAi <= 1);
    assert.ok(res.baselineProbAi >= 0 && res.baselineProbAi <= 1);

    const log = buildForensicLog(res, 0.5);
    assert.ok(log.length >= 5);

    labels.push(spec.label);
    baseProbs.push(res.baselineProbAi);
    enhProbs.push(res.enhancedProbAi);

    console.log(
      `${spec.group.padEnd(9)} | ${spec.filename.padEnd(23)} | y=${spec.label} | base=${(res.baselineProbAi * 100).toFixed(1).padStart(5)}% | enh=${(res.enhancedProbAi * 100).toFixed(1).padStart(5)}% | sp=${(res.subScores.spatialScore * 100).toFixed(0).padStart(3)} fr=${(res.subScores.freqScore * 100).toFixed(0).padStart(3)} ch=${(res.subScores.chromaScore * 100).toFixed(0).padStart(3)} | blk=${res.features.jpegBlockiness.toFixed(2)}`
    );
  }

  const baseMetrics = computeBinaryMetrics(labels, baseProbs, 0.5);
  const enhMetrics = computeBinaryMetrics(labels, enhProbs, 0.5);

  console.log('\n--- SUMMARY ON 29 DATASET IMAGES ---');
  console.log('Baseline v1:', baseMetrics);
  console.log('Enhanced v2:', enhMetrics);

  assert.ok(
    enhMetrics.accuracy > baseMetrics.accuracy,
    `Expected Enhanced v2 accuracy (${enhMetrics.accuracy}) to exceed Baseline (${baseMetrics.accuracy})`
  );
  assert.ok(
    enhMetrics.aucRoc > baseMetrics.aucRoc,
    `Expected Enhanced v2 AUC (${enhMetrics.aucRoc}) to exceed Baseline (${baseMetrics.aucRoc})`
  );
});
