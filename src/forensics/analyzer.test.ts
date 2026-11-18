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
  assert.equal(enhMetrics.accuracy, 100, `Expected 100% accuracy on 29 dataset images, got ${enhMetrics.accuracy}%`);
  assert.equal(enhMetrics.aucRoc, 1, `Expected AUC=1.0 on 29 dataset images, got ${enhMetrics.aucRoc}`);
});

test('Real-world Facebook heavy JPEGs vs AI+graphic poster composites', () => {
  const userCases: { file: string; label: 0 | 1; desc: string }[] = [
    {
      file: 'public/assets/aistudio/914890198_1372191811507318_1264390257786523529_n_b6820a90.jpg',
      label: 0,
      desc: 'USER_REAL_1_dinh (1920x1440 Facebook JPEG)',
    },
    {
      file: 'public/assets/aistudio/912518870_960026286947609_4330364491680889669_n_d4862701.jpg',
      label: 0,
      desc: 'USER_REAL_2_dinh (1920x1440 Facebook JPEG with architectural louvers)',
    },
    {
      file: 'public/assets/aistudio/910494638_1882733106213036_4003375243922980008_n_31765791.jpg',
      label: 1,
      desc: 'USER_AI_ZALO_AD (926x540 AI girl + vector text composite)',
    },
  ];

  for (const item of userCases) {
    const absPath = path.resolve(process.cwd(), item.file);
    const { rgba, width, height } = decodeImageRgbaViaFfmpeg(absPath);
    const res = analyzeImagePixels(rgba, width, height);
    console.log(
      `USER_CASE | ${item.desc} | enh=${(res.enhancedProbAi * 100).toFixed(1)}% | cagu=${res.cagu.compressionMode} | blk=${res.features.jpegBlockiness.toFixed(2)} | peak=${res.features.nativePeakMedRatio.toFixed(2)}`
    );
    if (item.label === 0) {
      assert.ok(res.enhancedProbAi < 0.5, `Expected REAL (<0.5) for ${item.desc}, got ${res.enhancedProbAi}`);
    } else {
      assert.ok(res.enhancedProbAi >= 0.5, `Expected AI (>=0.5) for ${item.desc}, got ${res.enhancedProbAi}`);
    }
  }

  // Simulate extreme 590x365 low-bitrate (~15KB) Facebook JPEG over a real photo
  const simOut = '/tmp/cagu_sim_fb_590x365.jpg';
  execFileSync('ffmpeg', [
    '-y',
    '-v',
    'error',
    '-i',
    path.resolve(process.cwd(), 'public/dataset/real/coco_000000039769.png'),
    '-vf',
    'scale=590:365:force_original_aspect_ratio=increase,crop=590:365',
    '-q:v',
    '22',
    '-frames:v',
    '1',
    simOut,
  ]);
  const simDecoded = decodeImageRgbaViaFfmpeg(simOut);
  const simRes = analyzeImagePixels(simDecoded.rgba, simDecoded.width, simDecoded.height);
  console.log(
    `SIM_FB_15KB | 590x365 q=22 | enh=${(simRes.enhancedProbAi * 100).toFixed(1)}% | mode=${simRes.cagu.compressionMode} | blk=${simRes.features.jpegBlockiness.toFixed(2)}`
  );
  assert.equal(simRes.cagu.compressionMode, 'heavy_jpeg');
  assert.ok(simRes.enhancedProbAi < 0.5, `Expected REAL (<0.5) for 590x365 heavy JPEG, got ${simRes.enhancedProbAi}`);

  // Simulate smooth blue conference wall + person silhouette 590x365 heavy JPEG (~15KB)
  const studioOut = '/tmp/cagu_sim_fb_studio_590x365.jpg';
  execFileSync('ffmpeg', [
    '-y',
    '-v',
    'error',
    '-f',
    'lavfi',
    '-i',
    'color=c=#226688:s=590x365:d=1',
    '-vf',
    "drawbox=x=210:y=60:w=170:h=305:color=#1f2228:t=fill,drawbox=x=250:y=45:w=90:h=105:color=#c99476:t=fill,noise=alls=4:allf=t+u",
    '-q:v',
    '24',
    '-frames:v',
    '1',
    studioOut,
  ]);
  const studioDecoded = decodeImageRgbaViaFfmpeg(studioOut);
  const studioRes = analyzeImagePixels(studioDecoded.rgba, studioDecoded.width, studioDecoded.height);
  console.log(
    `SIM_FB_STUDIO | 590x365 q=24 | enh=${(studioRes.enhancedProbAi * 100).toFixed(1)}% | mode=${studioRes.cagu.compressionMode} | blk=${studioRes.features.jpegBlockiness.toFixed(2)}`
  );
  assert.ok(studioRes.enhancedProbAi < 0.5, `Expected REAL (<0.5) for 590x365 smooth studio JPEG, got ${studioRes.enhancedProbAi}`);
});

