export type ColormapName = 'JET' | 'TURBO' | 'INFERNO' | 'MAGMA' | 'BONE';

function lerpColor(stops: [number, number, number][], t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  const scaled = clamped * (stops.length - 1);
  const idx = Math.min(stops.length - 2, Math.floor(scaled));
  const frac = scaled - idx;
  const a = stops[idx];
  const b = stops[idx + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac),
  ];
}

const STOPS: Record<ColormapName, [number, number, number][]> = {
  JET: [
    [0, 0, 128],
    [0, 0, 255],
    [0, 255, 255],
    [255, 255, 0],
    [255, 0, 0],
    [128, 0, 0],
  ],
  TURBO: [
    [48, 18, 59],
    [65, 69, 171],
    [40, 188, 235],
    [164, 252, 60],
    [251, 126, 33],
    [122, 4, 3],
  ],
  INFERNO: [
    [0, 0, 4],
    [66, 10, 104],
    [147, 38, 103],
    [221, 81, 58],
    [252, 165, 10],
    [252, 255, 164],
  ],
  MAGMA: [
    [0, 0, 4],
    [59, 15, 112],
    [140, 41, 129],
    [222, 73, 104],
    [254, 159, 109],
    [252, 253, 191],
  ],
  BONE: [
    [0, 0, 0],
    [62, 62, 86],
    [125, 140, 156],
    [194, 214, 214],
    [255, 255, 255],
  ],
};

// Precomputed 256-entry RGB LUT per colormap
const LUT_CACHE = new Map<ColormapName, Uint8ClampedArray>();

export function getColormapLut(name: ColormapName): Uint8ClampedArray {
  const existing = LUT_CACHE.get(name);
  if (existing) return existing;
  const stops = STOPS[name];
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = lerpColor(stops, i / 255);
    lut[i * 3] = r;
    lut[i * 3 + 1] = g;
    lut[i * 3 + 2] = b;
  }
  LUT_CACHE.set(name, lut);
  return lut;
}

export function mapScalarToRgba(
  scalar01: Float32Array | Uint8Array,
  width: number,
  height: number,
  colormap: ColormapName,
  isUint8 = false
): Uint8ClampedArray {
  const lut = getColormapLut(colormap);
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const idx255 = isUint8
      ? (scalar01[i] as number)
      : Math.max(0, Math.min(255, Math.round((scalar01[i] as number) * 255)));
    const p = i * 4;
    const c = idx255 * 3;
    rgba[p] = lut[c];
    rgba[p + 1] = lut[c + 1];
    rgba[p + 2] = lut[c + 2];
    rgba[p + 3] = 255;
  }
  return rgba;
}
