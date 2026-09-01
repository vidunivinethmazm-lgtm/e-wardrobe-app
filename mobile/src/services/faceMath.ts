import * as tf from '@tensorflow/tfjs';

import type { EyeColor } from '../types';

export type Rgb = [number, number, number];

export function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), Math.max(min, max));
}

export function brightness([r, g, b]: Rgb): number {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

export function colorDistance(a: Rgb, b: Rgb): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

export function averageRgb(colors: Rgb[]): Rgb {
  const sum = colors.reduce<Rgb>((acc, c) => [acc[0] + c[0], acc[1] + c[1], acc[2] + c[2]], [0, 0, 0]);
  return [Math.round(sum[0] / colors.length), Math.round(sum[1] / colors.length), Math.round(sum[2] / colors.length)];
}

export function rgbToHex([r, g, b]: Rgb): string {
  const toHex = (c: number) => clamp(Math.round(c), 0, 255).toString(16).padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

export function rgbToHsv([r, g, b]: Rgb): [number, number, number] {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const d = max - min;

  let h = 0;
  if (d !== 0) {
    if (max === rn) h = ((gn - bn) / d) % 6;
    else if (max === gn) h = (bn - rn) / d + 2;
    else h = (rn - gn) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  const s = max === 0 ? 0 : d / max;
  return [h, s, max];
}

/** Ports `face_features._classify_hair_color`'s HSV ranges to RGB [0,1]/0-360 scale. */
export function classifyHairColor(rgb: Rgb): string {
  const [h, s, v] = rgbToHsv(rgb);
  if (v <= 0.196) return 'black';
  if (h <= 60 && s >= 0.118 && v >= 0.196 && v <= 0.588) return 'brown';
  if (h >= 30 && h <= 70 && s >= 0.078 && s <= 0.588 && v >= 0.588) return 'blonde';
  if (h <= 40 && s >= 0.392 && v >= 0.392 && v <= 0.784) return 'red';
  if (s <= 0.118 && v >= 0.392 && v <= 0.784) return 'gray';
  return 'brown';
}

const EYE_COLOR_RANGES: Record<EyeColor, [Rgb, Rgb]> = {
  brown: [[50, 20, 10], [120, 60, 30]],
  blue: [[80, 100, 150], [180, 200, 255]],
  green: [[60, 100, 40], [150, 180, 100]],
  hazel: [[100, 80, 40], [160, 140, 80]],
  gray: [[100, 100, 100], [180, 180, 180]],
};

/** Ports `face_features._detect_eye_color`'s nearest-midpoint RGB matching. */
export function classifyEyeColor(rgb: Rgb): EyeColor {
  let best: EyeColor = 'brown';
  let bestDist = Infinity;
  for (const [color, [lo, hi]] of Object.entries(EYE_COLOR_RANGES) as [EyeColor, [Rgb, Rgb]][]) {
    const mid: Rgb = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    const d = colorDistance(rgb, mid);
    if (d < bestDist) {
      bestDist = d;
      best = color;
    }
  }
  return best;
}

/** Crop rectangle (in pixels) for a detected face's "avatar face" region —
 * face plus hair, stopping above the neck/shoulders. A generous top margin
 * captures the hairline and top of the head, generous side margins capture
 * hair around the ears, and a small bottom margin keeps the chin in frame
 * without pulling in the neck. Clamped to the image bounds. Shared by
 * `faceAnalysis.ts`'s local face-texture crop and `faceCrop.ts`'s
 * server-bound face image. */
export function computeFaceCropRect(
  box: { xMin: number; yMin: number; width: number; height: number },
  imgWidth: number,
  imgHeight: number
): { originX: number; originY: number; width: number; height: number } {
  const topMargin = box.height * 0.18; // wider forehead coverage so the warp reaches the avatar's head cap
  const bottomMargin = box.height * 0.06;
  const sideMargin = box.width * 0.08;

  const originX = Math.max(0, Math.round(box.xMin - sideMargin));
  const originY = Math.max(0, Math.round(box.yMin - topMargin));
  const width = Math.min(imgWidth - originX, Math.round(box.width + 2 * sideMargin));
  const height = Math.min(imgHeight - originY, Math.round(box.height + topMargin + bottomMargin));
  return { originX, originY, width, height };
}

export function estimateFaceShape(widthHeightRatio: number, jawCheekRatio: number): string {
  if (widthHeightRatio > 0.95) {
    return jawCheekRatio > 0.85 ? 'square' : 'round';
  }
  if (widthHeightRatio < 0.72) {
    return 'oblong';
  }
  return jawCheekRatio < 0.78 ? 'heart' : 'oval';
}

/** Average RGB of a small square patch centered on `(x, y)`. */
export async function samplePatchRgb(imageTensor: tf.Tensor3D, x: number, y: number, size = 6): Promise<Rgb> {
  const [height, width] = imageTensor.shape;
  const w = Math.max(1, Math.min(size, width));
  const h = Math.max(1, Math.min(size, height));
  const x0 = clamp(Math.round(x - w / 2), 0, width - w);
  const y0 = clamp(Math.round(y - h / 2), 0, height - h);

  const patch = tf.slice(imageTensor, [y0, x0, 0], [h, w, 3]);
  const data = await patch.data();
  patch.dispose();

  let r = 0;
  let g = 0;
  let b = 0;
  const count = data.length / 3;
  for (let i = 0; i < data.length; i += 3) {
    r += data[i];
    g += data[i + 1];
    b += data[i + 2];
  }
  return [Math.round(r / count), Math.round(g / count), Math.round(b / count)];
}

/** Variance of grayscale values in a small square patch centered on `(x, y)`. */
export async function samplePatchVariance(imageTensor: tf.Tensor3D, x: number, y: number, size: number): Promise<number> {
  const [height, width] = imageTensor.shape;
  const w = Math.max(1, Math.min(size, width));
  const h = Math.max(1, Math.min(size, height));
  const x0 = clamp(Math.round(x - w / 2), 0, width - w);
  const y0 = clamp(Math.round(y - h / 2), 0, height - h);

  const patch = tf.slice(imageTensor, [y0, x0, 0], [h, w, 3]);
  const data = await patch.data();
  patch.dispose();

  const gray: number[] = [];
  for (let i = 0; i < data.length; i += 3) {
    gray.push(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
  }
  const mean = gray.reduce((a, b) => a + b, 0) / gray.length;
  return gray.reduce((a, b) => a + (b - mean) ** 2, 0) / gray.length;
}
