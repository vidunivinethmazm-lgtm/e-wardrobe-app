import * as tf from '@tensorflow/tfjs';

import type { CaptureIssue, CapturePose, CaptureValidationResult } from '../types';
import { getDetectorWithTimeout } from './faceDetector';
import { loadImageAsTensor } from './imageTensor';

/** Mean grayscale brightness (0-255) must fall in this range. */
const MIN_BRIGHTNESS = 40;
const MAX_BRIGHTNESS = 230;

/** Detected face's bounding-box center must be within this fraction of the
 * frame's width/height from the frame center (same tolerance for every pose -
 * the silhouette guide keeps the head centered even when turned). */
const CENTER_TOLERANCE = 0.2;

/** Below this variance of the Laplacian, the image is considered blurry. */
const BLUR_VARIANCE_THRESHOLD = 50;

/** 3x3 Laplacian kernel, shaped for `tf.conv2d` ([h, w, inChannels, outChannels]).
 * Created lazily (not at module load) — the RN WebGL backend isn't
 * registered/ready yet when this module is first imported, and building a
 * tensor before `tf.ready()` resolves throws "highest priority backend
 * 'rn-webgl' has not yet been initialized". By the time `computeLaplacian
 * Variance` first runs, `validateCapture` has already awaited
 * `loadImageAsTensor` (which itself awaits `tf.ready()`), so it's safe here. */
let laplacianKernel: tf.Tensor4D | null = null;
function getLaplacianKernel(): tf.Tensor4D {
  if (!laplacianKernel) {
    laplacianKernel = tf.tensor4d([0, 1, 0, 1, -4, 1, 0, 1, 0], [3, 3, 1, 1]);
  }
  return laplacianKernel;
}

/**
 * Thrown when the on-device face detection model itself can't be loaded
 * (e.g. no network for its first-time download) - distinct from a failed
 * validation check, so the screen can offer a "couldn't verify - continue
 * anyway?" override instead of permanently blocking capture.
 */
export class ModelUnavailableError extends Error {
  constructor(cause: unknown) {
    super('Could not load the on-device face detection model.');
    this.name = 'ModelUnavailableError';
    this.cause = cause;
  }
}

/**
 * Validates a captured photo against the guided-capture requirements: a face
 * must be detected, centered, well-lit, and in focus. `pose` is accepted for
 * API symmetry with the capture step machine; the same checks/tolerances
 * apply to every pose.
 */
export async function validateCapture(uri: string, _pose: CapturePose): Promise<CaptureValidationResult> {
  const { tensor: imageTensor, width, height } = await loadImageAsTensor(uri);

  try {
    let faces;
    try {
      const detector = await getDetectorWithTimeout();
      faces = await detector.estimateFaces(imageTensor, { flipHorizontal: false });
    } catch (err) {
      throw new ModelUnavailableError(err);
    }

    const issues: CaptureIssue[] = [];

    if (faces.length === 0) {
      issues.push('no_face');
    } else {
      const { box } = faces[0];
      const faceCenterX = box.xMin + box.width / 2;
      const faceCenterY = box.yMin + box.height / 2;
      const dx = Math.abs(faceCenterX - width / 2) / width;
      const dy = Math.abs(faceCenterY - height / 2) / height;
      if (dx > CENTER_TOLERANCE || dy > CENTER_TOLERANCE) {
        issues.push('not_centered');
      }
    }

    const meanBrightness = computeMeanBrightness(imageTensor);
    if (meanBrightness < MIN_BRIGHTNESS) {
      issues.push('too_dark');
    } else if (meanBrightness > MAX_BRIGHTNESS) {
      issues.push('too_bright');
    }

    if (computeLaplacianVariance(imageTensor) < BLUR_VARIANCE_THRESHOLD) {
      issues.push('blurry');
    }

    return { ok: issues.length === 0, issues };
  } finally {
    imageTensor.dispose();
  }
}

function toGrayscale(imageTensor: tf.Tensor3D): tf.Tensor3D {
  const [r, g, b] = tf.split(imageTensor.toFloat(), 3, 2);
  return r.mul(0.299).add(g.mul(0.587)).add(b.mul(0.114)) as tf.Tensor3D;
}

function computeMeanBrightness(imageTensor: tf.Tensor3D): number {
  return tf.tidy(() => toGrayscale(imageTensor).mean().dataSync()[0]);
}

function computeLaplacianVariance(imageTensor: tf.Tensor3D): number {
  return tf.tidy(() => {
    const conv = tf.conv2d(toGrayscale(imageTensor), getLaplacianKernel(), 1, 'same');
    return tf.moments(conv).variance.dataSync()[0];
  });
}
