import * as tf from '@tensorflow/tfjs';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';

let detectorPromise: Promise<faceLandmarksDetection.FaceLandmarksDetector> | null = null;

/**
 * Lazily creates (once) and returns the shared MediaPipe Face Mesh detector,
 * used by `faceAnalysis`, `captureValidation`, and `facialFeatures`.
 */
export function getDetector(): Promise<faceLandmarksDetection.FaceLandmarksDetector> {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      await tf.ready();
      return faceLandmarksDetection.createDetector(faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh, {
        runtime: 'tfjs',
        refineLandmarks: true,
        maxFaces: 1,
      });
    })();
  }
  return detectorPromise;
}

/** The MediaPipe Face Mesh weights are tens of MB and only download once per
 * session. If that first download is still in flight, `getDetectorWithTimeout`
 * gives up after this long instead of leaving a caller stuck - the download
 * keeps going in the background and is cached for later callers. */
export const DETECTOR_LOAD_TIMEOUT_MS = 30000;

export class DetectorTimeoutError extends Error {
  constructor() {
    super('Timed out waiting for the on-device face detection model to load.');
    this.name = 'DetectorTimeoutError';
  }
}

/**
 * Like `getDetector()`, but rejects with `DetectorTimeoutError` after
 * `DETECTOR_LOAD_TIMEOUT_MS` if the shared model download hasn't finished yet.
 */
export function getDetectorWithTimeout(): Promise<faceLandmarksDetection.FaceLandmarksDetector> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new DetectorTimeoutError()), DETECTOR_LOAD_TIMEOUT_MS);
    getDetector().then(
      (detector) => {
        clearTimeout(timer);
        resolve(detector);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      }
    );
  });
}
