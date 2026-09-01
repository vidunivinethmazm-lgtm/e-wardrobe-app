import type { Face } from '@tensorflow-models/face-landmarks-detection';
import * as ImageManipulator from 'expo-image-manipulator';

import { computeFaceCropRect } from './faceMath';

/** Matches `mesh_builder._HEAD_TEXTURE_SIZE` — the server resizes this crop
 * to fit the avatar head mesh's front-facing UV region either way. */
const FACE_IMAGE_SIZE = 256;

/**
 * Crops the detected face (with margin) out of `uri`, returning a square
 * JPEG as base64 (no `data:` prefix) for use as the avatar's head texture
 * via `POST /api/avatars/<id>/customize-face`.
 */
export async function cropFaceToBase64(uri: string, face: Face, imgWidth: number, imgHeight: number): Promise<string> {
  const rect = computeFaceCropRect(face.box, imgWidth, imgHeight);
  const result = await ImageManipulator.manipulateAsync(
    uri,
    [{ crop: rect }, { resize: { width: FACE_IMAGE_SIZE, height: FACE_IMAGE_SIZE } }],
    { format: ImageManipulator.SaveFormat.JPEG, compress: 0.9, base64: true }
  );
  if (!result.base64) {
    throw new Error('Failed to encode the cropped face image.');
  }
  return result.base64;
}

/**
 * Crops the same face+hair "avatar face" region as `cropFaceToBase64`, but
 * returns a local file URI for on-screen display (the "Avatar face" preview
 * on the final avatar screen) instead of base64.
 */
export async function cropFaceToUri(uri: string, face: Face, imgWidth: number, imgHeight: number): Promise<string> {
  const rect = computeFaceCropRect(face.box, imgWidth, imgHeight);
  const result = await ImageManipulator.manipulateAsync(
    uri,
    [{ crop: rect }, { resize: { width: FACE_IMAGE_SIZE, height: FACE_IMAGE_SIZE } }],
    { format: ImageManipulator.SaveFormat.JPEG, compress: 0.9 }
  );
  return result.uri;
}

/**
 * Multi-angle helper: detects the face in a photo and returns a base64-
 * encoded 256×256 JPEG crop of just the face region.
 *
 * This is a lighter-weight version of ``analyzeFace()`` — it only runs face
 * detection (no full feature analysis) and returns the cropped base64 for
 * sending to the multi-angle server endpoint.
 *
 * Returns ``null`` if no face is detected.
 */
export async function detectAndCropFace(
  photoUri: string,
  detector?: import('@tensorflow-models/face-landmarks-detection').FaceLandmarksDetector,
): Promise<string | null> {
  const { decodeImageUri, loadImageAsTensor } = await import('./imageTensor');
  const { getDetectorWithTimeout } = await import('./faceDetector');

  try {
    const { tensor, uri, width, height } = await loadImageAsTensor(photoUri);
    try {
      const det = detector ?? await getDetectorWithTimeout();
      const faces = await det.estimateFaces(tensor, { flipHorizontal: false });

      if (faces.length === 0) {
        return null;
      }

      return cropFaceToBase64(uri, faces[0], width, height);
    } finally {
      tensor.dispose();
    }
  } catch {
    return null;
  }
}
