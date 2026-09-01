import type { FaceCustomizationRequest, FaceMeshMeasurements } from '../types';
import { cropFaceToBase64, cropFaceToUri } from './faceCrop';
import { getDetectorWithTimeout } from './faceDetector';
import { averageRgb, classifyHairColor, dist, estimateFaceShape, rgbToHex, samplePatchRgb } from './faceMath';
import { loadImageAsTensor } from './imageTensor';

/**
 * Runs MediaPipe Face Mesh on the front-facing capture and derives the raw
 * pixel `measurements`, the `normalized` facial-feature JSON (including the
 * cropped face photo as `faceImage`, painted onto the avatar mesh's head
 * texture server-side), and a local `faceTextureUri` kept separate for the
 * "Avatar face" preview.
 */
export async function extractNormalizedFeatures(
  frontUri: string
): Promise<{
  measurements: FaceMeshMeasurements;
  normalized: FaceCustomizationRequest;
  faceTextureUri: string;
}> {
  const { tensor: imageTensor, uri, width, height } = await loadImageAsTensor(frontUri);

  try {
    const detector = await getDetectorWithTimeout();
    const faces = await detector.estimateFaces(imageTensor, { flipHorizontal: false });
    if (faces.length === 0) {
      throw new Error('No face detected in the front-facing photo.');
    }

    const face = faces[0];
    const kp = (index: number) => face.keypoints[index];

    const forehead = kp(10);
    const chin = kp(152);
    const leftFace = kp(234);
    const rightFace = kp(454);
    const leftJaw = kp(172);
    const rightJaw = kp(397);
    const leftNoseAla = kp(129);
    const rightNoseAla = kp(358);
    const leftIris = kp(468);
    const rightIris = kp(473);
    const leftCheek = kp(50);
    const rightCheek = kp(280);

    const faceWidth = dist(leftFace, rightFace);
    const faceHeight = dist(forehead, chin);
    const jawWidth = dist(leftJaw, rightJaw);
    const noseWidth = dist(leftNoseAla, rightNoseAla);
    const eyeSpacing = dist(leftIris, rightIris);
    const headProportions = faceHeight / faceWidth;

    const measurements: FaceMeshMeasurements = {
      faceWidth,
      faceHeight,
      jawWidth,
      noseWidth,
      eyeSpacing,
      headProportions,
    };

    const [leftCheekRgb, rightCheekRgb] = await Promise.all([
      samplePatchRgb(imageTensor, leftCheek.x, leftCheek.y),
      samplePatchRgb(imageTensor, rightCheek.x, rightCheek.y),
    ]);
    const skinRgb = averageRgb([leftCheekRgb, rightCheekRgb]);

    const hairSampleY = Math.max(0, forehead.y - faceHeight * 0.45);
    const hairRgb = await samplePatchRgb(imageTensor, forehead.x, hairSampleY);

    const widthHeightRatio = faceWidth / faceHeight;
    const jawCheekRatio = jawWidth / faceWidth;

    const [faceTextureUri, faceImage] = await Promise.all([
      cropFaceToUri(uri, face, width, height),
      cropFaceToBase64(uri, face, width, height),
    ]);

    const normalized: FaceCustomizationRequest = {
      faceShape: estimateFaceShape(widthHeightRatio, jawCheekRatio),
      jawWidth: jawCheekRatio,
      noseWidth: noseWidth / faceWidth,
      eyeSpacing: eyeSpacing / faceWidth,
      skinTone: rgbToHex(skinRgb),
      hairColor: classifyHairColor(hairRgb),
      faceImage,
    };

    return { measurements, normalized, faceTextureUri };
  } finally {
    imageTensor.dispose();
  }
}
