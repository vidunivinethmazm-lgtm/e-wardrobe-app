import type * as tf from '@tensorflow/tfjs';
import type { Face } from '@tensorflow-models/face-landmarks-detection';
import * as ImageManipulator from 'expo-image-manipulator';

import type { AvatarFeatures, EyeColor, FaceCustomizationRequest, FacialHair, HairStyle, NormalizedFacialFeatures } from '../types';
import { cropFaceToBase64, cropFaceToUri } from './faceCrop';
import { getDetectorWithTimeout } from './faceDetector';
import {
  averageRgb,
  brightness,
  classifyEyeColor,
  classifyHairColor,
  colorDistance,
  computeFaceCropRect,
  dist,
  estimateFaceShape,
  rgbToHex,
  samplePatchRgb,
  samplePatchVariance,
  type Rgb,
} from './faceMath';
import { decodeImageUri, loadImageAsTensor } from './imageTensor';

/** Used when no face can be detected in the photo (or analysis fails). */
export const DEFAULT_AVATAR_FEATURES: AvatarFeatures = {
  gender: 'female',
  ageGroup: '20s',
  skinTone: '#C68642',
  hairColor: 'brown',
  hairStyle: 'short',
  faceShape: 'oval',
  facialHair: 'none',
  eyeColor: 'brown',
  skinRgb: [198, 134, 66],
  hairRgb: [60, 45, 35],
  confidence: 0,
};

export interface FaceAnalysisResult {
  features: AvatarFeatures;
  /** Local file URI for the detected face crop, or null when no face was found. */
  faceTextureUri: string | null;
  /**
   * Pass to `customizeFaceAvatar()` to update the 3D avatar's head shape and
   * skin tone. The user's face image is kept separate for the "Avatar face"
   * preview and is not painted onto the avatar mesh.
   */
  faceCustomization: FaceCustomizationRequest;
}

/**
 * Used for `faceCustomization` when no face could be detected at all - keeps
 * the avatar's head proportions unchanged (`jawWidth` matches
 * `face_customization.py`'s `_NEUTRAL_JAW_RATIO`) while still applying the
 * fallback crop as the head texture.
 */
const FALLBACK_NORMALIZED_FEATURES: NormalizedFacialFeatures = {
  faceShape: DEFAULT_AVATAR_FEATURES.faceShape,
  jawWidth: 0.8,
  noseWidth: 0.25,
  eyeSpacing: 0.45,
  skinTone: DEFAULT_AVATAR_FEATURES.skinTone,
  hairColor: DEFAULT_AVATAR_FEATURES.hairColor,
};

/**
 * Analyzes a selfie/photo on-device using MediaPipe Face Mesh (via the TFJS
 * runtime) and derives the `AvatarFeatures` used to build a realistic avatar.
 *
 * This is a heuristic, best-effort analysis (no server round-trip): it reads
 * facial landmark geometry for face shape/gender, and samples pixel colors
 * around the cheeks, hairline, and irises for skin/hair/eye color. Confidence
 * is intentionally low relative to a dedicated classifier.
 *
 * If MediaPipe can't find a face (common for full-body gallery photos, where
 * the face is a small part of the frame), this retries on a zoomed-in crop of
 * the photo's top portion, then falls back to `DEFAULT_AVATAR_FEATURES` with a
 * null `faceTextureUri` so a dirty background crop is not presented as a face
 * texture or painted onto the avatar's head.
 */
export async function analyzeFace(photoUri: string): Promise<FaceAnalysisResult> {
  const { tensor: imageTensor, uri, width, height } = await loadImageAsTensor(photoUri);

  let activeTensor = imageTensor;
  let activeUri = uri;
  let activeWidth = width;
  let activeHeight = height;

  try {
    let faces: Face[] = [];
    try {
      const detector = await getDetectorWithTimeout();
      faces = await detector.estimateFaces(activeTensor, { flipHorizontal: false });

      if (faces.length === 0) {
        // Full-body photos (gallery picks) leave the face as a small fraction
        // of the frame, which the detector can miss. Zoom into the top
        // portion of the photo - where a standing subject's head is - and
        // retry before giving up.
        const zoomed = await cropTopPortion(activeUri, activeWidth, activeHeight);
        activeTensor.dispose();
        const decoded = await decodeImageUri(zoomed.uri);
        activeTensor = decoded.tensor;
        activeUri = zoomed.uri;
        activeWidth = decoded.width;
        activeHeight = decoded.height;
        faces = await detector.estimateFaces(activeTensor, { flipHorizontal: false });
      }
    } catch {
      // The on-device model may still be downloading (DetectorTimeoutError)
      // or fail to load entirely. Either way, fall through to the
      // no-face-detected fallback below so "Avatar face" still shows the
      // generic head-and-shoulders crop instead of nothing.
      faces = [];
    }

    if (faces.length === 0) {
      return {
        features: DEFAULT_AVATAR_FEATURES,
        faceTextureUri: null,
        faceCustomization: FALLBACK_NORMALIZED_FEATURES,
      };
    }

    const face = faces[0];
    const { features, normalized } = await deriveFeatures(activeTensor, face);
    const cropRect = computeFaceCropRect(face.box, activeWidth, activeHeight);
    const [faceTextureUri, faceImage] = await Promise.all([
      cropFaceToUri(activeUri, face, activeWidth, activeHeight),
      cropFaceToBase64(activeUri, face, activeWidth, activeHeight),
    ]);
    return {
      features,
      faceTextureUri,
      faceCustomization: {
        ...normalized,
        faceImage,
        faceCropWidth: cropRect.width,
        faceCropHeight: cropRect.height,
      },
    };
  } finally {
    activeTensor.dispose();
  }
}

/** Crops the top `fraction` of `uri` (full width), for re-running face
 * detection on a region where a standing subject's head is more likely to
 * fill the frame. */
async function cropTopPortion(uri: string, imgWidth: number, imgHeight: number, fraction = 0.55) {
  const cropHeight = Math.max(1, Math.round(imgHeight * fraction));
  return ImageManipulator.manipulateAsync(
    uri,
    [{ crop: { originX: 0, originY: 0, width: imgWidth, height: cropHeight } }, { resize: { width: imgWidth } }],
    { format: ImageManipulator.SaveFormat.JPEG, compress: 0.9 }
  );
}

async function deriveFeatures(
  imageTensor: tf.Tensor3D,
  face: Face
): Promise<{ features: AvatarFeatures; normalized: NormalizedFacialFeatures }> {
  const [imgHeight] = imageTensor.shape;
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

  const faceHeight = dist(forehead, chin);
  const faceWidth = dist(leftFace, rightFace);
  const jawWidth = dist(leftJaw, rightJaw);
  const noseWidth = dist(leftNoseAla, rightNoseAla);
  const eyeSpacing = dist(leftIris, rightIris);
  const widthHeightRatio = faceWidth / faceHeight;
  const jawCheekRatio = jawWidth / faceWidth;

  const faceShape = estimateFaceShape(widthHeightRatio, jawCheekRatio);

  // Sample colors first so facial-hair detection can inform gender.
  const [leftCheekRgb, rightCheekRgb] = await Promise.all([
    samplePatchRgb(imageTensor, leftCheek.x, leftCheek.y),
    samplePatchRgb(imageTensor, rightCheek.x, rightCheek.y),
  ]);
  const skinRgb = averageRgb([leftCheekRgb, rightCheekRgb]);

  const hairSampleY = Math.max(0, forehead.y - faceHeight * 0.45);
  const hairRgb = await samplePatchRgb(imageTensor, forehead.x, hairSampleY);

  const chinSampleY = Math.min(imgHeight - 1, chin.y + 4);
  const chinRgb = await samplePatchRgb(imageTensor, chin.x, chinSampleY);
  const facialHair = estimateFacialHair(skinRgb, chinRgb);

  // Lower thresholds (old 0.9/0.8 missed almost every male face).
  // Facial hair is a strong override signal — stubble/beard → male.
  const geometricMale = jawCheekRatio > 0.78 && widthHeightRatio > 0.75;
  const gender: 'male' | 'female' = (geometricMale || facialHair !== 'none') ? 'male' : 'female';

  const irisRgbs = await Promise.all(
    [468, 473].map((idx) => {
      const iris = face.keypoints[idx];
      return iris ? samplePatchRgb(imageTensor, iris.x, iris.y, 3) : Promise.resolve(null);
    })
  );
  const validIrisRgbs = irisRgbs.filter((c): c is Rgb => c !== null);
  const eyeColor = validIrisRgbs.length > 0 ? classifyEyeColor(averageRgb(validIrisRgbs)) : 'brown';

  const foreheadVariance = await samplePatchVariance(imageTensor, forehead.x, forehead.y - faceHeight * 0.1, 16);
  const ageGroup = estimateAgeGroup(foreheadVariance);

  const hairStyle = await estimateHairStyle(imageTensor, face, hairRgb, skinRgb, faceHeight);

  const skinTone = rgbToHex(skinRgb);
  const hairColor = classifyHairColor(hairRgb);

  const features: AvatarFeatures = {
    gender,
    ageGroup,
    skinTone,
    hairColor,
    hairStyle,
    faceShape,
    facialHair,
    eyeColor,
    skinRgb,
    hairRgb,
    confidence: 0.6,
  };

  const normalized: NormalizedFacialFeatures = {
    faceShape,
    jawWidth: jawCheekRatio,
    noseWidth: noseWidth / faceWidth,
    eyeSpacing: eyeSpacing / faceWidth,
    skinTone,
    hairColor,
  };

  return { features, normalized };
}

function estimateFacialHair(skinRgb: Rgb, chinRgb: Rgb): FacialHair {
  const darkening = brightness(skinRgb) - brightness(chinRgb);
  if (darkening > 45) return 'beard';
  if (darkening > 20) return 'stubble';
  return 'none';
}

function estimateAgeGroup(foreheadVariance: number): AvatarFeatures['ageGroup'] {
  if (foreheadVariance > 600) return '50+';
  if (foreheadVariance > 350) return '40s';
  if (foreheadVariance > 200) return '30s';
  if (foreheadVariance > 100) return '20s';
  return 'teen';
}

async function estimateHairStyle(
  imageTensor: tf.Tensor3D,
  face: Face,
  hairRgb: Rgb,
  skinRgb: Rgb,
  faceHeight: number
): Promise<HairStyle> {
  const [imgHeight] = imageTensor.shape;
  const forehead = face.keypoints[10];
  const chin = face.keypoints[152];

  const scalpY = Math.max(0, forehead.y - faceHeight * 0.55);
  const scalpRgb = await samplePatchRgb(imageTensor, forehead.x, scalpY);
  const scalpVariance = await samplePatchVariance(imageTensor, forehead.x, scalpY, 12);

  const belowChinY = chin.y + faceHeight * 0.4;
  let hairBelowChin = false;
  if (belowChinY < imgHeight) {
    const belowChinRgb = await samplePatchRgb(imageTensor, chin.x, belowChinY);
    hairBelowChin = colorDistance(belowChinRgb, hairRgb) < colorDistance(belowChinRgb, skinRgb);
  }

  const scalpVisible = colorDistance(scalpRgb, skinRgb) < 25;

  if (scalpVisible) return 'buzz';
  if (hairBelowChin) return scalpVariance > 700 ? 'wavy' : 'long';
  return scalpVariance > 700 ? 'curly' : 'short';
}
