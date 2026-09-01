import type { AvatarConfig, BodyAdjustments, BodyScale, Measurements, MorphTargetName, ProportionKey } from '../types';

/** Reference heights (cm) — matches generate_test_avatars.py's base meshes. */
const REFERENCE_HEIGHT: Record<'male' | 'female', number> = { male: 170, female: 160 };

const HEIGHT_SCALE_RANGE: [number, number] = [0.85, 1.25];
// heightOffset of +-1 -> +-15% height change, layered on the detected heightScale.
const HEIGHT_OFFSET_SCALE = 0.15;
const NEUTRAL_WIDTH_FRACTION = 0.25; // PARAM_RANGES midpoint -> morph weight 0
const SHOULDER_WIDTH_RANGE: [number, number] = [0.18, 0.32];
const HIP_WIDTH_RANGE: [number, number] = [0.16, 0.34];

const GENDER_FACTORS: Record<'male' | 'female', { shoulder: number; hip: number; chest: number }> = {
  male: { shoulder: 1.12, hip: 0.88, chest: 1.08 },
  female: { shoulder: 0.90, hip: 1.12, chest: 0.95 },
};

/** faceShape -> headWidth weight. */
const HEAD_WIDTH_BY_FACE_SHAPE: Record<string, number> = {
  round: 0.6, square: 0.5, oblong: -0.5, oval: 0, heart: 0.1,
};

export function clamp(value: number, [lo, hi]: [number, number]): number {
  return Math.min(hi, Math.max(lo, value));
}

/** Normalizes a width-of-height fraction to [-1,1], 0 == NEUTRAL_WIDTH_FRACTION. */
function widthFractionToWeight(fraction: number, range: [number, number]): number {
  const clipped = clamp(fraction, range);
  if (clipped >= NEUTRAL_WIDTH_FRACTION) {
    const span = range[1] - NEUTRAL_WIDTH_FRACTION;
    return span > 0 ? clamp((clipped - NEUTRAL_WIDTH_FRACTION) / span, [0, 1]) : 0;
  }
  const span = NEUTRAL_WIDTH_FRACTION - range[0];
  return span > 0 ? clamp((clipped - NEUTRAL_WIDTH_FRACTION) / span, [-1, 0]) : 0;
}

/**
 * Maps body measurements + detected face shape to a {@link BodyScale} for the
 * base humanoid mesh's morph targets (see `MORPH_TARGET_NAMES` in
 * `generate_test_avatars.py`). Ports the chest/hip-width-fraction formulas
 * and `PARAM_RANGES["shoulder_width"]`/`["hip_width"]` from
 * `avatar_pipeline/model6_body3d/params.py`.
 */
export function computeBodyScale(
  measurements: Measurements,
  gender: 'male' | 'female',
  faceShape: string
): BodyScale {
  const bust = Number(measurements.bust);
  const waist = Number(measurements.waist);
  const hips = Number(measurements.hips);
  const heightCm = Number(measurements.height);

  const heightScale = clamp(
    Number.isFinite(heightCm) && heightCm > 0 ? heightCm / REFERENCE_HEIGHT[gender] : 1,
    HEIGHT_SCALE_RANGE
  );

  const gf = GENDER_FACTORS[gender];
  const chestWidthFraction =
    Number.isFinite(bust) && bust > 0 && heightCm > 0
      ? ((bust / Math.PI) * 1.1) / heightCm * gf.chest
      : NEUTRAL_WIDTH_FRACTION;
  const hipWidthFraction =
    Number.isFinite(hips) && hips > 0 && heightCm > 0
      ? ((hips / Math.PI) * 1.1) / heightCm * gf.hip
      : NEUTRAL_WIDTH_FRACTION;
  const shoulderWidthFraction = chestWidthFraction * gf.shoulder;

  const shoulderWidth = widthFractionToWeight(shoulderWidthFraction, SHOULDER_WIDTH_RANGE);
  const hipWidth = widthFractionToWeight(hipWidthFraction, HIP_WIDTH_RANGE);

  // Taller-than-reference -> modestly longer limbs.
  const heightDeviation = heightScale - 1;
  const armLength = clamp(heightDeviation * 2, [-1, 1]);
  const legLength = clamp(heightDeviation * 2, [-1, 1]);

  // Waist large relative to bust/hips -> positive (rounder); small -> negative (slimmer).
  let bodyType = 0;
  if (bust > 0 && waist > 0 && hips > 0) {
    const waistToFrame = waist / ((bust + hips) / 2);
    bodyType = clamp((waistToFrame - 1.0) * 2.5, [-1, 1]);
  }

  const headWidth = HEAD_WIDTH_BY_FACE_SHAPE[faceShape] ?? 0;

  const morphWeights: Record<MorphTargetName, number> = {
    shoulderWidth, hipWidth, armLength, legLength, bodyType, headWidth,
  };

  return { heightScale, morphWeights };
}

export interface BodyTypePreset {
  label: string;
  value: number;
}

/** Discrete body-type silhouettes a user can pick to override the detected
 * `bodyType` morph weight, from slim (-1) to plus (1). */
export const BODY_TYPE_PRESETS: BodyTypePreset[] = [
  { label: 'Slim', value: -1 },
  { label: 'Athletic', value: -0.5 },
  { label: 'Average', value: 0 },
  { label: 'Curvy', value: 0.5 },
  { label: 'Plus', value: 1 },
];

/** Index into BODY_TYPE_PRESETS whose value is nearest `bodyType` - used to
 * highlight the preset matching the detected (or overridden) body type. */
export function closestBodyTypePresetIndex(bodyType: number): number {
  let closest = 0;
  let bestDist = Infinity;
  BODY_TYPE_PRESETS.forEach((preset, i) => {
    const dist = Math.abs(preset.value - bodyType);
    if (dist < bestDist) {
      bestDist = dist;
      closest = i;
    }
  });
  return closest;
}

export const DEFAULT_BODY_ADJUSTMENTS: BodyAdjustments = {
  bodyTypeOverride: null,
  proportionOffsets: { shoulderWidth: 0, hipWidth: 0, armLength: 0, legLength: 0, headWidth: 0 },
  heightOffset: 0,
  skinTone: 0,
};

/** Blends `value` (0-1) toward `target` (0 or 1) by fraction `t` (0-1). */
function blendToward(value: number, target: number, t: number): number {
  return clamp(value + (target - value) * t, [0, 1]);
}

/**
 * Returns a new `AvatarConfig` with `bodyScale.morphWeights` and `skinColor`
 * adjusted per `adjustments`. Does not mutate `config`.
 */
export function applyBodyAdjustments(config: AvatarConfig, adjustments: BodyAdjustments): AvatarConfig {
  const detected = config.bodyScale.morphWeights;
  const proportionKeys: ProportionKey[] = ['shoulderWidth', 'hipWidth', 'armLength', 'legLength', 'headWidth'];

  const morphWeights = { ...detected };
  for (const key of proportionKeys) {
    morphWeights[key] = clamp(detected[key] + adjustments.proportionOffsets[key], [-1, 1]);
  }
  morphWeights.bodyType = adjustments.bodyTypeOverride ?? detected.bodyType;

  const heightScale = clamp(
    config.bodyScale.heightScale + adjustments.heightOffset * HEIGHT_OFFSET_SCALE,
    HEIGHT_SCALE_RANGE
  );

  const t = Math.abs(adjustments.skinTone) * 0.5;
  const target = adjustments.skinTone >= 0 ? 1 : 0;
  const skinColor: [number, number, number] = [
    blendToward(config.skinColor[0], target, t),
    blendToward(config.skinColor[1], target, t),
    blendToward(config.skinColor[2], target, t),
  ];

  return { ...config, bodyScale: { heightScale, morphWeights }, skinColor };
}
