import type { AvatarConfig, AvatarFeatures, Measurements } from '../types';
import { computeBodyScale } from './bodyScaling';

/** Default hair GLB per gender. Female gets long straight hair; male gets short hair.
 * Metro requires static require() calls, so both are spelled out. */
export const HAIR_ASSETS: Record<'male' | 'female', number> = {
  female: require('../../assets/hair/long.glb'),
  male: require('../../assets/hair/short.glb'),
};

/** `assets/avatars/rp/*.glb` — the base humanoid mesh used for every avatar
 * of that gender, replacing the old per-body-shape MakeHuman variants.
 * Both are separate downloaded Sketchfab models (not derived from one
 * another) — `female_base_mesh.glb` is 18,609 verts (mesh `Object_0`),
 * `male_base_mesh.glb` is 24,767 verts (`kakashi_Default OBJ.001_0`).
 * A reshaped-copy-of-the-male-mesh version was tried instead
 * (`scripts/feminize_base_mesh.py`, formerly wired up here as
 * `female_base_mesh_v2.glb`) but didn't read as convincingly female once
 * rendered, so this reverted to the real female-specific scan.
 *
 * These meshes have no baked morph targets, so `applyBodyMorphs` is a no-op
 * on them (only `heightScale` actually changes their look). */
export const RP_BODY_ASSETS: Record<'male' | 'female', number> = {
  male: require('../../assets/avatars/rp/male_base_mesh.glb'),
  female: require('../../assets/avatars/rp/female_base_mesh.glb'),
};

/** The default upper-body garment worn by male avatars' "Add a top" flow
 * (see `MaleAvatarScreen`) — a standalone t-shirt mesh, not pre-aligned to
 * any body, fitted onto the body at render time by
 * `AvatarViewer3D`'s `fitGarmentToBody`. */
export const TSHIRT_ASSET: number = require('../../assets/avatars/cloths/tshirt_model.glb');

/** Generic adult body measurements, used since the app no longer asks the
 * user to enter their own — proportions can be fine-tuned afterwards with
 * the avatar screens' "Customize body" sliders. */
export const DEFAULT_MEASUREMENTS: Measurements = {
  bust: '90',
  waist: '75',
  hips: '95',
  height: '165',
};

/**
 * Selects the gender-matched RP base mesh and derives the material tints
 * used to render a realistic avatar that resembles `features` (Step 5).
 */
export function buildAvatar(
  features: AvatarFeatures,
  faceTextureUri: string | null,
  measurements: Measurements
): AvatarConfig {
  return {
    gender: features.gender,
    bodyAsset: RP_BODY_ASSETS[features.gender],
    hairAsset: HAIR_ASSETS[features.gender],
    skinColor: rgbToUnit(features.skinRgb),
    faceTextureUri,
    features,
    bodyScale: computeBodyScale(measurements, features.gender, features.faceShape),
  };
}

function rgbToUnit([r, g, b]: [number, number, number]): [number, number, number] {
  return [r / 255, g / 255, b / 255];
}
