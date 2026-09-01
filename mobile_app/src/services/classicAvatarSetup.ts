import { API_BASE_URL, createAvatar, customizeFaceAvatar, predictBodyShape } from '../api/client';
import { analyzeFace, DEFAULT_AVATAR_FEATURES } from './faceAnalysis';
import { buildAvatar, DEFAULT_MEASUREMENTS } from './avatarBuilder';
import type { AvatarConfig, AvatarResponse, BodyShape, Measurements, PickedPhoto } from '../types';

export interface ClassicAvatarSetupResult {
  avatar: AvatarResponse;
  avatarConfig: AvatarConfig;
  remoteAvatarUrl?: string;
  remoteTextureUrl?: string;
}

/**
 * Runs the classic avatar pipeline (`POST /api/avatars`, on-device face
 * analysis, `POST /api/predict-body-shape`, then
 * `POST /api/avatars/<id>/customize-face`) for `photo` + `measurements`.
 * Used both by the classic setup flow (ClassicSetupScreen's measurement
 * form) and as the AI try-on flow's fallback when `/api/ai-tryon` fails (no
 * measurement form precedes that path, so `measurements` defaults to
 * DEFAULT_MEASUREMENTS there).
 */
export async function runClassicAvatarSetup(
  photo: PickedPhoto,
  measurements: Measurements = DEFAULT_MEASUREMENTS
): Promise<ClassicAvatarSetupResult> {
  const avatar = await createAvatar(photo, measurements);

  // /api/avatars runs in AVATAR_PIPELINE_MOCK mode by default, whose
  // body_shape comes from a TF-free rule rather than Model 1's trained
  // artifact (see server/app.py). Always re-predict with the real trained
  // model and override - this is a separate endpoint rather than a mock-mode
  // flag flip because Models 3/4/6 aren't trained yet, so
  // AVATAR_PIPELINE_MOCK=0 would crash the whole server on startup.
  let bodyShape: BodyShape = avatar.body_shape;
  try {
    const prediction = await predictBodyShape(measurements);
    bodyShape = prediction.body_shape;
    avatar.body_shape = prediction.body_shape;
    avatar.body_shape_confidence = prediction.confidence;
  } catch {
    // Real-model prediction unavailable (e.g. server started without the
    // saved_models/model1_body_shape artifacts) - fall back to whatever
    // /api/avatars already returned rather than failing the whole flow.
  }

  // The server's customize-face pipeline (`POST /api/avatars/<id>/customize-
  // face`) builds its own body mesh from a separate asset
  // (`avatar_pipeline/model6_body3d/assets/makehuman/{gender}.glb`), not the
  // RP_BODY_ASSETS this app otherwise renders. Its male.glb is a real,
  // detailed body and is used here; its female.glb is a leftover 242-vertex
  // placeholder column (see that directory's PLACEHOLDER_README.txt), so
  // it's deliberately never called for a female avatar - the local RP model
  // (via `buildAvatar`'s `bodyAsset`) is used for female instead, and
  // `remoteAvatarUrl` stays unset in that case.
  let avatarConfig: AvatarConfig;
  let remoteAvatarUrl: string | undefined;
  let remoteTextureUrl: string | undefined;
  try {
    const { features, faceTextureUri, faceCustomization } = await analyzeFace(photo.uri);
    avatarConfig = buildAvatar(features, faceTextureUri, measurements);

    if (features.gender === 'male') {
      try {
        const { avatar_mesh_url, face_texture_url } = await customizeFaceAvatar(avatar.avatar_id, {
          ...faceCustomization,
          gender: features.gender,
        });
        const ts = Date.now();
        remoteAvatarUrl = `${API_BASE_URL}${avatar_mesh_url}?v=${ts}`;
        remoteTextureUrl = face_texture_url ? `${API_BASE_URL}${face_texture_url}?v=${ts}` : undefined;
      } catch {
        // Keep the local body model; the "Avatar face" card still shows faceTextureUri.
      }
    }
  } catch {
    avatarConfig = buildAvatar(DEFAULT_AVATAR_FEATURES, null, measurements);
  }

  return { avatar, avatarConfig, remoteAvatarUrl, remoteTextureUrl };
}