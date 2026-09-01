import Constants from 'expo-constants';
import { Platform } from 'react-native';

import type {
  AiTryOnConfig,
  AiTryOnResponse,
  AvatarResponse,
  BodyShapePrediction,
  FaceCustomizationRequest,
  FitGarmentResponse,
  GarmentCatalogItem,
  GarmentFitType,
  Measurements,
  PickedPhoto,
  PipelineMode,
  WearGarmentResponse,
} from '../types';

/**
 * Resolves the Flask backend's base URL (see server/app.py).
 *
 * - `EXPO_PUBLIC_API_URL` always wins, if set (e.g. pointing at a deployed
 *   server).
 * - Otherwise, when running in Expo Go / a dev client, reuse the Metro
 *   bundler's host IP (`Constants.expoConfig.hostUri`) so a physical device
 *   on the same network can reach the backend running on the dev machine.
 * - Falls back to the Android emulator's host alias or `localhost`.
 */
function resolveApiBaseUrl(): string {
  const envUrl = process.env.EXPO_PUBLIC_API_URL;
  if (envUrl) return envUrl.replace(/\/$/, '');

  const hostUri = Constants.expoConfig?.hostUri;
  if (hostUri) {
    const host = hostUri.split(':')[0];
    if (host) return `http://${host}:5000`;
  }

  return Platform.OS === 'android' ? 'http://10.0.2.2:5000' : 'http://localhost:5000';
}

export const API_BASE_URL = resolveApiBaseUrl();

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const message = (body && (body as { error?: string }).error) || res.statusText;
    throw new ApiError(res.status, message);
  }
  return body as T;
}

type UploadFile = PickedPhoto;

async function appendFile(form: FormData, field: string, file: UploadFile) {
  if (Platform.OS === 'web') {
    // The browser's FormData (used by react-native-web) needs a real Blob,
    // unlike React Native's FormData which accepts {uri, name, type}.
    const blob = await (await fetch(file.uri)).blob();
    form.append(field, blob, file.name);
  } else {
    form.append(field, file as unknown as Blob);
  }
}

export async function checkHealth(): Promise<{ status: string; mock: boolean }> {
  const res = await fetch(`${API_BASE_URL}/api/health`);
  return handleResponse(res);
}

/** Submits the email collected during onboarding (see `screens/EmailScreen`). */
export async function submitEmail(email: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/users/email`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email }),
  });
  await handleResponse<{ status: string }>(res);
}

export async function createAvatar(
  photo: UploadFile,
  measurements: Measurements
): Promise<AvatarResponse> {
  const form = new FormData();
  await appendFile(form, 'photo', photo);
  form.append('bust', measurements.bust);
  form.append('waist', measurements.waist);
  form.append('hips', measurements.hips);
  form.append('height', measurements.height);

  const res = await fetch(`${API_BASE_URL}/api/avatars`, {
    method: 'POST',
    body: form,
  });
  return handleResponse<AvatarResponse>(res);
}

/**
 * Runs Model 1's trained body-shape classifier directly on bust/waist/hips/
 * height (`POST /api/predict-body-shape`) - independent of
 * AVATAR_PIPELINE_MOCK, with no dependency on Models 3/4/6 (see
 * server/app.py's `_get_body_shape_artifacts`).
 */
export async function predictBodyShape(measurements: Measurements): Promise<BodyShapePrediction> {
  const res = await fetch(`${API_BASE_URL}/api/predict-body-shape`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(measurements),
  });
  return handleResponse<BodyShapePrediction>(res);
}

export async function getAvatar(avatarId: string): Promise<AvatarResponse> {
  const res = await fetch(`${API_BASE_URL}/api/avatars/${avatarId}`);
  return handleResponse<AvatarResponse>(res);
}

/** Resolves an `AvatarResponse.avatar_mesh_url` (e.g. `/api/avatars/<id>/mesh.glb`) to an absolute URL. */
export function getAvatarMeshUrl(avatar: AvatarResponse): string {
  return `${API_BASE_URL}${avatar.avatar_mesh_url}`;
}

/**
 * Sends the normalized facial-feature JSON derived from the guided capture
 * (see `services/facialFeatures.ts`) to rebuild the avatar's GLB mesh with
 * matching face proportions, skin tone, and hair color.
 */
export async function customizeFaceAvatar(
  avatarId: string,
  features: FaceCustomizationRequest
): Promise<{ avatar_mesh_url: string; face_texture_url?: string }> {
  const res = await fetch(`${API_BASE_URL}/api/avatars/${avatarId}/customize-face`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(features),
  });
  return handleResponse(res);
}

/** Lists the 3D garment catalog (`GET /api/garments`, see `garment_mesh.py`). */
export async function getGarments(): Promise<GarmentCatalogItem[]> {
  const res = await fetch(`${API_BASE_URL}/api/garments`);
  const body = await handleResponse<{ garments: GarmentCatalogItem[] }>(res);
  return body.garments;
}

/**
 * Builds a garment mesh shaped to this avatar's body (`POST
 * /api/avatars/<id>/wear`) and returns a URL to fetch its `.glb`.
 */
export async function wearGarment(avatarId: string, garmentId: string): Promise<WearGarmentResponse> {
  const res = await fetch(`${API_BASE_URL}/api/avatars/${avatarId}/wear`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ garment_id: garmentId }),
  });
  return handleResponse<WearGarmentResponse>(res);
}

/** Resolves a `WearGarmentResponse.garment_mesh_url` to an absolute URL. */
export function getGarmentMeshUrl(response: WearGarmentResponse): string {
  return `${API_BASE_URL}${response.garment_mesh_url}`;
}

/** Response from `POST /api/avatars/<id>/wear-photo` and `.../remove-photo`. */
export interface WearPhotoResponse {
  avatar_mesh_url: string;
  face_texture_url: string;
}

/**
 * Paints a clothing photo directly onto the avatar's own body texture
 * (`POST /api/avatars/<id>/wear-photo`, see `garment_texture_paint.py`) —
 * the primary wardrobe mechanism: unlike `wearGarment`'s separate cut
 * mesh, this follows the body's real geometry exactly (no floating/
 * misaligned overlay), at the cost of being a flat decal rather than a
 * modeled garment. `photo` is a user-picked image; pass `garmentId` instead
 * to reuse a catalog swatch's own source image server-side.
 */
export async function wearPhoto(
  avatarId: string,
  category: GarmentFitType,
  photoOrGarmentId: UploadFile | { garmentId: string }
): Promise<WearPhotoResponse> {
  const form = new FormData();
  form.append('category', category);
  if ('garmentId' in photoOrGarmentId) {
    form.append('garment_id', photoOrGarmentId.garmentId);
  } else {
    await appendFile(form, 'garment', photoOrGarmentId);
  }

  const res = await fetch(`${API_BASE_URL}/api/avatars/${avatarId}/wear-photo`, {
    method: 'POST',
    body: form,
  });
  return handleResponse<WearPhotoResponse>(res);
}

/**
 * Best-effort undo for `wearPhoto`: repaints one category's band back to a
 * flat skin tone (`POST /api/avatars/<id>/remove-photo`).
 */
export async function removePhoto(avatarId: string, category: GarmentFitType): Promise<WearPhotoResponse> {
  const form = new FormData();
  form.append('category', category);
  const res = await fetch(`${API_BASE_URL}/api/avatars/${avatarId}/remove-photo`, {
    method: 'POST',
    body: form,
  });
  return handleResponse<WearPhotoResponse>(res);
}

/** Resolves a `WearPhotoResponse`'s URLs to absolute URLs, cache-busted so
 * the viewer reloads the mesh instead of reusing a stale one at the same path. */
export function getWearPhotoUrls(response: WearPhotoResponse): { avatarMeshUrl: string; textureUrl: string } {
  const v = Date.now();
  return {
    avatarMeshUrl: `${API_BASE_URL}${response.avatar_mesh_url}?v=${v}`,
    textureUrl: `${API_BASE_URL}${response.face_texture_url}?v=${v}`,
  };
}

/** EXPERIMENTAL — extra options for `fitGarment` when opting into the
 * `multiview_tryon` research pipeline. See `docs/multiview_tryon_setup.md`. */
export interface FitGarmentOptions {
  pipelineMode?: PipelineMode;
  /** Required by the backend when `pipelineMode === 'multiview_tryon'`. */
  personFront?: UploadFile;
  /** Optional — the server auto-generates a back view from `personFront`
   * via Gemini when omitted. */
  personBack?: UploadFile;
  /** 'dress' (default): fits the `garmentFront`/`garmentBack` pair passed
   * to `fitGarment` (any garment_type — a dress, or a single top/bottom).
   * 'top_and_bottom': fits a separate top + bottom outfit instead — pass
   * `topFront`/`topBack`/`bottomFront`/`bottomBack`; `garmentFront`/
   * `garmentBack` are ignored by the server in this mode. */
  garmentMode?: 'dress' | 'top_and_bottom';
  topFront?: UploadFile;
  topBack?: UploadFile;
  bottomFront?: UploadFile;
  bottomBack?: UploadFile;
}

/**
 * Adaptive garment fitting (`POST /api/avatars/<id>/fit-garment`, see
 * `avatar_pipeline/model7_garment_fitting`): uploads a front + back photo of
 * a garment and fits it to this avatar's body via automatic feature
 * extraction + region-wise adaptive scaling. Replaces the fixed catalog
 * (`getGarments`/`wearGarment`) as the primary wardrobe flow.
 *
 * Pass `options.pipelineMode: 'multiview_tryon'` (plus `personFront`/
 * `personBack`) to opt into the EXPERIMENTAL research pipeline instead —
 * omitting `options` entirely reproduces the exact existing behavior.
 */
export async function fitGarment(
  avatarId: string,
  garmentFront: UploadFile | undefined,
  garmentBack: UploadFile | undefined,
  garmentType: GarmentFitType,
  options?: FitGarmentOptions
): Promise<FitGarmentResponse> {
  const form = new FormData();
  const isTopAndBottom = options?.garmentMode === 'top_and_bottom';

  if (!isTopAndBottom) {
    if (!garmentFront || !garmentBack) {
      throw new Error('garmentFront/garmentBack are required unless options.garmentMode is "top_and_bottom"');
    }
    await appendFile(form, 'garment_front', garmentFront);
    await appendFile(form, 'garment_back', garmentBack);
    form.append('garment_type', garmentType);
  }

  if (options?.pipelineMode) {
    form.append('pipeline_mode', options.pipelineMode);
  }
  if (options?.garmentMode) {
    form.append('garment_mode', options.garmentMode);
  }
  if (options?.personFront) {
    await appendFile(form, 'person_front', options.personFront);
  }
  if (options?.personBack) {
    await appendFile(form, 'person_back', options.personBack);
  }
  if (options?.topFront) await appendFile(form, 'top_front', options.topFront);
  if (options?.topBack) await appendFile(form, 'top_back', options.topBack);
  if (options?.bottomFront) await appendFile(form, 'bottom_front', options.bottomFront);
  if (options?.bottomBack) await appendFile(form, 'bottom_back', options.bottomBack);

  const res = await fetch(`${API_BASE_URL}/api/avatars/${avatarId}/fit-garment`, {
    method: 'POST',
    body: form,
  });
  return handleResponse<FitGarmentResponse>(res);
}

/** Resolves a `FitGarmentResponse.garment_mesh_url` to an absolute URL. */
export function getFittedGarmentMeshUrl(response: FitGarmentResponse): string {
  return `${API_BASE_URL}${response.garment_mesh_url}`;
}

/** EXPERIMENTAL — resolves the front/back virtual-try-on preview URLs from a
 * `multiview_tryon` `FitGarmentResponse`, or `null` if not present (i.e. the
 * result came from `adaptive_template`). */
export function getTryonFrontPreviewUrl(response: FitGarmentResponse): string | null {
  return response.garment_tryon_front_url ? `${API_BASE_URL}${response.garment_tryon_front_url}` : null;
}

export function getTryonBackPreviewUrl(response: FitGarmentResponse): string | null {
  return response.garment_tryon_back_url ? `${API_BASE_URL}${response.garment_tryon_back_url}` : null;
}

/** Resolves a `FitGarmentResponse.garment_texture_url` to an absolute URL, or `null` if it has none. */
export function getFittedGarmentTextureUrl(response: FitGarmentResponse): string | null {
  return response.garment_texture_url ? `${API_BASE_URL}${response.garment_texture_url}` : null;
}

/**
 * Sends a person photo + one clothing photo to the AI try-on pipeline
 * (`POST /api/ai-tryon`, see `server/ai_tryon/`). Returns a `tryon_id` used
 * to fetch the generated 2D image and to request a 3D avatar.
 */
export async function createAiTryOn(
  personPhoto: UploadFile,
  clothingPhotos: UploadFile[]
): Promise<AiTryOnResponse> {
  const form = new FormData();
  await appendFile(form, 'person_photo', personPhoto);
  for (const clothingPhoto of clothingPhotos) {
    await appendFile(form, 'clothing_photo', clothingPhoto);
  }

  const res = await fetch(`${API_BASE_URL}/api/ai-tryon`, {
    method: 'POST',
    body: form,
  });
  return handleResponse<AiTryOnResponse>(res);
}

/** Resolves an `AiTryOnResponse.generated_image_url` to an absolute URL. */
export function getAiTryOnImageUrl(tryon: AiTryOnResponse): string {
  return `${API_BASE_URL}${tryon.generated_image_url}`;
}

/**
 * Fetches the AI try-on pipeline's runtime config (`GET /api/ai-tryon/config`)
 * - whether Gemini/image-to-3D are mocked, which provider/model is active,
 * and whether their API keys are set (never the key values themselves).
 */
export async function getAiTryOnConfig(): Promise<AiTryOnConfig> {
  const res = await fetch(`${API_BASE_URL}/api/ai-tryon/config`);
  return handleResponse<AiTryOnConfig>(res);
}

/**
 * Requests the image-to-3D step for a previously generated try-on image
 * (`POST /api/ai-tryon/<id>/avatar3d`). Returns the path to the generated
 * `.glb` avatar.
 */
export async function createAiAvatar3D(tryonId: string): Promise<{ avatar_mesh_url: string }> {
  const res = await fetch(`${API_BASE_URL}/api/ai-tryon/${tryonId}/avatar3d`, {
    method: 'POST',
  });
  return handleResponse(res);
}
