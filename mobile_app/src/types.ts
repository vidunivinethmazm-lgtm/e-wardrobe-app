export type BodyShape = 'Hourglass' | 'Pear' | 'Apple' | 'Rectangle' | 'InvertedTriangle';

export type Gender = 'male' | 'female' | 'neutral';

export type AgeGroup = 'teen' | '20s' | '30s' | '40s' | '50+';

export type HairStyle = 'short' | 'medium' | 'long' | 'curly' | 'wavy' | 'straight' | 'buzz' | 'ponytail';

export type EyeColor = 'brown' | 'blue' | 'green' | 'hazel' | 'gray';

export type FacialHair = 'none' | 'stubble' | 'beard' | 'mustache';

export type ClothingCategory = 'upper_body' | 'lower_body' | 'dress';

export interface SkinToneResult {
  label: string;
  hex: string;
  lab: number[];
  distance: number;
  confidence: number;
  avatar_render: {
    base_color: string;
    shadow_color: string;
    highlight_color: string;
  };
}

/** Comprehensive facial analysis results from photo. */
export interface FacialAnalysis {
  gender: Gender;
  age_group: AgeGroup;
  hair_color: string;
  hair_style: HairStyle;
  facial_hair: FacialHair;
  eye_color: EyeColor;
  face_shape: string;
  confidence: number;
}

/** Model 6's body-mesh parameters, as fractions of height — see
 * `avatar_pipeline/model6_body3d/params.py`'s `PARAM_NAMES`. */
export interface Body3DParams {
  head_radius: number;
  neck_radius: number;
  shoulder_width: number;
  chest_width: number;
  chest_depth: number;
  waist_width: number;
  waist_depth: number;
  hip_width: number;
  hip_depth: number;
  upper_arm_radius: number;
  forearm_radius: number;
  thigh_radius: number;
  calf_radius: number;
}

/** Response from `POST /api/predict-body-shape` — runs Model 1's trained
 * measurement-only MLP directly, independent of AVATAR_PIPELINE_MOCK and
 * with no dependency on Models 3/4/6 (see server/app.py). */
export interface BodyShapePrediction {
  body_shape: BodyShape;
  confidence: number;
  probabilities: Record<BodyShape, number>;
}

export interface AvatarResponse {
  avatar_id: string;
  avatar_image: string;
  body_shape: BodyShape;
  body_shape_confidence: number;
  skin_tone: SkinToneResult;
  /** Path to the avatar's 3D mesh (Model 6), e.g. `/api/avatars/<id>/mesh.glb`. */
  avatar_mesh_url: string;
  body3d_params: Body3DParams;
  /** Detected gender for avatar generation. */
  gender: Gender;
  /** Comprehensive facial analysis from the uploaded photo. */
  facial_analysis: FacialAnalysis;
}

/** One entry from `GET /api/garments` (see
 * `avatar_pipeline/model6_body3d/garment_mesh._GARMENT_CATALOG`). */
export interface GarmentCatalogItem {
  id: string;
  name: string;
  category: ClothingCategory;
  /** `#rrggbb` flat tint applied to the garment mesh — there's no photo,
   * the garment is a solid-color 3D mesh (see `garment_mesh.py`). */
  color: string;
}

/** Response from `POST /api/avatars/<id>/wear`. */
export interface WearGarmentResponse {
  /** Path to the garment's 3D mesh, e.g. `/api/avatars/<id>/garment.glb?garment_id=...`. */
  garment_mesh_url: string;
}

/** Request `garment_type` for `POST /api/avatars/<id>/fit-garment` — see
 * `avatar_pipeline/model7_garment_fitting`. */
export type GarmentFitType = 'dress' | 'upper_body' | 'lower_body';

/** One ranked wardrobe item from the team's `/recommendation/recommend`
 * feature (GNN + NLP outfit ranking, see `backend/recommendation/main.py`,
 * mounted by `backend/main.py` alongside this app's own `/visualization`
 * feature). `image_url` is what "Try this on your avatar" feeds into
 * `fitGarment` as `garmentFront`/`garmentBack` — see
 * `WardrobeScreen`'s `route.params.presetGarment`. */
export interface RecommendationItem {
  outfit: string;
  item_id: string | number | null;
  fabric: string;
  color: string;
  price: number;
  category: string;
  image_url: string | null;
  confidence: string;
  reason: string;
  combination: string;
  score: number;
}

export interface RecommendationResponse {
  event_class: string;
  location_detected: string;
  weather: string;
  temperature: number;
  humidity: number;
  wardrobe_source: string;
  items_considered: number;
  logic_summary: string;
  recommendations: RecommendationItem[];
}

/** Normalized (dimensionless) garment proportions extracted from the
 * uploaded front/back photos — fractions of the garment's own bounding
 * silhouette, never a claimed centimetre measurement. See
 * `avatar_pipeline/model7_garment_fitting/garment_features.py`. */
export interface GarmentFeatures {
  shoulder_width: number;
  chest_width: number;
  waist_width: number;
  hip_width: number;
  sleeve_length: number;
  garment_length: number;
  neck_width: number;
  hem_width: number;
}

/** Region-wise scale multipliers applied to the garment template mesh — see
 * `avatar_pipeline/model7_garment_fitting/region_scaling.py`. */
export interface RegionScales {
  shoulder_scale: number;
  chest_scale: number;
  waist_scale: number;
  hip_scale: number;
  sleeve_scale: number;
  length_scale: number;
}

/** Response from `POST /api/avatars/<id>/fit-garment` (Model 7 — adaptive
 * garment fitting from user-uploaded front/back photos, replaces the
 * catalog `/wear` flow). */
export interface FitGarmentResponse {
  fit_id: string;
  garment_features: GarmentFeatures;
  /** `null` for `multiview_tryon` results (PIVOT: that mode reconstructs a
   * full avatar directly — nothing is fitted onto a separate one, so there
   * are no region-wise scale factors to report). Always present for
   * `adaptive_template`. */
  region_scales: RegionScales | null;
  /** Path to the fitted garment's 3D mesh, e.g.
   * `/api/avatars/<id>/fitted-garment/<fit_id>.glb`. For `multiview_tryon`
   * results, see `is_full_avatar_replacement` — this is a full replacement
   * avatar mesh in that mode, not a garment overlay. */
  garment_mesh_url: string;
  /** Path to the fitted garment's texture atlas (uploaded front photo on
   * top half, back photo on bottom half — see `garment_mesh_generation.
   * build_front_back_atlas`), or `null` if no photo texture is available.
   * Served separately from the GLB because React Native's GLTFLoader can't
   * decode a GLB's embedded image. */
  garment_texture_url: string | null;
  status: 'ready' | 'processing';
  /** Non-fatal notices (e.g. front/back photos look inconsistent) — the fit
   * still completed, but may be less reliable. */
  warnings: string[];
  /** True unless a real Unique3D-generated mesh went through the real
   * Blender fitting backend (see `avatar_pipeline.model7_garment_fitting.
   * garment_mesh_generation`/`garment_fit_runner`). When true, this is a
   * non-production preview — the UI must label it clearly, never present
   * it as the final fitted garment. */
  is_mock: boolean;
  /** Which pipeline produced this result. Always present; `undefined` only
   * for responses cached before this field existed. See
   * `PipelineMode`/`docs/multiview_tryon_setup.md`. */
  pipeline_mode?: PipelineMode;
  /** EXPERIMENTAL fields — only present when `pipeline_mode ===
   * 'multiview_tryon'`. See `docs/multiview_tryon_setup.md`. */
  virtual_tryon_provider?: 'mock' | 'idm_vton' | 'gemini';
  /** The full-avatar 3D reconstruction provider (PIVOT: Unique3D, not a
   * garment mesh provider). */
  image_to_3d_provider?: 'mock' | 'unique3d';
  /** Always `null` for `multiview_tryon` (no separate texture stage — the
   * reconstructed avatar mesh carries its own baked texture). */
  texture_provider?: 'mock' | 'hunyuan3d_paint' | null;
  is_real_3d_generation?: boolean;
  /** True for `multiview_tryon` results: `garment_mesh_url`/
   * `garment_texture_url` point to a full reconstructed avatar mesh that
   * REPLACES the existing avatar, not a garment to overlay onto it. */
  is_full_avatar_replacement?: boolean;
  /** Virtual try-on preview images (person wearing the uploaded garment,
   * before 3D reconstruction) — only present for `multiview_tryon`. */
  garment_tryon_front_url?: string;
  garment_tryon_back_url?: string;
}

/** EXPERIMENTAL — selects which garment-fitting pipeline `fitGarment` runs.
 * `adaptive_template` is the existing, production pipeline and the
 * default; `multiview_tryon` is a research pipeline (virtual try-on ->
 * multi-view image-to-3D -> garment isolation -> avatar fitting) gated
 * behind an explicit opt-in. See `docs/multiview_tryon_setup.md`. */
export type PipelineMode = 'adaptive_template' | 'multiview_tryon';

/** A user-picked photo's file info, passed between screens and to
 * `api/client.ts` upload helpers. */
export interface PickedPhoto {
  uri: string;
  name: string;
  type: string;
}

/** Response from `POST /api/ai-tryon` (see `server/app.py`). */
export interface AiTryOnResponse {
  tryon_id: string;
  generated_image_url: string;
}

/** Response from `GET /api/ai-tryon/config` (see `server/app.py`) - the AI
 * try-on pipeline's runtime config, shown in the mobile app's debug panels
 * so mock output isn't mistaken for a real Gemini/Meshy result. */
export interface AiTryOnConfig {
  ai_tryon_mock: boolean;
  gemini_model: string;
  gemini_api_key_present: boolean;
  image_to_3d_provider: string;
  image_to_3d_timeout_s: number;
  image_to_3d_api_key_present: boolean;
}

export interface Measurements {
  bust: string;
  waist: string;
  hips: string;
  height: string;
}

/**
 * On-device facial analysis result (see `services/faceAnalysis.ts`), used to
 * pick a realistic avatar body and to color/texture it.
 */
export interface AvatarFeatures {
  gender: 'male' | 'female';
  ageGroup: AgeGroup;
  /** Detected skin tone as a `#rrggbb` hex string. */
  skinTone: string;
  hairColor: string;
  hairStyle: HairStyle;
  faceShape: string;
  facialHair?: FacialHair;
  eyeColor?: EyeColor;
  /** Detected skin color, 0-255 RGB — used to tint the avatar's skin material. */
  skinRgb: [number, number, number];
  /** Detected hair color, 0-255 RGB — used to tint the avatar's hair material. */
  hairRgb: [number, number, number];
  /** 0 (no face detected, all defaults) .. 1 (high confidence). */
  confidence: number;
}

/** Names of the 6 morph targets baked into `assets/avatars/{male,female}.glb`
 * by `generate_test_avatars.py` (see `gltf.meshes[0].extras.targetNames`). */
export type MorphTargetName =
  | 'shoulderWidth' | 'hipWidth' | 'armLength' | 'legLength' | 'bodyType' | 'headWidth';

/** Body-measurement-derived scaling applied to the base humanoid mesh.
 * `heightScale` uniformly scales the body's Y axis; `morphWeights` are set
 * as `mesh.morphTargetInfluences[...]`, each roughly in [-1, 1] (0 = base
 * mesh's built-in proportions). */
export interface BodyScale {
  heightScale: number;
  morphWeights: Record<MorphTargetName, number>;
}

/** The 5 body-proportion morph targets a user can manually adjust (excludes
 * `bodyType`, which has its own preset/override control). */
export type ProportionKey = Exclude<MorphTargetName, 'bodyType'>;

/** User-driven manual adjustments layered on top of the detected
 * `AvatarConfig.bodyScale` / `skinColor` by `applyBodyAdjustments()`. */
export interface BodyAdjustments {
  /** Overrides detected `bodyScale.morphWeights.bodyType`; null = use detected value. */
  bodyTypeOverride: number | null;
  /** Added to each detected proportion morph weight, then clamped to [-1,1]. */
  proportionOffsets: Record<ProportionKey, number>;
  /** -1 (shorter) .. 1 (taller), 0 = detected height. Scaled by HEIGHT_OFFSET_SCALE and
   * added to `bodyScale.heightScale`, then clamped to HEIGHT_SCALE_RANGE. */
  heightOffset: number;
  /** -1 (darker) .. 1 (lighter), 0 = no correction. */
  skinTone: number;
}

/** One step of the guided avatar-capture flow (`AvatarCreatorScreen`). */
export type CapturePose = 'front' | 'left' | 'right';

/** A single failed check from `services/captureValidation.ts`'s `validateCapture()`. */
export type CaptureIssue = 'no_face' | 'too_dark' | 'too_bright' | 'not_centered' | 'blurry';

/** Result of validating one captured photo against the Step 4 requirements. */
export interface CaptureValidationResult {
  ok: boolean;
  issues: CaptureIssue[];
}

/** Raw MediaPipe Face Mesh measurements (pixels) extracted from `front.jpg` by
 * `services/facialFeatures.ts` — the inputs to the normalized JSON below. */
export interface FaceMeshMeasurements {
  faceWidth: number;
  faceHeight: number;
  jawWidth: number;
  noseWidth: number;
  eyeSpacing: number;
  /** `faceHeight / faceWidth` — used to derive `faceShape`. */
  headProportions: number;
}

/**
 * Normalized facial-feature JSON sent to the server's
 * `POST /api/avatars/<id>/customize-face` to customize the humanoid base
 * avatar (Step 7/8). `jawWidth`/`noseWidth`/`eyeSpacing` are ratios of
 * `faceWidth` (resolution-independent, roughly 0-1).
 */
export interface NormalizedFacialFeatures {
  faceShape: string;
  jawWidth: number;
  noseWidth: number;
  eyeSpacing: number;
  skinTone: string;
  hairColor: string;
}

/**
 * Request body for `POST /api/avatars/<id>/customize-face`: the normalized
 * facial-feature JSON plus a cropped photo of the user's face, applied as
 * the rebuilt avatar's head texture.
 */
export interface FaceCustomizationRequest extends NormalizedFacialFeatures {
  /** Base64-encoded JPEG (no `data:` prefix) of the cropped front-facing photo. */
  faceImage?: string;
  /** User-selected gender — overrides the server's auto-detected gender when building the head mesh. */
  gender?: 'male' | 'female';
  /** Pixel width of the face crop in the original selfie (before resize to 256×256). Used server-side to scale the avatar head to match the user's actual face proportions. */
  faceCropWidth?: number;
  /** Pixel height of the face crop in the original selfie (before resize to 256×256). */
  faceCropHeight?: number;
  /**
   * Base64-encoded JPEG (no `data:` prefix) of the cropped LEFT profile photo.
   * Used by the multi-angle face texture pipeline to capture the left side of
   * the face (cheek, ear, temple) for a 360°-natural UV texture.
   */
  leftFaceImage?: string;
  /**
   * Base64-encoded JPEG (no `data:` prefix) of the cropped RIGHT profile photo.
   * Used by the multi-angle face texture pipeline to capture the right side of
   * the face (cheek, ear, temple) for a 360°-natural UV texture.
   */
  rightFaceImage?: string;
}

/**
 * Output of `services/avatarBuilder.ts`'s `buildAvatar()` — everything
 * `AvatarViewer3D` needs to compose and render a realistic GLB avatar.
 */
export interface AvatarConfig {
  gender: 'male' | 'female';
  /** require()'d module id for `assets/avatars/{male,female}.glb`. */
  bodyAsset: number;
  /** require()'d module id for the gender-default hair GLB:
   *  female → assets/hair/long.glb, male → assets/hair/short.glb. */
  hairAsset: number;
  /** 0-1 RGB, applied as the body material's color tint. */
  skinColor: [number, number, number];
  /** Local file URI for the cropped face photo, applied as the head's texture. */
  faceTextureUri: string | null;
  features: AvatarFeatures;
  /** Measurement-derived height/proportion scaling, applied to the body mesh. */
  bodyScale: BodyScale;
}
