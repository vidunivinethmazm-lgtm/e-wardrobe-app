import type { AvatarConfig, AvatarResponse, GarmentFitType, PickedPhoto } from '../types';

/** Pre-fills "Advanced: automatic garment fitting"'s Front/Back photos and
 * "Garment type" from a `RecommendationItem` (see `getRecommendations` in
 * `api/client.ts`) — the "Try this on your avatar" bridge from the team's
 * `/recommendation` feature into this app's AI 3D fitting flow. The user
 * still supplies their own photo; only the garment side is pre-filled. */
export interface PresetGarment {
  frontUrl: string;
  backUrl?: string;
  garmentType: GarmentFitType;
}

export type RootStackParamList = {
  Profile: undefined;
  // AI try-on flow (default): Profile -> DressPhoto -> AiTryOn -> AiAvatar.
  // If AI generation fails, AiTryOn falls back to ClassicSetup -> Email -> ...
  DressPhoto: { personPhoto: PickedPhoto };
  AiTryOn: { personPhoto: PickedPhoto; clothingPhotos: PickedPhoto[] };
  AiAvatar: { tryonId: string; generatedImageUrl: string; personPhoto: PickedPhoto };
  ClassicSetup: { personPhoto: PickedPhoto };
  // Classic flow: Email -> GenderSelect -> AvatarCreator -> Male/FemaleAvatar -> FinalizedAvatar -> Wardrobe.
  Email: { avatar: AvatarResponse; avatarConfig: AvatarConfig; remoteAvatarUrl?: string; remoteTextureUrl?: string };
  GenderSelect: { avatar: AvatarResponse; avatarConfig: AvatarConfig; remoteAvatarUrl?: string; remoteTextureUrl?: string };
  AvatarCreator: { avatar: AvatarResponse; avatarConfig: AvatarConfig; remoteAvatarUrl?: string; remoteTextureUrl?: string };
  FacePreview: { avatar: AvatarResponse; avatarConfig: AvatarConfig; remoteAvatarUrl?: string; remoteTextureUrl?: string };
  MaleAvatar: { avatar: AvatarResponse; avatarConfig: AvatarConfig; remoteAvatarUrl?: string; remoteTextureUrl?: string };
  FemaleAvatar: { avatar: AvatarResponse; avatarConfig: AvatarConfig; remoteAvatarUrl?: string; remoteTextureUrl?: string };
  FinalizedAvatar: { avatar: AvatarResponse; avatarConfig: AvatarConfig; remoteAvatarUrl?: string; remoteTextureUrl?: string };
  Wardrobe: {
    avatar: AvatarResponse;
    avatarConfig: AvatarConfig;
    remoteAvatarUrl?: string;
    remoteTextureUrl?: string;
    presetGarment?: PresetGarment;
  };
};
