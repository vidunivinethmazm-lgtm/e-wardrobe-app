import type { AvatarConfig, AvatarResponse, PickedPhoto } from '../types';

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
  Wardrobe: { avatar: AvatarResponse; avatarConfig: AvatarConfig; remoteAvatarUrl?: string; remoteTextureUrl?: string };
};
