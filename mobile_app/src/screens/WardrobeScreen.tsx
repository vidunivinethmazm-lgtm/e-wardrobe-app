import { Ionicons } from '@expo/vector-icons';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import * as ImagePicker from 'expo-image-picker';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Image, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

import {
  ApiError,
  fitGarment,
  getFittedGarmentMeshUrl,
  getFittedGarmentTextureUrl,
  getGarments,
  getTryonBackPreviewUrl,
  getTryonFrontPreviewUrl,
  getWearPhotoUrls,
  removePhoto,
  wearPhoto,
} from '../api/client';
import { AvatarViewer3D } from '../components/AvatarViewer3D';
import { Card } from '../components/Card';
import { GradientButton } from '../components/GradientButton';
import { Header } from '../components/Header';
import { PillBadge } from '../components/PillBadge';
import { ScreenContainer } from '../components/ScreenContainer';
import type { RootStackParamList } from '../navigation/types';
import { colors, radii, spacing, typography } from '../theme';
import type { FitGarmentResponse, GarmentCatalogItem, GarmentFitType, PickedPhoto, RegionScales } from '../types';

type Props = NativeStackScreenProps<RootStackParamList, 'Wardrobe'>;

const GARMENT_TYPES: { value: GarmentFitType; label: string; icon: keyof typeof Ionicons.glyphMap }[] = [
  { value: 'upper_body', label: 'Top', icon: 'shirt-outline' },
  { value: 'lower_body', label: 'Bottoms', icon: 'body-outline' },
  { value: 'dress', label: 'Dress', icon: 'woman-outline' },
];

const FEATURE_LABELS: Record<keyof FitGarmentResponse['garment_features'], string> = {
  shoulder_width: 'Shoulder width',
  chest_width: 'Chest width',
  waist_width: 'Waist width',
  hip_width: 'Hip width',
  sleeve_length: 'Sleeve length',
  garment_length: 'Garment length',
  neck_width: 'Neck width',
  hem_width: 'Hem width',
};

const SCALE_LABELS: Record<keyof RegionScales, string> = {
  shoulder_scale: 'Shoulder',
  chest_scale: 'Chest',
  waist_scale: 'Waist',
  hip_scale: 'Hip',
  sleeve_scale: 'Sleeve',
  length_scale: 'Length',
};

/** Whether `hex` (`#rrggbb`) is light enough that a white checkmark on top
 * of it would be hard to see — used to pick a dark checkmark instead for
 * pale swatches like the white t-shirt. */
function isLightColor(hex: string): boolean {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.7;
}

const PICKER_OPTIONS: Partial<ImagePicker.ImagePickerOptions> = {
  mediaTypes: ['images'],
  quality: 0.85,
};

export function WardrobeScreen({ route }: Props) {
  const { avatar, avatarConfig, remoteAvatarUrl, remoteTextureUrl, presetGarment } = route.params;

  // `remoteAvatarUrl` is only ever set when the server actually built a real
  // personalized mesh for THIS avatar (male, guided face-capture completed —
  // see classicAvatarSetup.ts/AvatarCreatorScreen.tsx). Without one,
  // `avatar.avatar_id`'s server-side mesh is still whatever
  // `POST /api/avatars` created initially — a generic mock placeholder,
  // unrelated to gender or RP_BODY_ASSETS. Painting onto it (the old
  // wearPhoto flow below) silently swapped the visible body to that
  // placeholder for every female avatar, and any male avatar that skipped
  // guided capture. In that case, dress the local RP model directly instead
  // (same mechanism as MaleAvatarScreen/FemaleAvatarScreen's Top/Bottoms
  // cards), never touching the server's mesh at all.
  const isLocalOnly = !remoteAvatarUrl;

  const [garmentType, setGarmentType] = useState<GarmentFitType>(presetGarment?.garmentType ?? 'upper_body');
  const [catalog, setCatalog] = useState<GarmentCatalogItem[]>([]);

  // The avatar's own mesh/texture, evolving as wearPhoto/removePhoto paint
  // (or clear) clothing directly onto the body's surface — see
  // `garment_texture_paint.py`. Starts from what FinalizedAvatarScreen
  // already built (face customization baked in). Only meaningful when
  // `!isLocalOnly`.
  const [meshUrl, setMeshUrl] = useState(remoteAvatarUrl);
  const [textureUrl, setTextureUrl] = useState(remoteTextureUrl);
  // Which swatch (or '__photo__' for a user-uploaded image) is currently
  // painted per category — UI-only, for the checkmark/highlight state;
  // doesn't affect what's actually rendered (the server-painted texture does).
  const [appliedId, setAppliedId] = useState<Partial<Record<GarmentFitType, string>>>({});
  const [wearBusy, setWearBusy] = useState(false);
  const [wearError, setWearError] = useState<string | null>(null);

  // isLocalOnly path: fabric photo applied directly onto the local RP
  // model's own geometry — both upper_body/dress (torso + short sleeve, see
  // applyUpperBodyFabric) and lower_body/dress (legs, see
  // applyLowerBodyFabric). Not a separate garment mesh: TSHIRT_ASSET's rigid
  // scan doesn't fit this body's T-pose (see applyUpperBodyFabric's doc
  // comment for why), and there's no bottoms glb at all.
  const [localTopTextureUri, setLocalTopTextureUri] = useState<string | null>(null);
  const [localBottomTextureUri, setLocalBottomTextureUri] = useState<string | null>(null);

  // "Try this on your avatar" bridge from the team's `/recommendation`
  // feature (see `PresetGarment` in navigation/types.ts) — pre-fills the
  // garment Front/Back photos from a recommended item's `image_url` (its
  // Back is the same photo too, since the recommender only has one image
  // per item; swap in a real back photo before fitting for a better result).
  const [frontPhoto, setFrontPhoto] = useState<ImagePicker.ImagePickerAsset | null>(
    presetGarment ? urlToPseudoAsset(presetGarment.frontUrl) : null
  );
  const [backPhoto, setBackPhoto] = useState<ImagePicker.ImagePickerAsset | null>(
    presetGarment ? urlToPseudoAsset(presetGarment.backUrl ?? presetGarment.frontUrl) : null
  );
  // "Advanced: automatic garment fitting"'s OWN garment-type choice — a
  // dress (the Front/Back pair above), or a separate top + bottom outfit
  // (4 photos). Entirely separate from the "Garment type" pills above
  // (those drive the swatch/paint-on "Dress your avatar" section only).
  const [advancedGarmentMode, setAdvancedGarmentMode] = useState<'dress' | 'top_and_bottom'>('dress');
  const [topFrontPhoto, setTopFrontPhoto] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [topBackPhoto, setTopBackPhoto] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [bottomFrontPhoto, setBottomFrontPhoto] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [bottomBackPhoto, setBottomBackPhoto] = useState<ImagePicker.ImagePickerAsset | null>(null);
  // "AI 3D fitting" — always on, no toggle: every fit goes through the
  // multiview_tryon research pipeline (see docs/multiview_tryon_setup.md),
  // i.e. Gemini (photo normalization + back-view generation) + Replicate
  // (IDM-VTON virtual try-on) + 3D avatar reconstruction. Its only input is
  // your own front photo below — the back view is always auto-generated,
  // never uploaded (a manually-picked "back" photo that's accidentally
  // another front-facing shot was a real failure mode; always generating
  // it guarantees an actual back view).
  const [personFrontPhoto, setPersonFrontPhoto] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [fitting, setFitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<FitGarmentResponse | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [garmentLoadStatus, setGarmentLoadStatus] = useState<'idle' | 'ready' | 'error'>('idle');
  const [garmentLoadError, setGarmentLoadError] = useState<string | null>(null);

  type PhotoSide =
    | 'front' | 'back' | 'personFront'
    | 'topFront' | 'topBack' | 'bottomFront' | 'bottomBack';
  const PHOTO_SETTERS: Record<PhotoSide, (asset: ImagePicker.ImagePickerAsset) => void> = {
    front: setFrontPhoto,
    back: setBackPhoto,
    personFront: setPersonFrontPhoto,
    topFront: setTopFrontPhoto,
    topBack: setTopBackPhoto,
    bottomFront: setBottomFrontPhoto,
    bottomBack: setBottomBackPhoto,
  };

  useEffect(() => {
    let cancelled = false;
    getGarments()
      .then((items) => {
        if (!cancelled) setCatalog(items);
      })
      .catch(() => {
        // Quick-pick swatches are a convenience on top of the upload-a-photo
        // flow below, so a catalog load failure is silent — the rest of the
        // screen (custom garment fitting) still works.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function toPickedPhoto(asset: ImagePicker.ImagePickerAsset, name: string): PickedPhoto {
    return { uri: asset.uri, name: asset.fileName ?? name, type: asset.mimeType ?? 'image/jpeg' };
  }

  async function applyQuickPick(item: GarmentCatalogItem) {
    if (isLocalOnly) {
      setWearError("Swatches aren't available for this avatar yet — use \"Upload a photo\" instead.");
      return;
    }
    setWearError(null);
    setWearBusy(true);
    try {
      const response = await wearPhoto(avatar.avatar_id, garmentType, { garmentId: item.id });
      const urls = getWearPhotoUrls(response);
      setMeshUrl(urls.avatarMeshUrl);
      setTextureUrl(urls.textureUrl);
      setAppliedId((prev) => ({ ...prev, [garmentType]: item.id }));
    } catch (err) {
      setWearError(err instanceof ApiError ? err.message : 'Could not paint this garment onto your avatar.');
    } finally {
      setWearBusy(false);
    }
  }

  async function pickAndWearPhoto() {
    setWearError(null);
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setWearError('Photo library permission is needed to choose a photo.');
      return;
    }
    const picked = await ImagePicker.launchImageLibraryAsync(PICKER_OPTIONS);
    if (picked.canceled || picked.assets.length === 0) return;

    if (isLocalOnly) {
      // No server round-trip — applied directly to the local RP model by
      // AvatarViewer3D's live-update effects (garmentMeshUrls/
      // bottomTextureUri), same as MaleAvatarScreen/FemaleAvatarScreen.
      const uri = picked.assets[0].uri;
      if (garmentType === 'upper_body' || garmentType === 'dress') setLocalTopTextureUri(uri);
      if (garmentType === 'lower_body' || garmentType === 'dress') setLocalBottomTextureUri(uri);
      setAppliedId((prev) => ({ ...prev, [garmentType]: '__photo__' }));
      return;
    }

    setWearBusy(true);
    try {
      const response = await wearPhoto(avatar.avatar_id, garmentType, toPickedPhoto(picked.assets[0], 'garment.jpg'));
      const urls = getWearPhotoUrls(response);
      setMeshUrl(urls.avatarMeshUrl);
      setTextureUrl(urls.textureUrl);
      setAppliedId((prev) => ({ ...prev, [garmentType]: '__photo__' }));
    } catch (err) {
      setWearError(err instanceof ApiError ? err.message : 'Could not paint this photo onto your avatar.');
    } finally {
      setWearBusy(false);
    }
  }

  async function handleRemove() {
    if (isLocalOnly) {
      if (garmentType === 'upper_body' || garmentType === 'dress') setLocalTopTextureUri(null);
      if (garmentType === 'lower_body' || garmentType === 'dress') setLocalBottomTextureUri(null);
      setAppliedId((prev) => {
        const next = { ...prev };
        delete next[garmentType];
        return next;
      });
      return;
    }
    setWearError(null);
    setWearBusy(true);
    try {
      const response = await removePhoto(avatar.avatar_id, garmentType);
      const urls = getWearPhotoUrls(response);
      setMeshUrl(urls.avatarMeshUrl);
      setTextureUrl(urls.textureUrl);
      setAppliedId((prev) => {
        const next = { ...prev };
        delete next[garmentType];
        return next;
      });
    } catch (err) {
      setWearError(err instanceof ApiError ? err.message : 'Could not remove this garment.');
    } finally {
      setWearBusy(false);
    }
  }

  async function pickPhoto(side: PhotoSide) {
    setError(null);
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setError('Photo library permission is needed to choose a photo.');
      return;
    }
    const picked = await ImagePicker.launchImageLibraryAsync(PICKER_OPTIONS);
    if (picked.canceled || picked.assets.length === 0) return;
    PHOTO_SETTERS[side](picked.assets[0]);
  }

  // "Top + Bottom" chains two virtual-try-on passes (see
  // run_multiview_tryon_fitting_top_and_bottom) instead of the usual one.
  const isTopAndBottom = advancedGarmentMode === 'top_and_bottom';

  async function handleFit() {
    setError(null);
    if (isTopAndBottom) {
      if (!topFrontPhoto || !topBackPhoto || !bottomFrontPhoto || !bottomBackPhoto) {
        setError('Please choose front and back photos for both the top and the bottom.');
        return;
      }
    } else if (!frontPhoto || !backPhoto) {
      setError('Please choose both a front and back photo of the garment.');
      return;
    }
    if (!personFrontPhoto) {
      setError('AI 3D fitting needs a front photo of yourself too.');
      return;
    }

    setFitting(true);
    setResult(null);
    setGarmentLoadStatus('idle');
    setGarmentLoadError(null);
    try {
      const response = await fitGarment(
        avatar.avatar_id,
        isTopAndBottom ? undefined : toPickedPhoto(frontPhoto!, 'garment-front.jpg'),
        isTopAndBottom ? undefined : toPickedPhoto(backPhoto!, 'garment-back.jpg'),
        garmentType,
        {
          pipelineMode: 'multiview_tryon',
          personFront: toPickedPhoto(personFrontPhoto, 'person-front.jpg'),
          // No person-back upload at all — the server always
          // auto-generates the back view from personFront via Gemini, so
          // it's guaranteed to actually be a back view.
          ...(isTopAndBottom
            ? {
                garmentMode: 'top_and_bottom' as const,
                topFront: toPickedPhoto(topFrontPhoto!, 'top-front.jpg'),
                topBack: toPickedPhoto(topBackPhoto!, 'top-back.jpg'),
                bottomFront: toPickedPhoto(bottomFrontPhoto!, 'bottom-front.jpg'),
                bottomBack: toPickedPhoto(bottomBackPhoto!, 'bottom-back.jpg'),
              }
            : { garmentMode: 'dress' as const }),
        }
      );
      setResult(response);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not fit this garment.');
    } finally {
      setFitting(false);
    }
  }

  function handleGarmentStatus(status: 'ready' | 'error', message?: string) {
    setGarmentLoadStatus(status);
    setGarmentLoadError(status === 'error' ? message ?? 'Could not display the fitted garment.' : null);
  }

  // Shared by topFabricTextureUri/bottomTextureUri's status callbacks
  // (isLocalOnly path) — reuses the same error banner as the server-painted
  // wearPhoto flow below.
  function handleWearError(status: 'ready' | 'error', message?: string) {
    setWearError(status === 'error' ? message ?? 'Could not paint this fabric onto your avatar.' : null);
  }

  // PIVOT: when the multiview_tryon pipeline reconstructs a full avatar
  // (Unique3D), the returned mesh REPLACES the base avatar rather than
  // being overlaid onto it as a garment — feed its URLs into the viewer's
  // avatar slot instead of the garment-overlay slot in that case.
  // Only swap the viewer's avatar for a REAL reconstruction — the mock
  // placeholder (no FULL_AVATAR_3D_PROVIDER=unique3d configured) is just a
  // rough body-shaped shell, not a usable avatar, so leave the existing
  // avatar on screen untouched and let the "Virtual try-on preview" images
  // below carry the result instead.
  const isFullAvatarReplacement = result?.is_full_avatar_replacement === true && result?.is_real_3d_generation === true;

  // The "Garment photos" (front/back) flow below still uses the older
  // separate-mesh overlay (garment_mesh.build_garment_glb /
  // model7_garment_fitting) layered on top of whatever the texture-paint
  // flow above has already baked into meshUrl/textureUrl. It's server-driven
  // regardless of isLocalOnly. Only the adaptive_template pipeline's result
  // is an overlay mesh at all — a multiview_tryon result (real OR mock) is
  // always a full-avatar mesh, never something to layer on top of the
  // existing avatar as a "garment".
  const usingAdvancedFitResult = result != null && result.pipeline_mode !== 'multiview_tryon';
  const garmentMeshUrls = usingAdvancedFitResult ? [getFittedGarmentMeshUrl(result)] : [];
  const garmentTextureUrls = usingAdvancedFitResult ? [getFittedGarmentTextureUrl(result)] : [];
  const viewerAvatarUrl = isFullAvatarReplacement
    ? getFittedGarmentMeshUrl(result!)
    : isLocalOnly
      ? undefined // falls back to the local RP model via `config`
      : meshUrl;
  const viewerTextureUrl = isFullAvatarReplacement
    ? (getFittedGarmentTextureUrl(result!) ?? undefined)
    : isLocalOnly
      ? undefined
      : textureUrl;

  return (
    <ScreenContainer>
      <Header title="Wardrobe" subtitle="Pick a swatch or upload a clothing photo to dress your avatar." />

      <Card style={styles.avatarCard}>
        <AvatarViewer3D
          key={`${viewerAvatarUrl ?? 'none'}`}
          config={avatarConfig}
          remoteAvatarUrl={viewerAvatarUrl}
          remoteTextureUrl={viewerTextureUrl}
          garmentMeshUrls={garmentMeshUrls}
          garmentTextureUrls={garmentTextureUrls}
          topFabricTextureUri={isLocalOnly ? localTopTextureUri : null}
          onTopFabricStatus={handleWearError}
          bottomTextureUri={isLocalOnly ? localBottomTextureUri : null}
          onBottomFabricStatus={handleWearError}
          onGarmentStatus={handleGarmentStatus}
        />
        {fitting && (
          <View style={styles.wearingOverlay}>
            <ActivityIndicator color={colors.primary} />
            <Text style={styles.overlayText}>Fitting garment to your avatar…</Text>
          </View>
        )}
      </Card>

      {usingAdvancedFitResult && garmentLoadStatus === 'ready' && !result.is_mock && (
        <Card style={styles.successCard}>
          <Ionicons name="checkmark-circle" size={20} color={colors.primary} />
          <Text style={styles.successText}>
            {isFullAvatarReplacement ? 'Avatar reconstructed successfully' : 'Garment fitted successfully'}
          </Text>
        </Card>
      )}

      {usingAdvancedFitResult && garmentLoadStatus === 'ready' && result.is_mock && (
        <Card style={styles.previewCard}>
          <Ionicons name="construct-outline" size={20} color={colors.textMuted} />
          <Text style={styles.previewText}>
            {isFullAvatarReplacement
              ? "Preview only — IDM-VTON/Unique3D aren't configured, so this is a non-production placeholder " +
                'avatar, not a real reconstruction from your photos.'
              : "Preview only — Unique3D/Blender aren't configured, so this is a non-production placeholder fit, " +
                'not the final garment result.'}
          </Text>
        </Card>
      )}

      {result && garmentLoadStatus === 'error' && (
        <Card style={styles.warningCard}>
          <Text style={styles.warningText}>⚠ {garmentLoadError}</Text>
        </Card>
      )}

      {result?.pipeline_mode === 'multiview_tryon' && !isFullAvatarReplacement && (
        <Card style={styles.previewCard}>
          <Ionicons name="information-circle-outline" size={20} color={colors.textMuted} />
          <Text style={styles.previewText}>
            Your 3D avatar above is unchanged — see the try-on result in "Virtual try-on preview" below.
          </Text>
        </Card>
      )}

      <Card>
        <Text style={typography.heading}>Garment type</Text>
        <View style={styles.typeRow}>
          {GARMENT_TYPES.map((type) => {
            const selected = garmentType === type.value;
            return (
              <TouchableOpacity
                key={type.value}
                activeOpacity={0.85}
                onPress={() => setGarmentType(type.value)}
                style={[styles.typePill, selected && styles.typePillSelected]}
              >
                <Ionicons name={type.icon} size={18} color={selected ? colors.surface : colors.text} />
                <Text style={[typography.body, selected && styles.typePillTextSelected]}>{type.label}</Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </Card>

      <Card>
        <Text style={typography.heading}>Dress your avatar</Text>
        <Text style={[typography.body, styles.helperText]}>
          {isLocalOnly
            ? garmentType === 'dress'
              ? "Upload a clothing/fabric photo — it's applied as both a top and full-length bottom on your avatar."
              : 'Upload a clothing/fabric photo — it fits directly onto your avatar.'
            : "Tap a swatch for an instant look, or upload any clothing/fabric photo — it's painted directly onto " +
              "your avatar's own body, so it always follows the real shape (no floating or misaligned overlays). " +
              'Top and bottoms/dress can be worn together.'}
        </Text>
        {!isLocalOnly && (
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.swatchRow}>
          {catalog
            .filter((item) => item.category === garmentType)
            .map((item) => {
              const selected = appliedId[garmentType] === item.id;
              return (
                <TouchableOpacity
                  key={item.id}
                  activeOpacity={0.85}
                  disabled={wearBusy}
                  onPress={() => applyQuickPick(item)}
                  style={styles.swatchItem}
                >
                  <View
                    style={[
                      styles.swatchCircle,
                      { backgroundColor: item.color },
                      selected && styles.swatchCircleSelected,
                    ]}
                  >
                    {selected && (
                      <Ionicons name="checkmark" size={18} color={isLightColor(item.color) ? colors.text : '#FFFFFF'} />
                    )}
                  </View>
                  <Text style={[typography.label, styles.swatchLabel]} numberOfLines={1}>
                    {item.name}
                  </Text>
                </TouchableOpacity>
              );
            })}
        </ScrollView>
        )}

        <View style={styles.wearActionsRow}>
          <GradientButton
            title={appliedId[garmentType] === '__photo__' ? 'Change photo' : 'Upload a photo'}
            variant="accent"
            onPress={pickAndWearPhoto}
            loading={wearBusy}
            style={styles.wearActionButton}
          />
          {appliedId[garmentType] && (
            <TouchableOpacity activeOpacity={0.85} disabled={wearBusy} onPress={handleRemove} style={styles.removeFabric}>
              <Ionicons name="close-circle-outline" size={20} color={colors.danger} />
              <Text style={[typography.body, styles.removeFabricText]}>Remove</Text>
            </TouchableOpacity>
          )}
        </View>
        {wearError ? <Text style={styles.error}>{wearError}</Text> : null}
      </Card>

      <Card>
        <Text style={typography.heading}>Advanced: automatic garment fitting</Text>
        <Text style={[typography.body, styles.helperText]}>
          Deforms a separate garment mesh to match front/back photos, instead of painting onto the body above — more
          detail (sleeve shape, hems) but less reliable fit on this avatar. Lay the garment flat (or on a hanger)
          against a plain background, front and back.
        </Text>

        {presetGarment && (
          <Card style={styles.previewCard}>
            <Ionicons name="sparkles-outline" size={20} color={colors.primary} />
            <Text style={styles.previewText}>
              Pre-filled from your recommended outfit — swap in the garment's own back photo below for a better
              result (its front photo was reused as a stand-in).
            </Text>
          </Card>
        )}

        <View style={styles.garmentModeRow}>
          <TouchableOpacity
            activeOpacity={0.85}
            onPress={() => setAdvancedGarmentMode('dress')}
            style={[styles.garmentModePill, advancedGarmentMode === 'dress' && styles.typePillSelected]}
          >
            <Text style={[typography.body, advancedGarmentMode === 'dress' && styles.typePillTextSelected]}>Dress</Text>
          </TouchableOpacity>
          <TouchableOpacity
            activeOpacity={0.85}
            onPress={() => setAdvancedGarmentMode('top_and_bottom')}
            style={[styles.garmentModePill, advancedGarmentMode === 'top_and_bottom' && styles.typePillSelected]}
          >
            <Text style={[typography.body, advancedGarmentMode === 'top_and_bottom' && styles.typePillTextSelected]}>
              Top + Bottom
            </Text>
          </TouchableOpacity>
        </View>

        {advancedGarmentMode === 'dress' ? (
          <View style={styles.photoPickersRow}>
            <PhotoPicker label="Front" asset={frontPhoto} onPress={() => pickPhoto('front')} />
            <PhotoPicker label="Back" asset={backPhoto} onPress={() => pickPhoto('back')} />
          </View>
        ) : (
          <>
            <Text style={[typography.body, styles.helperText]}>
              The top is applied first, then the bottom on top of that result.
            </Text>
            <Text style={[typography.label, styles.sectionLabel]}>Top</Text>
            <View style={styles.photoPickersRow}>
              <PhotoPicker label="Top (front)" asset={topFrontPhoto} onPress={() => pickPhoto('topFront')} />
              <PhotoPicker label="Top (back)" asset={topBackPhoto} onPress={() => pickPhoto('topBack')} />
            </View>
            <Text style={[typography.label, styles.sectionLabel, styles.sectionLabelSpaced]}>Bottom</Text>
            <View style={styles.photoPickersRow}>
              <PhotoPicker label="Bottom (front)" asset={bottomFrontPhoto} onPress={() => pickPhoto('bottomFront')} />
              <PhotoPicker label="Bottom (back)" asset={bottomBackPhoto} onPress={() => pickPhoto('bottomBack')} />
            </View>
          </>
        )}
      </Card>

      <Card>
        <Text style={typography.heading}>AI 3D fitting</Text>
        <Text style={[typography.body, styles.helperText]}>
          Generates a photo of you wearing the garment above, via Gemini (photo cleanup + back-view generation) +
          Replicate/IDM-VTON (the actual try-on). Upload one full-body photo of yourself — the back view is always
          generated from it automatically, so it's guaranteed to be a real back view (never a second front photo by
          mistake).
        </Text>
        <View style={styles.photoPickersRow}>
          <PhotoPicker label="You (front)" asset={personFrontPhoto} onPress={() => pickPhoto('personFront')} />
        </View>
      </Card>

      {error ? <Text style={styles.error}>{error}</Text> : null}

      <GradientButton
        title={result ? 'Fit again' : 'Fit garment to avatar'}
        onPress={handleFit}
        loading={fitting}
        disabled={
          isTopAndBottom
            ? !topFrontPhoto || !topBackPhoto || !bottomFrontPhoto || !bottomBackPhoto || !personFrontPhoto
            : !frontPhoto || !backPhoto || !personFrontPhoto
        }
      />

      {result?.pipeline_mode && (
        <Card style={styles.pipelineCard}>
          <Ionicons
            name={result.pipeline_mode === 'multiview_tryon' ? 'flask-outline' : 'shirt-outline'}
            size={18}
            color={colors.textMuted}
          />
          <Text style={styles.pipelineText}>
            {result.pipeline_mode === 'multiview_tryon'
              ? `AI 3D fitting (${result.is_real_3d_generation ? 'real' : 'mock'} 3D generation)`
              : 'Standard fitting'}
          </Text>
        </Card>
      )}

      {result?.pipeline_mode === 'multiview_tryon' && (getTryonFrontPreviewUrl(result) || getTryonBackPreviewUrl(result)) && (
        <Card>
          <Text style={typography.heading}>Virtual try-on preview</Text>
          <Text style={[typography.body, styles.helperText]}>
            Generated before 3D reconstruction — how the garment looks on you in 2D.
          </Text>
          <View style={styles.photoPickersRow}>
            {getTryonFrontPreviewUrl(result) && (
              <Image source={{ uri: getTryonFrontPreviewUrl(result)! }} style={styles.tryonPreview} />
            )}
            {getTryonBackPreviewUrl(result) && (
              <Image source={{ uri: getTryonBackPreviewUrl(result)! }} style={styles.tryonPreview} />
            )}
          </View>
        </Card>
      )}

      {result?.warnings && result.warnings.length > 0 && (
        <Card style={styles.warningCard}>
          {result.warnings.map((warning, index) => (
            <Text key={index} style={styles.warningText}>
              ⚠ {warning}
            </Text>
          ))}
        </Card>
      )}

      {result && (
        <Card>
          <TouchableOpacity
            activeOpacity={0.85}
            style={styles.detailsHeader}
            onPress={() => setDetailsOpen((open) => !open)}
          >
            <Text style={typography.heading}>Fitting details</Text>
            <Ionicons name={detailsOpen ? 'chevron-up' : 'chevron-down'} size={20} color={colors.textMuted} />
          </TouchableOpacity>

          {detailsOpen && (
            <View style={styles.detailsBody}>
              <Text style={[typography.label, styles.sectionLabel]}>
                Normalized garment features (proportions, not measurements)
              </Text>
              {(Object.keys(FEATURE_LABELS) as (keyof FitGarmentResponse['garment_features'])[]).map((key) => (
                <DetailRow key={key} label={FEATURE_LABELS[key]} value={result.garment_features[key]} />
              ))}

              {result.region_scales && (
                <>
                  <Text style={[typography.label, styles.sectionLabel, styles.sectionLabelSpaced]}>
                    Region-wise scale factors
                  </Text>
                  {(Object.keys(SCALE_LABELS) as (keyof RegionScales)[]).map((key) => (
                    <DetailRow key={key} label={SCALE_LABELS[key]} value={result.region_scales![key]} suffix="×" />
                  ))}
                </>
              )}
            </View>
          )}
        </Card>
      )}
    </ScreenContainer>
  );
}

/** Wraps a remote image URL (e.g. a `RecommendationItem.image_url`) as a
 * minimal `ImagePicker.ImagePickerAsset` stand-in — just enough for
 * `<Image source={{uri}}/>` previews and `toPickedPhoto`'s `.uri`/
 * `.fileName`/`.mimeType` reads; never actually came from the picker. */
function urlToPseudoAsset(uri: string): ImagePicker.ImagePickerAsset {
  return { uri, fileName: 'recommended-garment.jpg', mimeType: 'image/jpeg' } as ImagePicker.ImagePickerAsset;
}

function PhotoPicker({
  label,
  asset,
  onPress,
}: {
  label: string;
  asset: ImagePicker.ImagePickerAsset | null;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity activeOpacity={0.85} onPress={onPress} style={styles.photoPicker}>
      {asset ? (
        <Image source={{ uri: asset.uri }} style={styles.photoPreview} />
      ) : (
        <View style={[styles.photoPreview, styles.photoPlaceholder]}>
          <Ionicons name="camera-outline" size={28} color={colors.textMuted} />
        </View>
      )}
      <PillBadge label={label} color={asset ? colors.primary : colors.border} textColor={asset ? '#FFFFFF' : colors.text} style={styles.photoLabel} />
    </TouchableOpacity>
  );
}

function DetailRow({ label, value, suffix }: { label: string; value: number; suffix?: string }) {
  return (
    <View style={styles.detailRow}>
      <Text style={typography.body}>{label}</Text>
      <Text style={[typography.body, styles.detailValue]}>
        {value.toFixed(2)}
        {suffix ?? ''}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  avatarCard: {
    padding: spacing.sm,
    alignItems: 'center',
  },
  wearingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    backgroundColor: 'rgba(0,0,0,0.15)',
  },
  overlayText: {
    color: colors.surface,
    fontWeight: '600',
  },
  typeRow: {
    flexDirection: 'row',
    gap: spacing.sm,
    marginTop: spacing.sm,
  },
  typePill: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.xs,
    paddingVertical: spacing.sm,
    borderRadius: radii.pill,
    borderWidth: 1,
    borderColor: colors.border,
  },
  typePillSelected: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  typePillTextSelected: {
    color: colors.surface,
  },
  helperText: {
    marginTop: spacing.xs,
    marginBottom: spacing.md,
  },
  swatchRow: {
    gap: spacing.md,
    paddingVertical: spacing.xs,
  },
  swatchItem: {
    alignItems: 'center',
    width: 64,
  },
  swatchCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    borderWidth: 2,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  swatchCircleSelected: {
    borderColor: colors.primary,
    borderWidth: 3,
  },
  swatchLabel: {
    marginTop: spacing.xs,
    textAlign: 'center',
  },
  wearActionsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    marginTop: spacing.md,
  },
  wearActionButton: {
    flex: 1,
  },
  removeFabric: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  removeFabricText: {
    color: colors.danger,
    fontWeight: '600',
  },
  photoPickersRow: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  photoPicker: {
    flex: 1,
    alignItems: 'center',
    gap: spacing.xs,
  },
  photoPreview: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: radii.md,
    backgroundColor: colors.background,
  },
  photoPlaceholder: {
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: colors.border,
    borderStyle: 'dashed',
  },
  photoLabel: {
    alignSelf: 'center',
  },
  garmentModeRow: {
    flexDirection: 'row',
    gap: spacing.sm,
    marginTop: spacing.sm,
    marginBottom: spacing.md,
  },
  garmentModePill: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: spacing.sm,
    borderRadius: radii.pill,
    borderWidth: 1,
    borderColor: colors.border,
  },
  pipelineCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  pipelineText: {
    color: colors.textMuted,
    fontWeight: '600',
  },
  tryonPreview: {
    flex: 1,
    aspectRatio: 1,
    borderRadius: radii.md,
    backgroundColor: colors.background,
  },
  error: {
    color: colors.danger,
    fontWeight: '600',
    textAlign: 'center',
  },
  successCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    borderColor: colors.primary,
    borderWidth: 1,
  },
  successText: {
    color: colors.primary,
    fontWeight: '600',
  },
  previewCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    borderColor: colors.border,
    borderWidth: 1,
  },
  previewText: {
    flex: 1,
    color: colors.textMuted,
    fontWeight: '600',
  },
  warningCard: {
    borderColor: colors.danger,
    borderWidth: 1,
  },
  warningText: {
    color: colors.danger,
  },
  detailsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  detailsBody: {
    marginTop: spacing.sm,
  },
  sectionLabel: {
    color: colors.textMuted,
    marginBottom: spacing.xs,
  },
  sectionLabelSpaced: {
    marginTop: spacing.md,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: spacing.xs,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  detailValue: {
    fontWeight: '600',
  },
});
