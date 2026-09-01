import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Image, ScrollView, StyleSheet, Text, View } from 'react-native';

import { API_BASE_URL, customizeFaceAvatar } from '../api/client';
import { AvatarViewer3D } from '../components/AvatarViewer3D';
import { GradientButton } from '../components/GradientButton';
import { Header } from '../components/Header';
import { PillBadge } from '../components/PillBadge';
import { ScreenContainer } from '../components/ScreenContainer';
import type { RootStackParamList } from '../navigation/types';
import { colors, radii, spacing, typography } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'FacePreview'>;

/**
 * Face Texture Preview step — shows two cards:
 *   1. The cropped face photo with all on-device model predictions.
 *   2. The 3D avatar — the server's personalized mesh for male (fetched
 *      automatically here if `remoteAvatarUrl` wasn't already set upstream),
 *      or the local `RP_BODY_ASSETS` model for female, whose server-side
 *      equivalent is a broken placeholder (see `classicAvatarSetup.ts`).
 */
export function FacePreviewScreen({ route, navigation }: Props) {
  const { avatar, avatarConfig, remoteAvatarUrl: initialUrl, remoteTextureUrl } = route.params;
  const { features } = avatarConfig;
  const isMale = avatarConfig.gender === 'male';

  const [meshUrl, setMeshUrl] = useState<string | undefined>(initialUrl);
  const [meshLoading, setMeshLoading] = useState(isMale && !initialUrl);
  const [faceTextureUrl, setFaceTextureUrl] = useState<string | undefined>(remoteTextureUrl);
  const fetchedRef = useRef(false);

  useEffect(() => {
    if (!isMale || initialUrl || fetchedRef.current) return;
    fetchedRef.current = true;

    // No mesh URL yet for a male avatar — fetch one using the features we
    // already have from the avatar config (skin tone, face shape, hair
    // color). Neutral geometry ratios, since the face wasn't measured from
    // a selfie on this path.
    customizeFaceAvatar(avatar.avatar_id, {
      faceShape: features.faceShape,
      jawWidth: 0.8,
      noseWidth: 0.25,
      eyeSpacing: 0.45,
      skinTone: features.skinTone,
      hairColor: features.hairColor,
      gender: avatarConfig.gender,
    })
      .then(({ avatar_mesh_url, face_texture_url }) => {
        setMeshUrl(`${API_BASE_URL}${avatar_mesh_url}?v=${Date.now()}`);
        if (face_texture_url) {
          setFaceTextureUrl(`${API_BASE_URL}${face_texture_url}?v=${Date.now()}`);
        }
      })
      .catch(() => {
        // Server unavailable — falls back to the local RP model below.
      })
      .finally(() => setMeshLoading(false));
  }, []);

  function handleConfirm() {
    const bodyScreen = avatarConfig.gender === 'female' ? 'FemaleAvatar' : 'MaleAvatar';
    navigation.replace(bodyScreen, { avatar, avatarConfig, remoteAvatarUrl: meshUrl, remoteTextureUrl: faceTextureUrl });
  }

  function handleRetake() {
    navigation.goBack();
  }

  const skinRgbCss = `rgb(${avatarConfig.skinColor.map((c) => Math.round(c * 255)).join(',')})`;

  return (
    <ScreenContainer>
      <Header title="Face preview" subtitle="Confirm your face texture looks correct." />

      <PillBadge label="Face captured" color={colors.primary} textColor={colors.surface} />

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scroll}>

        {/* ── Card 1: standalone face crop + model predictions ── */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Your face analysis</Text>

          {avatarConfig.faceTextureUri ? (
            <Image
              source={{ uri: avatarConfig.faceTextureUri }}
              style={styles.faceImage}
              resizeMode="cover"
            />
          ) : (
            <View style={[styles.faceImage, styles.placeholder]}>
              <Text style={typography.body}>No face detected</Text>
            </View>
          )}

          <View style={styles.grid}>
            <InfoRow label="Skin tone">
              <View style={[styles.swatch, { backgroundColor: skinRgbCss }]} />
            </InfoRow>

            <InfoRow label="Age group">
              <PillBadge label={features.ageGroup} color={colors.primary} textColor={colors.surface} />
            </InfoRow>

            <InfoRow label="Face shape">
              <PillBadge label={features.faceShape} color={colors.accent} textColor={colors.surface} />
            </InfoRow>

            <InfoRow label="Gender">
              <PillBadge label={features.gender} color={colors.primary} textColor={colors.surface} />
            </InfoRow>

            {features.eyeColor && (
              <InfoRow label="Eye color">
                <PillBadge label={features.eyeColor} color={colors.success} textColor={colors.surface} />
              </InfoRow>
            )}

            <InfoRow label="Hair color">
              <PillBadge label={features.hairColor} color={colors.success} textColor={colors.surface} />
            </InfoRow>

            {features.facialHair && features.facialHair !== 'none' && (
              <InfoRow label="Facial hair">
                <PillBadge label={features.facialHair} color={colors.accent} textColor={colors.surface} />
              </InfoRow>
            )}
          </View>

          {features.confidence > 0 && (
            <Text style={styles.confidence}>
              Analysis confidence: {Math.round(features.confidence * 100)} %
            </Text>
          )}
        </View>

        {/* ── Card 2: 3D avatar with face texture wrapped on head ── */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Face on avatar</Text>

          {meshLoading ? (
            <View style={[styles.viewerContainer, styles.placeholder]}>
              <ActivityIndicator color={colors.primary} size="large" />
              <Text style={[typography.body, { color: colors.textMuted, marginTop: spacing.sm }]}>
                Building your avatar…
              </Text>
            </View>
          ) : (
            <>
              <View style={styles.viewerContainer}>
                <AvatarViewer3D config={avatarConfig} remoteAvatarUrl={meshUrl} remoteTextureUrl={faceTextureUrl} />
              </View>
              <Text style={styles.hint}>Drag to rotate</Text>
            </>
          )}
        </View>

        <Text style={styles.hint}>
          Make sure the photo is well-lit and your face is clearly visible before confirming.
        </Text>

        <GradientButton title="Confirm — Customize my body" onPress={handleConfirm} />
        <GradientButton title="Retake photo" variant="accent" onPress={handleRetake} />
      </ScrollView>
    </ScreenContainer>
  );
}

function InfoRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <View style={styles.infoRow}>
      <Text style={typography.label}>{label}</Text>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  scroll: {
    gap: spacing.md,
    paddingBottom: spacing.xl,
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    padding: spacing.md,
    alignItems: 'center',
    gap: spacing.sm,
    shadowColor: '#000',
    shadowOpacity: 0.08,
    shadowRadius: 12,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: colors.text,
    alignSelf: 'flex-start',
  },
  faceImage: {
    width: 200,
    height: 200,
    borderRadius: radii.md,
    backgroundColor: colors.background,
  },
  viewerContainer: {
    width: '100%',
    height: 340,
    borderRadius: radii.md,
    overflow: 'hidden',
    backgroundColor: colors.background,
  },
  placeholder: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  grid: {
    width: '100%',
    gap: spacing.xs,
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
  },
  swatch: {
    width: 28,
    height: 28,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
  },
  confidence: {
    fontSize: 12,
    color: colors.textMuted,
    alignSelf: 'flex-end',
    paddingRight: spacing.sm,
  },
  hint: {
    textAlign: 'center',
    color: colors.textMuted,
    fontSize: 13,
  },
});
