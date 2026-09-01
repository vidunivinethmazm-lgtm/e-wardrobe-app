import { Ionicons } from '@expo/vector-icons';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import * as ImagePicker from 'expo-image-picker';
import { useMemo, useState } from 'react';
import { StyleSheet, Switch, Text, TouchableOpacity, View } from 'react-native';

import { AvatarViewer3D } from '../components/AvatarViewer3D';
import { Card } from '../components/Card';
import { GradientButton } from '../components/GradientButton';
import { Header } from '../components/Header';
import { PillBadge } from '../components/PillBadge';
import { ProportionSliderRow } from '../components/ProportionSliderRow';
import { ScreenContainer } from '../components/ScreenContainer';
import { BODY_SHAPE_INFO } from '../data/bodyShapes';
import type { RootStackParamList } from '../navigation/types';
import { applyBodyAdjustments, DEFAULT_BODY_ADJUSTMENTS } from '../services/bodyScaling';
import { colors, radii, spacing, typography } from '../theme';
import type { BodyAdjustments, ProportionKey } from '../types';

type Props = NativeStackScreenProps<RootStackParamList, 'FemaleAvatar'>;

const PROPORTION_ROWS: { key: ProportionKey; label: string }[] = [
  { key: 'shoulderWidth', label: 'Shoulders' },
  { key: 'armLength', label: 'Arms' },
  { key: 'legLength', label: 'Legs' },
  { key: 'hipWidth', label: 'Hips / waist' },
];

export function FemaleAvatarScreen({ route, navigation }: Props) {
  const { avatar, avatarConfig, remoteAvatarUrl, remoteTextureUrl } = route.params;
  const shapeInfo = BODY_SHAPE_INFO[avatar.body_shape];
  const { features } = avatarConfig;

  const [adjustments, setAdjustments] = useState<BodyAdjustments>(DEFAULT_BODY_ADJUSTMENTS);
  const adjustedConfig = useMemo(
    () => applyBodyAdjustments(avatarConfig, adjustments),
    [avatarConfig, adjustments]
  );

  // The separate t-shirt model (TSHIRT_ASSET) doesn't fit this body well —
  // its rigid, unskinned mesh was scaled/positioned against a body with a
  // much narrower T-pose arm-spread, so on this body it either falls short
  // of the arm or gaps open at the side (see AvatarViewer3D's
  // fitGarmentToBody comment). Instead, the fabric photo is painted directly
  // onto the body's own torso + sleeve geometry, the same technique already
  // used for bottoms below — see `applyUpperBodyFabric`.
  const [wearTop, setWearTop] = useState(false);
  const [topTextureUri, setTopTextureUri] = useState<string | null>(null);
  const [topError, setTopError] = useState<string | null>(null);

  // Bottoms have no dedicated garment glb either — the fabric photo is
  // painted straight onto the body's own leg geometry. See
  // AvatarViewer3D's `applyLowerBodyFabric`.
  const [wearBottom, setWearBottom] = useState(false);
  const [bottomTextureUri, setBottomTextureUri] = useState<string | null>(null);
  // 0 (short shorts) .. 1 (full-length trousers).
  const [bottomCoverage, setBottomCoverage] = useState(0.5);
  const [bottomError, setBottomError] = useState<string | null>(null);

  async function pickFabricPhoto() {
    setTopError(null);
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setTopError('Photo library permission is needed to choose a fabric photo.');
      return;
    }
    const picked = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.85 });
    if (picked.canceled || picked.assets.length === 0) return;
    setTopTextureUri(picked.assets[0].uri);
  }

  function handleTopFabricStatus(status: 'ready' | 'error', message?: string) {
    setTopError(status === 'error' ? message ?? 'Could not paint this fabric onto your avatar.' : null);
  }

  async function pickBottomFabricPhoto() {
    setBottomError(null);
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setBottomError('Photo library permission is needed to choose a fabric photo.');
      return;
    }
    const picked = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.85 });
    if (picked.canceled || picked.assets.length === 0) return;
    setBottomTextureUri(picked.assets[0].uri);
  }

  function handleBottomFabricStatus(status: 'ready' | 'error', message?: string) {
    setBottomError(status === 'error' ? message ?? 'Could not paint this fabric onto your avatar.' : null);
  }

  return (
    <ScreenContainer>
      <Header title="Your avatar" subtitle="Generated from your photo and measurements." />

      <Card style={styles.avatarCard}>
        <Text style={typography.label}>3D body model</Text>
        <AvatarViewer3D
          key={remoteAvatarUrl ?? 'local'}
          config={adjustedConfig}
          remoteAvatarUrl={remoteAvatarUrl}
          remoteTextureUrl={remoteTextureUrl}
          topFabricTextureUri={wearTop ? topTextureUri : null}
          onTopFabricStatus={handleTopFabricStatus}
          bottomTextureUri={wearBottom ? bottomTextureUri : null}
          bottomCoverage={bottomCoverage}
          onBottomFabricStatus={handleBottomFabricStatus}
        />
        <Text style={[typography.body, styles.spaced]}>Drag to rotate</Text>
      </Card>

      <Card>
        <View style={styles.sliderLabelRow}>
          <View style={styles.flexShrink}>
            <Text style={typography.label}>Top</Text>
            <Text style={[typography.body, styles.helperText]}>
              Apply a fabric photo directly onto your avatar's torso and shoulders.
            </Text>
          </View>
          <Switch value={wearTop} onValueChange={setWearTop} />
        </View>

        {wearTop && (
          <View style={styles.topActionsRow}>
            <GradientButton
              title={topTextureUri ? 'Change fabric photo' : 'Apply fabric photo'}
              variant="accent"
              onPress={pickFabricPhoto}
              style={styles.topActionButton}
            />
            {topTextureUri && (
              <TouchableOpacity activeOpacity={0.85} onPress={() => setTopTextureUri(null)} style={styles.removeFabric}>
                <Ionicons name="close-circle-outline" size={20} color={colors.danger} />
                <Text style={[typography.body, styles.removeFabricText]}>Remove</Text>
              </TouchableOpacity>
            )}
          </View>
        )}

        {topError ? <Text style={styles.error}>{topError}</Text> : null}
      </Card>

      <Card>
        <View style={styles.sliderLabelRow}>
          <View style={styles.flexShrink}>
            <Text style={typography.label}>Bottoms</Text>
            <Text style={[typography.body, styles.helperText]}>
              No separate bottoms model — apply a fabric photo directly onto your avatar's legs.
            </Text>
          </View>
          <Switch value={wearBottom} onValueChange={setWearBottom} />
        </View>

        {wearBottom && (
          <View style={styles.topActionsRow}>
            <GradientButton
              title={bottomTextureUri ? 'Change fabric photo' : 'Apply fabric photo'}
              variant="accent"
              onPress={pickBottomFabricPhoto}
              style={styles.topActionButton}
            />
            {bottomTextureUri && (
              <TouchableOpacity
                activeOpacity={0.85}
                onPress={() => setBottomTextureUri(null)}
                style={styles.removeFabric}
              >
                <Ionicons name="close-circle-outline" size={20} color={colors.danger} />
                <Text style={[typography.body, styles.removeFabricText]}>Remove</Text>
              </TouchableOpacity>
            )}
          </View>
        )}

        {wearBottom && bottomTextureUri && (
          <ProportionSliderRow
            label="Leg length (shorts ↔ trousers)"
            value={bottomCoverage}
            min={0}
            max={1}
            onValueChange={setBottomCoverage}
          />
        )}

        {bottomError ? <Text style={styles.error}>{bottomError}</Text> : null}
      </Card>

      <Card>
        <View style={styles.sliderLabelRow}>
          <Text style={typography.label}>Customize body</Text>
          <TouchableOpacity onPress={() => setAdjustments(DEFAULT_BODY_ADJUSTMENTS)}>
            <Text style={[typography.label, styles.resetLink]}>Reset</Text>
          </TouchableOpacity>
        </View>
        {PROPORTION_ROWS.map(({ key, label }) => (
          <ProportionSliderRow
            key={key}
            label={label}
            value={adjustments.proportionOffsets[key]}
            min={-1}
            max={1}
            onValueChange={(value) =>
              setAdjustments((prev) => ({
                ...prev,
                proportionOffsets: { ...prev.proportionOffsets, [key]: value },
              }))
            }
          />
        ))}
      </Card>

      <Card>
        <Text style={typography.label}>Detected features</Text>
        <View style={[styles.row, styles.spaced]}>
          <PillBadge label={features.gender} color={colors.primary} textColor={colors.surface} />
          <PillBadge label={features.ageGroup} color={colors.primary} textColor={colors.surface} />
          <PillBadge label={features.faceShape} color={colors.primary} textColor={colors.surface} />
        </View>
        <View style={[styles.row, styles.spaced]}>
          <PillBadge label={features.hairStyle} color={colors.accent} textColor={colors.surface} />
          <PillBadge label={features.hairColor} color={colors.accent} textColor={colors.surface} />
          {features.eyeColor ? (
            <PillBadge label={`${features.eyeColor} eyes`} color={colors.accent} textColor={colors.surface} />
          ) : null}
          {features.facialHair && features.facialHair !== 'none' ? (
            <PillBadge label={features.facialHair} color={colors.accent} textColor={colors.surface} />
          ) : null}
        </View>
      </Card>

      <Card>
        <Text style={typography.label}>Body shape</Text>
        <Text style={[typography.heading, styles.spaced]}>{shapeInfo.title}</Text>
        <Text style={typography.body}>{shapeInfo.description}</Text>
        <PillBadge
          label={`${Math.round(avatar.body_shape_confidence * 100)}% match`}
          color={colors.primary}
          textColor={colors.surface}
          style={styles.spaced}
        />
      </Card>

      <Card style={styles.row}>
        <View style={[styles.swatch, { backgroundColor: avatar.skin_tone.hex }]} />
        <View style={styles.flexShrink}>
          <Text style={typography.label}>Skin tone match</Text>
          <Text style={[typography.heading, styles.spaced]}>{avatar.skin_tone.label}</Text>
          <Text style={typography.body}>{avatar.skin_tone.hex.toUpperCase()}</Text>
        </View>
      </Card>

      <GradientButton
        title="Finalize avatar"
        onPress={() =>
          navigation.navigate('FinalizedAvatar', {
            avatar,
            avatarConfig: applyBodyAdjustments(avatarConfig, adjustments),
            remoteAvatarUrl,
            remoteTextureUrl,
          })
        }
      />
      <GradientButton
        title="Start over"
        variant="accent"
        onPress={() => navigation.popToTop()}
      />
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  avatarCard: {
    padding: spacing.sm,
    alignItems: 'center',
  },
  spaced: {
    marginTop: spacing.xs,
    marginBottom: spacing.xs,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: spacing.md,
  },
  swatch: {
    width: 48,
    height: 48,
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  flexShrink: {
    flexShrink: 1,
  },
  sliderLabelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: spacing.md,
  },
  resetLink: {
    color: colors.primary,
  },
  helperText: {
    marginTop: spacing.xs,
  },
  topActionsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    marginTop: spacing.md,
  },
  topActionButton: {
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
  error: {
    color: colors.danger,
    fontWeight: '600',
    textAlign: 'center',
    marginTop: spacing.sm,
  },
});
