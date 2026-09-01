import { Ionicons } from '@expo/vector-icons';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useState } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';

import { Card } from '../components/Card';
import { GradientButton } from '../components/GradientButton';
import { Header } from '../components/Header';
import { ScreenContainer } from '../components/ScreenContainer';
import type { RootStackParamList } from '../navigation/types';
import { DEFAULT_MEASUREMENTS, RP_BODY_ASSETS } from '../services/avatarBuilder';
import { computeBodyScale } from '../services/bodyScaling';
import { colors, radii, spacing, typography } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'GenderSelect'>;

const OPTIONS: { value: 'female' | 'male'; label: string; icon: keyof typeof Ionicons.glyphMap }[] = [
  { value: 'female', label: 'Female', icon: 'female' },
  { value: 'male', label: 'Male', icon: 'male' },
];

export function GenderSelectScreen({ route, navigation }: Props) {
  const { avatar, avatarConfig, remoteAvatarUrl, remoteTextureUrl } = route.params;
  const [gender, setGender] = useState<'female' | 'male'>(avatarConfig.gender);

  function handleContinue() {
    const updatedConfig = {
      ...avatarConfig,
      gender,
      bodyAsset: RP_BODY_ASSETS[gender],
      features: { ...avatarConfig.features, gender },
      bodyScale: computeBodyScale(DEFAULT_MEASUREMENTS, gender, avatarConfig.features.faceShape),
    };

    // `remoteAvatarUrl`/`remoteTextureUrl` (if set) are a server-built mesh
    // baked for `avatarConfig.gender` — the gender auto-detected from the
    // user's photo, BEFORE this screen ever ran (see
    // `classicAvatarSetup.runClassicAvatarSetup`). Re-baking it for a
    // different gender needs the original face-feature payload, which isn't
    // available this far down the nav chain, so if the user overrides
    // gender here, drop the now-mismatched remote mesh entirely — every
    // downstream screen falls back to the local, correctly-gendered
    // `RP_BODY_ASSETS` model instead of silently showing the wrong body.
    const genderChanged = gender !== avatarConfig.gender;

    navigation.navigate('AvatarCreator', {
      avatar,
      avatarConfig: updatedConfig,
      remoteAvatarUrl: genderChanged ? undefined : remoteAvatarUrl,
      remoteTextureUrl: genderChanged ? undefined : remoteTextureUrl,
    });
  }

  return (
    <ScreenContainer>
      <Header title="Choose your avatar" subtitle="Pick the avatar style you'd like to build." />

      <Card>
        <Text style={typography.heading}>Avatar gender</Text>
        <Text style={[typography.body, styles.helperText]}>
          This decides which body model and avatar page we'll build for you.
        </Text>

        <View style={styles.row}>
          {OPTIONS.map((option) => {
            const selected = gender === option.value;
            return (
              <TouchableOpacity
                key={option.value}
                style={[styles.option, selected && styles.optionSelected]}
                onPress={() => setGender(option.value)}
              >
                <Ionicons name={option.icon} size={32} color={selected ? colors.surface : colors.primary} />
                <Text style={[typography.body, styles.optionLabel, selected && styles.optionLabelSelected]}>
                  {option.label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </Card>

      <GradientButton title="Continue" onPress={handleContinue} />
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  helperText: {
    marginTop: spacing.xs,
    marginBottom: spacing.md,
  },
  row: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  option: {
    flex: 1,
    alignItems: 'center',
    gap: spacing.xs,
    paddingVertical: spacing.lg,
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  optionSelected: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  optionLabel: {
    fontWeight: '700',
  },
  optionLabelSelected: {
    color: colors.surface,
  },
});
