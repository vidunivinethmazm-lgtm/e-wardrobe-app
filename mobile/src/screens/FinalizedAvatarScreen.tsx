import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { Image, StyleSheet, Text } from 'react-native';

import { AvatarViewer3D } from '../components/AvatarViewer3D';
import { Card } from '../components/Card';
import { GradientButton } from '../components/GradientButton';
import { Header } from '../components/Header';
import { PillBadge } from '../components/PillBadge';
import { ScreenContainer } from '../components/ScreenContainer';
import type { RootStackParamList } from '../navigation/types';
import { colors, radii, spacing, typography } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'FinalizedAvatar'>;

export function FinalizedAvatarScreen({ route, navigation }: Props) {
  const { avatar, avatarConfig, remoteAvatarUrl, remoteTextureUrl } = route.params;

  return (
    <ScreenContainer>
      <Header title="Finalized avatar" subtitle="Your body size choices are locked in." />

      <PillBadge label="Avatar finalized" color={colors.primary} textColor={colors.surface} />

      <Card style={styles.avatarCard}>
        <Text style={typography.label}>Your finalized avatar</Text>
        <AvatarViewer3D config={avatarConfig} remoteAvatarUrl={remoteAvatarUrl} remoteTextureUrl={remoteTextureUrl} />
        <Text style={[typography.body, styles.spaced]}>Drag to rotate</Text>
      </Card>

      {avatarConfig.faceTextureUri && (
        <Card style={styles.avatarCard}>
          <Text style={typography.label}>Avatar face</Text>
          <Image source={{ uri: avatarConfig.faceTextureUri }} style={[styles.faceImage, styles.spaced]} />
        </Card>
      )}

      <GradientButton title="Edit body" variant="accent" onPress={() => navigation.goBack()} />
      <GradientButton
        title="Next"
        onPress={() => navigation.navigate('Wardrobe', { avatar, avatarConfig, remoteAvatarUrl, remoteTextureUrl })}
      />
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  avatarCard: {
    padding: spacing.sm,
    alignItems: 'center',
  },
  faceImage: {
    width: 128,
    height: 128,
    borderRadius: radii.md,
    backgroundColor: colors.background,
  },
  spaced: {
    marginTop: spacing.xs,
    marginBottom: spacing.xs,
  },
});
