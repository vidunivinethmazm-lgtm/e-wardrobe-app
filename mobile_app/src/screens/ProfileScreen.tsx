import { Ionicons } from '@expo/vector-icons';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import * as ImagePicker from 'expo-image-picker';
import { useEffect, useState } from 'react';
import { Image, StyleSheet, Text, View } from 'react-native';

import { Card } from '../components/Card';
import { GradientButton } from '../components/GradientButton';
import { Header } from '../components/Header';
import { ScreenContainer } from '../components/ScreenContainer';
import type { RootStackParamList } from '../navigation/types';
import { getDetector } from '../services/faceDetector';
import { colors, radii, spacing, typography } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'Profile'>;

const PHOTO_PICKER_OPTIONS: Partial<ImagePicker.ImagePickerOptions> = {
  mediaTypes: ['images'],
  allowsEditing: true,
  aspect: [3, 4],
  quality: 0.8,
};

export function ProfileScreen({ navigation }: Props) {
  const [photo, setPhoto] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Starts the (tens-of-MB) face detection model downloading as soon as the
  // app opens, so it's likely already cached by the time the user submits a
  // photo for analysis further down the flow.
  useEffect(() => {
    getDetector().catch(() => {});
  }, []);

  async function pickFromGallery() {
    setError(null);
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setError('Photo library permission is needed to choose a photo.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync(PHOTO_PICKER_OPTIONS);

    if (!result.canceled && result.assets.length > 0) {
      setPhoto(result.assets[0]);
    }
  }

  async function takePhoto() {
    setError(null);
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      setError('Camera permission is needed to take a photo.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync(PHOTO_PICKER_OPTIONS);

    if (!result.canceled && result.assets.length > 0) {
      setPhoto(result.assets[0]);
    }
  }

  function handleContinue() {
    setError(null);

    if (!photo) {
      setError('Please choose a full-body photo, or take one with the camera.');
      return;
    }

    navigation.navigate('ClassicSetup', {
      personPhoto: {
        uri: photo.uri,
        name: photo.fileName ?? 'photo.jpg',
        type: photo.mimeType ?? 'image/jpeg',
      },
    });
  }

  return (
    <ScreenContainer>
      <Header title="eWardrobe" subtitle="Add your photo, then enter your measurements to build your 3D avatar." />

      <Card>
        <Text style={typography.heading}>Your photo</Text>
        <Text style={[typography.body, styles.helperText]}>
          A clear, well-lit, full-body photo gives the best result. Choose one from your gallery, or take a new
          one — you only need one photo.
        </Text>

        <View style={styles.photoRow}>
          {photo ? (
            <Image source={{ uri: photo.uri }} style={styles.photoPreview} />
          ) : (
            <View style={[styles.photoPreview, styles.photoPlaceholder]}>
              <Ionicons name="person-outline" size={40} color={colors.textMuted} />
            </View>
          )}

          <View style={styles.photoButtons}>
            <GradientButton
              title={photo ? 'Choose a different photo' : 'Choose photo'}
              onPress={pickFromGallery}
              variant="accent"
            />
            <GradientButton title="Take photo" onPress={takePhoto} variant="accent" />
          </View>
        </View>
      </Card>

      {error ? <Text style={styles.error}>{error}</Text> : null}

      <GradientButton title="Continue" onPress={handleContinue} />
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  helperText: {
    marginTop: spacing.xs,
    marginBottom: spacing.md,
  },
  photoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  photoPreview: {
    width: 88,
    height: 110,
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
  photoButtons: {
    flex: 1,
    gap: spacing.sm,
  },
  error: {
    color: colors.danger,
    fontWeight: '600',
    textAlign: 'center',
  },
});
