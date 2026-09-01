import { Ionicons } from '@expo/vector-icons';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import * as ImagePicker from 'expo-image-picker';
import { useState } from 'react';
import { Image, ScrollView, StyleSheet, Text, View } from 'react-native';

import { Card } from '../components/Card';
import { GradientButton } from '../components/GradientButton';
import { Header } from '../components/Header';
import { ScreenContainer } from '../components/ScreenContainer';
import type { RootStackParamList } from '../navigation/types';
import { colors, radii, spacing, typography } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'DressPhoto'>;

const PHOTO_PICKER_OPTIONS: Partial<ImagePicker.ImagePickerOptions> = {
  mediaTypes: ['images'],
  aspect: [1, 1],
  quality: 0.8,
};

const GALLERY_PICKER_OPTIONS: Partial<ImagePicker.ImagePickerOptions> = {
  ...PHOTO_PICKER_OPTIONS,
  allowsMultipleSelection: true,
  selectionLimit: 6,
};

const CAMERA_PICKER_OPTIONS: Partial<ImagePicker.ImagePickerOptions> = {
  ...PHOTO_PICKER_OPTIONS,
  allowsEditing: true,
};

/**
 * Second step of the AI try-on flow: collects one or more clothing photos.
 * Gallery picks can include several images; camera capture appends one image
 * at a time.
 */
export function DressPhotoScreen({ route, navigation }: Props) {
  const { personPhoto } = route.params;
  const [photos, setPhotos] = useState<ImagePicker.ImagePickerAsset[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function pickFromGallery() {
    setError(null);
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setError('Photo library permission is needed to choose a photo.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync(GALLERY_PICKER_OPTIONS);

    if (!result.canceled && result.assets.length > 0) {
      setPhotos(result.assets);
    }
  }

  /** Fallback for platforms/galleries where multi-select isn't available:
   * appends newly picked photo(s) to the existing selection instead of
   * replacing it. */
  async function addFromGallery() {
    setError(null);
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setError('Photo library permission is needed to choose a photo.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync(GALLERY_PICKER_OPTIONS);

    if (!result.canceled && result.assets.length > 0) {
      setPhotos((prev) => [...prev, ...result.assets]);
    }
  }

  async function takePhoto() {
    setError(null);
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      setError('Camera permission is needed to take a photo.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync(CAMERA_PICKER_OPTIONS);

    if (!result.canceled && result.assets.length > 0) {
      setPhotos((prev) => [...prev, result.assets[0]]);
    }
  }

  function handleContinue() {
    setError(null);

    if (photos.length === 0) {
      setError('Please choose at least one clothing photo, or take one with the camera.');
      return;
    }

    navigation.navigate('AiTryOn', {
      personPhoto,
      clothingPhotos: photos.map((photo, index) => ({
        uri: photo.uri,
        name: photo.fileName ?? `clothing-${index + 1}.jpg`,
        type: photo.mimeType ?? 'image/jpeg',
      })),
    });
  }

  return (
    <ScreenContainer>
      <Header title="Add clothing" subtitle="Choose one or more clothing photos for your AI avatar." />

      <Card>
        <Text style={typography.heading}>Clothing photo</Text>
        <Text style={[typography.body, styles.helperText]}>
          Clear photos of each item, laid flat or on a hanger, work best.
        </Text>

        <View style={styles.photoRow}>
          {photos.length > 0 ? (
            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.previewList}>
              {photos.map((item, index) => (
                <Image key={`${item.uri}-${index}`} source={{ uri: item.uri }} style={styles.photoPreview} />
              ))}
            </ScrollView>
          ) : (
            <View style={[styles.photoPreview, styles.photoPlaceholder]}>
              <Ionicons name="shirt-outline" size={40} color={colors.textMuted} />
            </View>
          )}

          <View style={styles.photoButtons}>
            <GradientButton
              title={photos.length > 0 ? 'Choose different photos' : 'Choose photos'}
              onPress={pickFromGallery}
              variant="accent"
            />
            {photos.length > 0 ? (
              <GradientButton title="Add another clothing photo" onPress={addFromGallery} variant="accent" />
            ) : null}
            <GradientButton title={photos.length > 0 ? 'Add with camera' : 'Take photo'} onPress={takePhoto} variant="accent" />
          </View>
        </View>

        {photos.length > 0 ? (
          <Text style={[typography.body, styles.countText]}>
            {photos.length} clothing photo{photos.length === 1 ? '' : 's'} selected
          </Text>
        ) : null}
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
  countText: {
    marginTop: spacing.sm,
    color: colors.textMuted,
  },
  photoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  photoPreview: {
    width: 96,
    height: 96,
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
  previewList: {
    gap: spacing.sm,
  },
  error: {
    color: colors.danger,
    fontWeight: '600',
    textAlign: 'center',
  },
});
