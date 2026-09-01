import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { useRef, useState } from 'react';
import { ActivityIndicator, Platform, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { API_BASE_URL, customizeFaceAvatar } from '../api/client';
import { CaptureSilhouette } from '../components/CaptureSilhouette';
import { GradientButton } from '../components/GradientButton';
import { ScreenContainer } from '../components/ScreenContainer';
import type { RootStackParamList } from '../navigation/types';
import * as captureStorage from '../services/captureStorage';
import { ModelUnavailableError, validateCapture } from '../services/captureValidation';
import { analyzeFace } from '../services/faceAnalysis';
import { colors, radii, spacing, typography } from '../theme';
import type { CaptureIssue, CapturePose } from '../types';

type Props = NativeStackScreenProps<RootStackParamList, 'AvatarCreator'>;

const POSES: CapturePose[] = ['front', 'left', 'right'];

const POSE_LABELS: Record<CapturePose, string> = {
  front: 'Front face',
  left: 'Left profile',
  right: 'Right profile',
};

const ISSUE_MESSAGES: Record<CaptureIssue, string> = {
  no_face: 'No face detected — make sure your face is clearly visible.',
  too_dark: 'Too dark — move to a brighter, evenly lit area.',
  too_bright: 'Too bright — reduce strong light or glare on your face.',
  not_centered: 'Center your face inside the outline.',
  blurry: 'That photo is blurry — hold steady and try again.',
};

type Step =
  | { kind: 'capture' }
  | { kind: 'issue'; issues: CaptureIssue[] }
  | { kind: 'model-unavailable' }
  | { kind: 'analyzing' }
  | { kind: 'error'; message: string; retry: () => void };

/**
 * Guided avatar capture: walks the user through front/left/right photos,
 * validates each on-device (face present, lit, centered, sharp), then runs
 * MediaPipe Face Mesh on the front photo. For a male avatar, sends the
 * result to the server's customize-face endpoint to rebuild a detailed
 * personalized body mesh; for female, the server's equivalent asset is a
 * broken placeholder (see `classicAvatarSetup.ts`), so this just refreshes
 * `avatarConfig.faceTextureUri` and leaves the local `RP_BODY_ASSETS` model
 * in place. Hands off to `FacePreview` either way.
 */
export function AvatarCreatorScreen({ route, navigation }: Props) {
  const { avatar, avatarConfig, remoteAvatarUrl: initialRemoteAvatarUrl, remoteTextureUrl: initialRemoteTextureUrl } = route.params;
  const insets = useSafeAreaInsets();
  const [permission, requestPermission] = useCameraPermissions();
  const [permissionHint, setPermissionHint] = useState<string | null>(null);
  const [stepIndex, setStepIndex] = useState(0);
  const [step, setStep] = useState<Step>({ kind: 'capture' });
  const [cameraReady, setCameraReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  const capturesRef = useRef<Partial<Record<CapturePose, string>>>({});
  const pendingUriRef = useRef<string | null>(null);

  const pose = POSES[stepIndex];

  function skip() {
    navigation.replace('FacePreview', { avatar, avatarConfig, remoteAvatarUrl: initialRemoteAvatarUrl, remoteTextureUrl: initialRemoteTextureUrl });
  }

  /**
   * On web, `expo-camera`'s permission request swallows the underlying
   * `getUserMedia` error and just re-reports `denied` with `canAskAgain: true`
   * - so a blocked/missing camera makes this button appear to do nothing.
   * Probe `getUserMedia` directly to surface *why* and what to do about it.
   */
  async function handleRequestPermission() {
    setPermissionHint(null);
    const result = await requestPermission();
    if (result.granted || Platform.OS !== 'web') return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.getTracks().forEach((track) => track.stop());
      await requestPermission();
    } catch (err) {
      const name = err instanceof Error ? err.name : '';
      if (name === 'NotFoundError' || name === 'OverconstrainedError') {
        setPermissionHint('No camera was found on this device. Connect a camera and try again.');
      } else if (name === 'NotReadableError') {
        setPermissionHint('Your camera is already in use by another app. Close it and try again.');
      } else if (name === 'NotAllowedError' || name === 'SecurityError') {
        setPermissionHint(
          'Camera access is blocked for this site. Click the camera icon in your browser\'s address bar, allow access, then tap "Grant camera access" again.'
        );
      } else {
        setPermissionHint('Camera access could not be enabled. Check your browser settings and try again.');
      }
    }
  }

  function retake() {
    pendingUriRef.current = null;
    setStep({ kind: 'capture' });
  }

  async function finishCapture(uri: string) {
    const savedUri = await captureStorage.saveCapture(pose, uri);
    capturesRef.current[pose] = savedUri;
    pendingUriRef.current = null;

    if (stepIndex + 1 < POSES.length) {
      setStepIndex(stepIndex + 1);
      setStep({ kind: 'capture' });
    } else {
      await analyze();
    }
  }

  async function processPickedPhoto(uri: string) {
    pendingUriRef.current = uri;

    let result;
    try {
      result = await validateCapture(uri, pose);
    } catch (err) {
      if (err instanceof ModelUnavailableError) {
        setStep({ kind: 'model-unavailable' });
        return;
      }
      throw err;
    }

    if (result.ok) {
      await finishCapture(uri);
    } else {
      setStep({ kind: 'issue', issues: result.issues });
    }
  }

  async function handleCapture() {
    if (!cameraRef.current || busy) return;
    setBusy(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.8 });
      await processPickedPhoto(photo.uri);
    } catch (err) {
      setStep({
        kind: 'error',
        message: err instanceof Error ? err.message : 'Could not capture a photo.',
        retry: retake,
      });
    } finally {
      setBusy(false);
    }
  }

  /**
   * Camera hardware isn't guaranteed (emulators, desktop browsers, devices
   * without one) - lets the user pick an existing photo for the current pose
   * instead, going through the same on-device validation as a live capture.
   */
  async function handlePickFromGallery() {
    if (busy) return;
    setBusy(true);
    try {
      const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permission.granted) {
        setStep({
          kind: 'error',
          message: 'Photo library permission is needed to choose a photo.',
          retry: retake,
        });
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        quality: 0.8,
      });
      if (result.canceled || result.assets.length === 0) return;

      await processPickedPhoto(result.assets[0].uri);
    } catch (err) {
      setStep({
        kind: 'error',
        message: err instanceof Error ? err.message : 'Could not load that photo.',
        retry: retake,
      });
    } finally {
      setBusy(false);
    }
  }

  async function continueAnyway() {
    const uri = pendingUriRef.current;
    if (!uri) {
      retake();
      return;
    }
    setBusy(true);
    try {
      await finishCapture(uri);
    } catch (err) {
      setStep({
        kind: 'error',
        message: err instanceof Error ? err.message : 'Could not save the captured photo.',
        retry: retake,
      });
    } finally {
      setBusy(false);
    }
  }

  async function analyze() {
    setStep({ kind: 'analyzing' });
    try {
      const frontUri = capturesRef.current.front;
      if (!frontUri) {
        throw new Error('Missing the front-facing photo.');
      }

      // ── Analyze front photo (full feature analysis) ─────────────────
      const { faceTextureUri, faceCustomization } = await analyzeFace(frontUri);

      let remoteAvatarUrl: string | undefined;
      let remoteTextureUrl: string | undefined;

      // The server's customize-face body mesh is only usable for male (see
      // classicAvatarSetup.ts — female's server-side asset is a broken
      // placeholder). `avatarConfig.gender` reflects whatever was chosen in
      // GenderSelectScreen, so this picks up a manual override too.
      if (avatarConfig.gender === 'male') {
        // ── Crop left & right profiles for multi-angle texture ───────
        const leftUri = capturesRef.current.left;
        const rightUri = capturesRef.current.right;

        if (leftUri && rightUri) {
          const { detectAndCropFace } = await import('../services/faceCrop');
          const [leftFaceImage, rightFaceImage] = await Promise.all([
            detectAndCropFace(leftUri),
            detectAndCropFace(rightUri),
          ]);

          if (leftFaceImage && rightFaceImage) {
            // Include side-profile images for multi-angle texture mapping
            faceCustomization.leftFaceImage = leftFaceImage;
            faceCustomization.rightFaceImage = rightFaceImage;
            console.log('[AvatarCreator] Multi-angle: left/right profile crops included');
          } else {
            console.log('[AvatarCreator] Multi-angle: side profile face detection failed, using front only');
          }
        }

        // ── Send to server (handles single or multi-angle automatically) ─
        const { avatar_mesh_url, face_texture_url } = await customizeFaceAvatar(avatar.avatar_id, {
          ...faceCustomization,
          gender: avatarConfig.gender,
        });
        const ts = Date.now();
        remoteAvatarUrl = `${API_BASE_URL}${avatar_mesh_url}?v=${ts}`;
        remoteTextureUrl = face_texture_url ? `${API_BASE_URL}${face_texture_url}?v=${ts}` : undefined;
      }

      navigation.replace('FacePreview', {
        avatar,
        avatarConfig: { ...avatarConfig, faceTextureUri },
        remoteAvatarUrl,
        remoteTextureUrl,
      });
    } catch (err) {
      setStep({
        kind: 'error',
        message: err instanceof Error ? err.message : 'Could not generate your customized avatar.',
        retry: analyze,
      });
    }
  }

  if (!permission) {
    return (
      <ScreenContainer contentStyle={styles.centerContent}>
        <ActivityIndicator color={colors.primary} size="large" />
      </ScreenContainer>
    );
  }

  if (!permission.granted) {
    return (
      <ScreenContainer contentStyle={styles.centerContent}>
        <Text style={[typography.heading, styles.centerText]}>Camera access needed</Text>
        <Text style={[typography.body, styles.centerText]}>
          eWardrobe needs your camera to capture front and profile photos for your avatar.
        </Text>
        {permission.canAskAgain ? (
          <GradientButton title="Grant camera access" onPress={handleRequestPermission} />
        ) : (
          <Text style={[typography.body, styles.centerText, styles.errorText]}>
            Camera access was denied. Enable it in your device settings to use the guided capture.
          </Text>
        )}
        {permissionHint ? (
          <Text style={[typography.body, styles.centerText, styles.errorText]}>{permissionHint}</Text>
        ) : null}
        <GradientButton title="Choose photo from gallery instead" onPress={handlePickFromGallery} loading={busy} />
        <GradientButton title="Skip for now" variant="accent" onPress={skip} />
      </ScreenContainer>
    );
  }

  if (step.kind === 'analyzing') {
    return (
      <ScreenContainer contentStyle={styles.centerContent}>
        <ActivityIndicator color={colors.primary} size="large" />
        <Text style={[typography.heading, styles.centerText]}>Analyzing your photos…</Text>
        <Text style={[typography.body, styles.centerText]}>
          Customizing your avatar's face shape, skin tone, and hair color.
        </Text>
      </ScreenContainer>
    );
  }

  if (step.kind === 'error') {
    return (
      <ScreenContainer contentStyle={styles.centerContent}>
        <Text style={[typography.heading, styles.centerText]}>Something went wrong</Text>
        <Text style={[typography.body, styles.centerText, styles.errorText]}>{step.message}</Text>
        <GradientButton title="Try again" onPress={step.retry} />
        <GradientButton title="Choose photo from gallery instead" onPress={handlePickFromGallery} loading={busy} />
        <GradientButton title="Skip for now" variant="accent" onPress={skip} />
      </ScreenContainer>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="front"
        onCameraReady={() => setCameraReady(true)}
        onMountError={(event) =>
          setStep({ kind: 'error', message: event.message || 'Could not start the camera.', retry: retake })
        }
      />
      <CaptureSilhouette pose={pose} />

      <View style={[styles.topBar, { paddingTop: insets.top + spacing.md }]}>
        <Text style={styles.stepLabel}>
          Step {stepIndex + 1} of {POSES.length} · {POSE_LABELS[pose]}
        </Text>
      </View>

      <View style={[styles.bottomBar, { paddingBottom: insets.bottom + spacing.md }]}>
        {step.kind === 'issue' ? (
          <View style={styles.feedback}>
            {step.issues.map((issue) => (
              <Text key={issue} style={styles.feedbackText}>
                {ISSUE_MESSAGES[issue]}
              </Text>
            ))}
            <GradientButton title="Retake" onPress={retake} />
          </View>
        ) : step.kind === 'model-unavailable' ? (
          <View style={styles.feedback}>
            <Text style={styles.feedbackText}>
              Couldn't verify this photo on-device — the face detection model failed to load. You can retake or
              continue anyway.
            </Text>
            <GradientButton title="Continue anyway" onPress={continueAnyway} loading={busy} />
            <GradientButton title="Retake" variant="accent" onPress={retake} />
          </View>
        ) : (
          <>
            <GradientButton title="Capture" onPress={handleCapture} loading={busy} disabled={!cameraReady} />
            <GradientButton title="Choose from gallery" variant="accent" onPress={handlePickFromGallery} loading={busy} />
          </>
        )}
        <GradientButton title="Skip for now" variant="accent" onPress={skip} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000000' },
  camera: { flex: 1 },
  topBar: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
  },
  stepLabel: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
    backgroundColor: 'rgba(0,0,0,0.45)',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radii.pill,
    overflow: 'hidden',
  },
  bottomBar: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    paddingHorizontal: spacing.lg,
    gap: spacing.sm,
  },
  feedback: {
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: radii.md,
    padding: spacing.md,
    gap: spacing.sm,
  },
  feedbackText: {
    color: '#FFFFFF',
    fontSize: 14,
    textAlign: 'center',
  },
  centerContent: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: spacing.md },
  centerText: { textAlign: 'center' },
  errorText: { color: colors.danger },
});
