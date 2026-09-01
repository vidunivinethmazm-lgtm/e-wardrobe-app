import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ActivityIndicator, Image, StyleSheet, Text, View } from 'react-native';

import { ApiError, API_BASE_URL, createAiAvatar3D, getAiTryOnConfig } from '../api/client';
import { AvatarViewer3D } from '../components/AvatarViewer3D';
import { Card } from '../components/Card';
import { DebugPanel, type DebugRow } from '../components/DebugPanel';
import { GradientButton } from '../components/GradientButton';
import { Header } from '../components/Header';
import { PillBadge } from '../components/PillBadge';
import { ScreenContainer } from '../components/ScreenContainer';
import type { RootStackParamList } from '../navigation/types';
import { buildAvatar, DEFAULT_MEASUREMENTS } from '../services/avatarBuilder';
import { DEFAULT_AVATAR_FEATURES } from '../services/faceAnalysis';
import type { AiTryOnConfig } from '../types';
import { colors, radii, spacing, typography } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'AiAvatar'>;

type Status = 'generating' | 'ready' | 'error';

/**
 * Second step of the AI try-on pipeline: `POST /api/ai-tryon/<id>/avatar3d`
 * (2D try-on image -> `.glb` avatar), shown alongside the 2D image from
 * `AiTryOnScreen` for comparison. Errors are shown with retry/fallback
 * options rather than silently switching flows - image-to-3D providers can
 * be slow or fail.
 */
export function AiAvatarScreen({ route, navigation }: Props) {
  const { tryonId, generatedImageUrl, personPhoto } = route.params;
  const [status, setStatus] = useState<Status>('generating');
  const [avatarMeshUrl, setAvatarMeshUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<AiTryOnConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const cancelledRef = useRef(false);

  // AvatarViewer3D's `config` prop is unused whenever `remoteAvatarUrl` is
  // set, but is still required - pass a harmless default.
  const placeholderConfig = useMemo(
    () => buildAvatar(DEFAULT_AVATAR_FEATURES, null, DEFAULT_MEASUREMENTS),
    []
  );

  const generate = useCallback(async () => {
    setStatus('generating');
    setError(null);

    let cfg = config;
    if (!cfg) {
      try {
        cfg = await getAiTryOnConfig();
        if (cancelledRef.current) return;
        setConfig(cfg);
        setConfigError(null);
      } catch (err) {
        if (cancelledRef.current) return;
        setConfigError(err instanceof ApiError ? err.message : 'Could not load AI try-on config from the backend.');
      }
    }

    // Fail fast with a clear message instead of waiting on a job that the
    // backend will reject for lack of an API key anyway.
    if (cfg && cfg.image_to_3d_provider === 'meshy' && !cfg.image_to_3d_api_key_present) {
      setError(
        'IMAGE_TO_3D_PROVIDER=meshy but IMAGE_TO_3D_API_KEY is not set on the server. Set IMAGE_TO_3D_API_KEY (or IMAGE_TO_3D_PROVIDER=mock) and restart the backend.'
      );
      setStatus('error');
      return;
    }

    try {
      const { avatar_mesh_url } = await createAiAvatar3D(tryonId);
      if (cancelledRef.current) return;
      setAvatarMeshUrl(`${API_BASE_URL}${avatar_mesh_url}?v=${Date.now()}`);
      setStatus('ready');
    } catch (err) {
      if (cancelledRef.current) return;
      setError(err instanceof ApiError ? err.message : 'Something went wrong building your 3D avatar.');
      setStatus('error');
    }
  }, [tryonId, config]);

  useEffect(() => {
    cancelledRef.current = false;
    generate();
    return () => {
      cancelledRef.current = true;
    };
    // tryonId is fixed for the lifetime of this screen.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleStartOver() {
    navigation.popToTop();
  }

  function handleUseClassic() {
    navigation.replace('ClassicSetup', { personPhoto });
  }

  const isMock3D = config?.image_to_3d_provider === 'mock';

  const debugRows: DebugRow[] = [
    { label: 'API_BASE_URL', value: API_BASE_URL },
    {
      label: 'AI_TRYON_MOCK',
      value: config ? String(config.ai_tryon_mock) : configError ? `error: ${configError}` : 'loading...',
    },
    { label: 'IMAGE_TO_3D_PROVIDER', value: config?.image_to_3d_provider ?? '—' },
    { label: 'IMAGE_TO_3D_API_KEY present', value: config ? String(config.image_to_3d_api_key_present) : '—' },
    { label: 'IMAGE_TO_3D_TIMEOUT_S', value: config ? String(config.image_to_3d_timeout_s) : '—' },
    { label: 'tryon_id', value: tryonId },
    { label: 'generated_image_url', value: generatedImageUrl },
    { label: 'avatar_mesh_url', value: avatarMeshUrl ?? '—' },
    { label: 'error', value: error ?? '—' },
  ];

  return (
    <ScreenContainer>
      <Header title="Your AI avatar" subtitle="Here's your avatar wearing the outfit you chose." />

      {isMock3D ? (
        <View style={styles.banner}>
          <Text style={[typography.body, styles.bannerText]}>Mock 3D provider: placeholder GLB</Text>
        </View>
      ) : null}

      <Card style={styles.previewCard}>
        <Text style={typography.label}>Generated outfit</Text>
        <Image source={{ uri: generatedImageUrl }} style={styles.preview} resizeMode="contain" />
      </Card>

      {status === 'generating' ? (
        <Card style={styles.avatarCard}>
          <ActivityIndicator color={colors.primary} style={styles.spinner} />
          <Text style={[typography.body, styles.message]}>Building your 3D avatar...</Text>
        </Card>
      ) : null}

      {status === 'ready' && avatarMeshUrl ? (
        <>
          {isMock3D ? (
            <PillBadge label="Mock 3D provider: placeholder GLB" color={colors.accent} textColor={colors.surface} />
          ) : (
            <PillBadge label="AI avatar ready" color={colors.primary} textColor={colors.surface} />
          )}
          <Card style={styles.avatarCard}>
            <Text style={typography.label}>Your avatar</Text>
            <AvatarViewer3D config={placeholderConfig} remoteAvatarUrl={avatarMeshUrl} />
            <Text style={[typography.body, styles.spaced]}>Drag to rotate</Text>
          </Card>
          <GradientButton title="Start over" variant="accent" onPress={handleStartOver} />
        </>
      ) : null}

      {status === 'error' ? (
        <>
          <Text style={[typography.body, styles.error]}>{error}</Text>
          <GradientButton title="Try again" onPress={generate} />
          <GradientButton title="Use classic setup" variant="accent" onPress={handleUseClassic} />
          <GradientButton title="Start over" variant="accent" onPress={handleStartOver} />
        </>
      ) : null}

      <DebugPanel rows={debugRows} />
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  previewCard: {
    alignItems: 'center',
    gap: spacing.sm,
  },
  preview: {
    width: '100%',
    aspectRatio: 3 / 4,
    borderRadius: radii.md,
    backgroundColor: colors.background,
  },
  avatarCard: {
    padding: spacing.sm,
    alignItems: 'center',
    gap: spacing.sm,
  },
  spinner: {
    marginTop: spacing.sm,
  },
  message: {
    textAlign: 'center',
  },
  spaced: {
    marginTop: spacing.xs,
    marginBottom: spacing.xs,
  },
  error: {
    color: colors.danger,
    fontWeight: '600',
    textAlign: 'center',
  },
  banner: {
    backgroundColor: colors.border,
    borderRadius: radii.md,
    padding: spacing.sm,
  },
  bannerText: {
    textAlign: 'center',
    fontWeight: '600',
  },
});
