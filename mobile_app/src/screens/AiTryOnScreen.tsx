import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Image, StyleSheet, Text, View } from 'react-native';

import { ApiError, API_BASE_URL, createAiTryOn, getAiTryOnConfig, getAiTryOnImageUrl } from '../api/client';
import { Card } from '../components/Card';
import { DebugPanel, type DebugRow } from '../components/DebugPanel';
import { GradientButton } from '../components/GradientButton';
import { Header } from '../components/Header';
import { ScreenContainer } from '../components/ScreenContainer';
import type { RootStackParamList } from '../navigation/types';
import type { AiTryOnConfig, AiTryOnResponse } from '../types';
import { colors, radii, spacing, typography } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'AiTryOn'>;

type Status = 'generating' | 'ready' | 'error';

/**
 * First step of the AI try-on pipeline: `POST /api/ai-tryon` (person +
 * clothing photos -> 2D try-on image). Shows the generated image, then lets
 * the user continue to the 3D avatar step, retry, start over, or fall back
 * to the classic setup. Does not auto-run the 3D step or auto-fallback on
 * error - Gemini/Meshy can be slow, so errors are shown with retry options
 * instead of silently switching flows.
 */
export function AiTryOnScreen({ route, navigation }: Props) {
  const { personPhoto, clothingPhotos } = route.params;
  const [status, setStatus] = useState<Status>('generating');
  const [tryon, setTryon] = useState<AiTryOnResponse | null>(null);
  const [generatedImageUrl, setGeneratedImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<AiTryOnConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const cancelledRef = useRef(false);

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

    // Fail fast with a clear message instead of uploading photos just to
    // have the backend reject them with the same underlying cause.
    if (cfg && !cfg.ai_tryon_mock && !cfg.gemini_api_key_present) {
      setError(
        'AI_TRYON_MOCK=0 but GEMINI_API_KEY is not set on the server. Set GEMINI_API_KEY (or AI_TRYON_MOCK=1) and restart the backend.'
      );
      setStatus('error');
      return;
    }

    try {
      const result = await createAiTryOn(personPhoto, clothingPhotos);
      if (cancelledRef.current) return;
      setTryon(result);
      setGeneratedImageUrl(`${getAiTryOnImageUrl(result)}?v=${Date.now()}`);
      setStatus('ready');
    } catch (err) {
      if (cancelledRef.current) return;
      setError(err instanceof ApiError ? err.message : 'Something went wrong generating your try-on image.');
      setStatus('error');
    }
  }, [personPhoto, clothingPhotos, config]);

  useEffect(() => {
    cancelledRef.current = false;
    generate();
    return () => {
      cancelledRef.current = true;
    };
    // personPhoto/clothingPhotos are fixed for the lifetime of this screen.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleCreate3D() {
    if (!tryon || !generatedImageUrl) return;
    navigation.navigate('AiAvatar', {
      tryonId: tryon.tryon_id,
      generatedImageUrl,
      personPhoto,
    });
  }

  function handleStartOver() {
    navigation.popToTop();
  }

  function handleUseClassic() {
    navigation.replace('ClassicSetup', { personPhoto });
  }

  const debugRows: DebugRow[] = [
    { label: 'API_BASE_URL', value: API_BASE_URL },
    {
      label: 'AI_TRYON_MOCK',
      value: config ? String(config.ai_tryon_mock) : configError ? `error: ${configError}` : 'loading...',
    },
    { label: 'GEMINI_MODEL', value: config?.gemini_model ?? '—' },
    { label: 'GEMINI_API_KEY present', value: config ? String(config.gemini_api_key_present) : '—' },
    { label: 'IMAGE_TO_3D_PROVIDER', value: config?.image_to_3d_provider ?? '—' },
    { label: 'tryon_id', value: tryon?.tryon_id ?? '—' },
    { label: 'generated_image_url', value: tryon?.generated_image_url ?? '—' },
    { label: 'avatar_mesh_url', value: '—' },
    { label: 'error', value: error ?? '—' },
  ];

  return (
    <ScreenContainer>
      <Header title="Creating your avatar" subtitle="This can take a moment - please don't go back." />

      {config?.ai_tryon_mock ? (
        <View style={styles.banner}>
          <Text style={[typography.body, styles.bannerText]}>Mock mode: Gemini is not being called</Text>
        </View>
      ) : null}

      <Card style={styles.card}>
        {generatedImageUrl ? (
          <Image source={{ uri: generatedImageUrl }} style={styles.preview} resizeMode="contain" />
        ) : null}

        {status === 'generating' ? (
          <>
            <ActivityIndicator color={colors.primary} style={styles.spinner} />
            <Text style={[typography.body, styles.message]}>Generating your try-on image...</Text>
          </>
        ) : null}

        {status === 'error' ? <Text style={[typography.body, styles.error]}>{error}</Text> : null}
      </Card>

      {status === 'ready' ? (
        <>
          <GradientButton title="Create 3D avatar" onPress={handleCreate3D} />
          <GradientButton title="Start over" variant="accent" onPress={handleStartOver} />
          <GradientButton title="Use classic setup" variant="accent" onPress={handleUseClassic} />
        </>
      ) : null}

      {status === 'error' ? (
        <>
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
  card: {
    alignItems: 'center',
    gap: spacing.md,
    paddingVertical: spacing.lg,
  },
  preview: {
    width: '100%',
    aspectRatio: 3 / 4,
    borderRadius: radii.md,
    backgroundColor: colors.background,
  },
  spinner: {
    marginTop: spacing.sm,
  },
  message: {
    textAlign: 'center',
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
