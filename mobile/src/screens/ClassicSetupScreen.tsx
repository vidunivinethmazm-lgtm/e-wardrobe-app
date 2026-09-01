import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useCallback, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, TextInput, View } from 'react-native';

import { ApiError } from '../api/client';
import { Card } from '../components/Card';
import { GradientButton } from '../components/GradientButton';
import { Header } from '../components/Header';
import { ScreenContainer } from '../components/ScreenContainer';
import type { RootStackParamList } from '../navigation/types';
import { runClassicAvatarSetup } from '../services/classicAvatarSetup';
import type { Measurements } from '../types';
import { colors, radii, spacing, typography } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'ClassicSetup'>;

type Field = keyof Measurements;

const FIELDS: { key: Field; label: string; placeholder: string }[] = [
  { key: 'bust', label: 'Bust (cm)', placeholder: 'e.g. 92' },
  { key: 'waist', label: 'Waist (cm)', placeholder: 'e.g. 70' },
  { key: 'hips', label: 'Hips (cm)', placeholder: 'e.g. 98' },
  { key: 'height', label: 'Height (cm)', placeholder: 'e.g. 165' },
];

function validate(measurements: Measurements): string | null {
  for (const { key, label } of FIELDS) {
    const raw = measurements[key].trim();
    if (!raw) return `${label} is required.`;
    const value = Number(raw);
    if (!Number.isFinite(value) || value <= 0) return `${label} must be a positive number.`;
  }
  return null;
}

/**
 * Fallback for the AI try-on flow: collects body measurements (used to
 * predict a body shape via `POST /api/predict-body-shape`, see
 * `services/classicAvatarSetup.ts`), then runs the classic avatar pipeline
 * on the photo already collected by `ProfileScreen`, then continues into the
 * classic chain at `EmailScreen`.
 */
export function ClassicSetupScreen({ route, navigation }: Props) {
  const { personPhoto } = route.params;
  const [step, setStep] = useState<'form' | 'running' | 'error'>('form');
  const [measurements, setMeasurements] = useState<Measurements>({
    bust: '',
    waist: '',
    hips: '',
    height: '',
  });
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (values: Measurements) => {
    setStep('running');
    setError(null);
    try {
      const { avatar, avatarConfig, remoteAvatarUrl, remoteTextureUrl } = await runClassicAvatarSetup(personPhoto, values);
      // Go through GenderSelect so the user can confirm their gender, then
      // AvatarCreator for a dedicated face selfie — both are needed for the
      // face to be applied correctly to the right body mesh.
      navigation.replace('GenderSelect', { avatar, avatarConfig, remoteAvatarUrl, remoteTextureUrl });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not reach the avatar service. Please try again.');
      setStep('error');
    }
  }, [personPhoto, navigation]);

  function handleSubmit() {
    const validationError = validate(measurements);
    if (validationError) {
      setError(validationError);
      return;
    }
    setError(null);
    run(measurements);
  }

  if (step === 'form' || step === 'error') {
    return (
      <ScreenContainer>
        <Header
          title="A few measurements"
          subtitle="AI try-on wasn't available, so we're building your avatar the classic way - this helps us pick your body shape."
        />

        <Card>
          {FIELDS.map(({ key, label, placeholder }) => (
            <View key={key} style={styles.field}>
              <Text style={typography.body}>{label}</Text>
              <TextInput
                style={styles.input}
                value={measurements[key]}
                onChangeText={(text) => setMeasurements((prev) => ({ ...prev, [key]: text }))}
                placeholder={placeholder}
                placeholderTextColor={colors.textMuted}
                keyboardType="numeric"
              />
            </View>
          ))}
        </Card>

        {error ? <Text style={styles.error}>{error}</Text> : null}

        <GradientButton title="Continue" onPress={handleSubmit} />
        <GradientButton title="Start over" variant="accent" onPress={() => navigation.popToTop()} />
      </ScreenContainer>
    );
  }

  return (
    <ScreenContainer>
      <Header title="Setting up your avatar" subtitle="Creating your avatar..." />

      <Card style={styles.card}>
        <ActivityIndicator color={colors.primary} />
        <Text style={[typography.body, styles.message]}>Creating your avatar...</Text>
      </Card>

      <GradientButton title="Start over" variant="accent" onPress={() => navigation.popToTop()} />
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  card: {
    alignItems: 'center',
    gap: spacing.md,
    paddingVertical: spacing.lg,
  },
  message: {
    textAlign: 'center',
  },
  field: {
    marginBottom: spacing.md,
    gap: spacing.xs,
  },
  input: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    fontSize: 16,
    color: colors.text,
    backgroundColor: colors.background,
  },
  error: {
    color: colors.danger,
    fontWeight: '600',
    textAlign: 'center',
  },
});
