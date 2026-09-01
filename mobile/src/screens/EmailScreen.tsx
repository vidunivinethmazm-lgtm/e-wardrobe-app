import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useState } from 'react';
import { StyleSheet, Text, TextInput } from 'react-native';

import { ApiError, submitEmail } from '../api/client';
import { Card } from '../components/Card';
import { GradientButton } from '../components/GradientButton';
import { Header } from '../components/Header';
import { ScreenContainer } from '../components/ScreenContainer';
import type { RootStackParamList } from '../navigation/types';
import { colors, radii, spacing, typography } from '../theme';

type Props = NativeStackScreenProps<RootStackParamList, 'Email'>;

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function EmailScreen({ route, navigation }: Props) {
  const { avatar, avatarConfig, remoteAvatarUrl, remoteTextureUrl } = route.params;
  const [email, setEmail] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleContinue() {
    setError(null);

    const trimmed = email.trim();
    if (!EMAIL_PATTERN.test(trimmed)) {
      setError('Please enter a valid email address.');
      return;
    }

    setLoading(true);
    try {
      await submitEmail(trimmed);
      navigation.navigate('GenderSelect', { avatar, avatarConfig, remoteAvatarUrl, remoteTextureUrl });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not reach the server. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <ScreenContainer>
      <Header title="Almost there" subtitle="Enter your email to continue building your avatar." />

      <Card>
        <Text style={typography.heading}>Your email</Text>
        <Text style={[typography.body, styles.helperText]}>We'll use this to save your avatar profile.</Text>

        <TextInput
          style={styles.input}
          value={email}
          onChangeText={setEmail}
          placeholder="you@example.com"
          placeholderTextColor={colors.textMuted}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="email-address"
          textContentType="emailAddress"
        />
      </Card>

      {error ? <Text style={styles.error}>{error}</Text> : null}

      <GradientButton title="Continue" onPress={handleContinue} loading={loading} />
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  helperText: {
    marginTop: spacing.xs,
    marginBottom: spacing.md,
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
