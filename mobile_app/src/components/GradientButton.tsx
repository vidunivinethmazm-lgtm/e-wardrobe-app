import { LinearGradient } from 'expo-linear-gradient';
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, ViewStyle } from 'react-native';

import { colors, gradients, radii, spacing, typography } from '../theme';

interface Props {
  title: string;
  onPress: () => void;
  disabled?: boolean;
  loading?: boolean;
  variant?: 'primary' | 'accent';
  style?: ViewStyle;
}

export function GradientButton({ title, onPress, disabled, loading, variant = 'primary', style }: Props) {
  const isDisabled = disabled || loading;
  const gradientColors = variant === 'primary' ? gradients.primary : gradients.accent;

  return (
    <TouchableOpacity
      activeOpacity={0.85}
      onPress={onPress}
      disabled={isDisabled}
      style={[styles.wrapper, style, isDisabled && styles.disabled]}
    >
      <LinearGradient
        colors={gradientColors}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.gradient}
      >
        {loading ? <ActivityIndicator color={colors.surface} /> : <Text style={typography.button}>{title}</Text>}
      </LinearGradient>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    borderRadius: radii.pill,
    overflow: 'hidden',
  },
  gradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radii.pill,
  },
  disabled: {
    opacity: 0.6,
  },
});
