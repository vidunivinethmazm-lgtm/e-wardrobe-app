import { LinearGradient } from 'expo-linear-gradient';
import { StyleSheet, Text } from 'react-native';

import { gradients, radii, spacing, typography } from '../theme';

interface Props {
  title: string;
  subtitle?: string;
}

export function Header({ title, subtitle }: Props) {
  return (
    <LinearGradient colors={gradients.hero} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={styles.header}>
      <Text style={styles.title}>{title}</Text>
      {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  header: {
    borderRadius: radii.lg,
    padding: spacing.lg,
    gap: spacing.xs,
  },
  title: {
    ...typography.title,
    color: '#FFFFFF',
  },
  subtitle: {
    ...typography.subtitle,
    color: 'rgba(255,255,255,0.85)',
  },
});
