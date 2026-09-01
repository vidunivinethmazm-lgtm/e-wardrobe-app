import { PropsWithChildren } from 'react';
import { StyleSheet, View, ViewStyle } from 'react-native';

import { colors, radii, shadow, spacing } from '../theme';

interface Props {
  style?: ViewStyle;
}

export function Card({ children, style }: PropsWithChildren<Props>) {
  return <View style={[styles.card, style]}>{children}</View>;
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    padding: spacing.md,
    ...shadow,
  },
});
