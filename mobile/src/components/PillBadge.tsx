import { StyleSheet, Text, View, ViewStyle } from 'react-native';

import { colors, radii, spacing, typography } from '../theme';

interface Props {
  label: string;
  color?: string;
  textColor?: string;
  style?: ViewStyle;
}

export function PillBadge({ label, color = colors.border, textColor = colors.text, style }: Props) {
  return (
    <View style={[styles.pill, { backgroundColor: color }, style]}>
      <Text style={[typography.label, { color: textColor }]}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  pill: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radii.pill,
    alignSelf: 'flex-start',
  },
});
