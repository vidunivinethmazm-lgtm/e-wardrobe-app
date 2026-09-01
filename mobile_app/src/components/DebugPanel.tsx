import { StyleSheet, Text, View } from 'react-native';

import { Card } from './Card';
import { colors, spacing, typography } from '../theme';

export interface DebugRow {
  label: string;
  value: string;
}

interface Props {
  rows: DebugRow[];
}

/**
 * Always-visible label/value panel for the AI try-on screens - shows the
 * resolved API_BASE_URL, backend mock/provider config, session ids/urls, and
 * the last error, so mock output or stale bundles aren't mistaken for a real
 * generation result.
 */
export function DebugPanel({ rows }: Props) {
  return (
    <Card style={styles.card}>
      <Text style={typography.label}>Debug info</Text>
      {rows.map((row) => (
        <View key={row.label} style={styles.row}>
          <Text style={[typography.body, styles.label]}>{row.label}</Text>
          <Text style={[typography.body, styles.value]} selectable>
            {row.value}
          </Text>
        </View>
      ))}
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    gap: spacing.xs,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: spacing.sm,
  },
  label: {
    color: colors.textMuted,
  },
  value: {
    flexShrink: 1,
    textAlign: 'right',
  },
});
