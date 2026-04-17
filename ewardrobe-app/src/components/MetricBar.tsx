import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { colors, font } from '../constants/theme';

interface Props {
  label: string;
  value: number;
  max: number;
  unit?: string;
}

export default function MetricBar({ label, value, max, unit = 'cm' }: Props) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <View style={styles.row}>
      <Text style={styles.label}>{label}</Text>
      <View style={styles.track}>
        <View style={[styles.fill, { width: `${pct}%` as any }]} />
      </View>
      <Text style={styles.val}>{value}{unit}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  row:   { flexDirection: 'row', alignItems: 'center', gap: 8, marginVertical: 4 },
  label: { fontSize: font.xs, color: colors.muted, width: 60 },
  track: { flex: 1, height: 6, backgroundColor: colors.border, borderRadius: 3, overflow: 'hidden' },
  fill:  { height: '100%', borderRadius: 3, backgroundColor: colors.accent },
  val:   { fontSize: font.xs, color: colors.text, width: 50, textAlign: 'right' },
});
