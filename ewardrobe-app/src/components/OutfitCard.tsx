import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { colors, font, radius } from '../constants/theme';

interface Item { name: string; category: string; colours?: string[] }
interface Outfit {
  name: string; style: string; occasion: string;
  score: number; items: Item[];
}

interface Props {
  outfit: Outfit;
  index: number;
  total: number;
  active: boolean;
  onPress: () => void;
}

const COLOUR_MAP: Record<string, string> = {
  white: '#FFFFFF', light_blue: '#ADD8E6', navy: '#001F5B', black: '#1A1A1A',
  charcoal: '#36454F', khaki: '#C3B091', olive: '#808000', emerald: '#50C878',
  terracotta: '#E2725B', dark_indigo: '#1B1464', mid_grey: '#888888',
  cobalt_blue: '#0047AB', burgundy: '#800020', deep_red: '#8B0000',
  champagne: '#F7E7CE',
};

export default function OutfitCard({ outfit, index, total, active, onPress }: Props) {
  return (
    <TouchableOpacity
      style={[styles.card, active && styles.cardActive]}
      onPress={onPress}
      activeOpacity={0.8}
    >
      <Text style={styles.num}>Outfit {index + 1} of {total}</Text>
      <Text style={styles.name}>{outfit.name}</Text>
      <Text style={styles.style}>{outfit.style.replace(/_/g, ' ')} · {outfit.occasion}</Text>

      <View style={styles.tags}>
        {outfit.items.map((item, i) => (
          <View key={i} style={styles.tag}>
            <View style={[styles.swatch, { backgroundColor: COLOUR_MAP[item.colours?.[0] ?? ''] || colors.accent }]} />
            <Text style={styles.tagText}>{item.name}</Text>
          </View>
        ))}
      </View>

      <View style={styles.scoreTrack}>
        <View style={[styles.scoreFill, { width: `${Math.round(outfit.score * 100)}%` as any }]} />
      </View>
      <Text style={styles.scoreLabel}>Match score: {(outfit.score * 100).toFixed(0)}%</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.bg,
    borderWidth: 1, borderColor: colors.border,
    borderRadius: radius.md, padding: 14, marginBottom: 10,
  },
  cardActive: { borderColor: colors.accent, backgroundColor: 'rgba(124,111,255,0.07)' },
  num:   { fontSize: font.xs, color: colors.muted, marginBottom: 2 },
  name:  { fontSize: font.md, fontWeight: '700', color: colors.text },
  style: { fontSize: font.xs, color: colors.accent, marginTop: 2 },
  tags:  { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 8 },
  tag:   { flexDirection: 'row', alignItems: 'center', gap: 4,
           backgroundColor: 'rgba(124,111,255,0.12)', paddingHorizontal: 8,
           paddingVertical: 3, borderRadius: radius.full },
  swatch: { width: 8, height: 8, borderRadius: 4 },
  tagText: { fontSize: font.xs, color: colors.accent },
  scoreTrack: { height: 3, backgroundColor: colors.border, borderRadius: 2, marginTop: 10, overflow: 'hidden' },
  scoreFill:  { height: '100%', borderRadius: 2, backgroundColor: colors.accent },
  scoreLabel: { fontSize: font.xs, color: colors.muted, marginTop: 3 },
});
