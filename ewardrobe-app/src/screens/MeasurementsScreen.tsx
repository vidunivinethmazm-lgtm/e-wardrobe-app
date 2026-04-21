import React, { useState } from 'react';
import {
  View, Text, TextInput, ScrollView,
  TouchableOpacity, StyleSheet,
} from 'react-native';
import MetricBar from '../components/MetricBar';
import { colors, font, radius } from '../constants/theme';

export interface Measurements {
  shoulder: string; chest: string; waist: string; height: string;
  hip: string; inseam: string; styles: string; occasion: string;
}

interface Props {
  onBack: () => void;
  onNext: (m: Measurements) => void;
}

const STYLE_OPTIONS = [
  { label: 'Smart Casual',    value: 'smart_casual,casual' },
  { label: 'Formal / Business', value: 'formal,smart_casual' },
  { label: 'Casual',          value: 'casual' },
  { label: 'Evening / Formal', value: 'evening,formal' },
];
const OCCASION_OPTIONS = [
  { label: 'Casual', value: 'casual' },
  { label: 'Office', value: 'office' },
  { label: 'Interview', value: 'interview' },
  { label: 'Date Night', value: 'date_night' },
  { label: 'Formal Event', value: 'formal' },
];

const SIZE_THRESHOLDS: [number, string][] = [
  [82,'XS'],[88,'S'],[96,'M'],[104,'L'],[112,'XL'],[124,'XXL'],[Infinity,'XXXL']
];

function estimateSize(chest: number) {
  return SIZE_THRESHOLDS.find(([t]) => chest <= t)?.[1] ?? 'XXXL';
}
function estimateBodyType(chest: number, waist: number, shoulder: number) {
  const hip  = waist + 25;
  const wDef = (chest + hip) / 2 - waist;
  const sHR  = hip / (shoulder * 2.3);
  if (wDef > 9 && Math.abs(sHR - 1) < 0.08) return 'Hourglass';
  if (sHR < 0.87) return 'Inv. Triangle';
  if (sHR > 1.13) return 'Pear';
  return 'Rectangle';
}

export default function MeasurementsScreen({ onBack, onNext }: Props) {
  const [m, setM] = useState<Measurements>({
    shoulder: '42', chest: '92', waist: '72', height: '168',
    hip: '', inseam: '', styles: 'smart_casual,casual', occasion: 'casual',
  });

  const num = (key: keyof Measurements) => parseFloat(m[key]) || 0;
  const chest = num('chest'), shoulder = num('shoulder'), waist = num('waist');

  const valid =
    num('shoulder') >= 30 && num('chest') >= 60 &&
    num('waist') >= 50 && num('height') >= 120;

  function field(key: keyof Measurements, label: string, min: number, max: number, unit = 'cm', optional = false) {
    const val = parseFloat(m[key]);
    const err = !optional && m[key] !== '' && (isNaN(val) || val < min || val > max);
    return (
      <View style={styles.formGroup} key={key}>
        <Text style={styles.label}>{label}{optional ? <Text style={styles.opt}> (optional)</Text> : ' *'}</Text>
        <View style={styles.inputWrap}>
          <TextInput
            style={[styles.input, err && styles.inputErr]}
            keyboardType="numeric"
            value={m[key]}
            placeholder={optional ? 'Auto' : ''}
            placeholderTextColor={colors.muted}
            onChangeText={v => setM(prev => ({ ...prev, [key]: v }))}
          />
          <Text style={styles.unit}>{unit}</Text>
        </View>
        {err && <Text style={styles.errMsg}>Range: {min}–{max} {unit}</Text>}
      </View>
    );
  }

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
      <Text style={styles.title}>📏 Body Measurements</Text>
      <Text style={styles.sub}>
        Enter measurements in centimetres. AI uses these to scale your 3D avatar.
      </Text>

      <View style={styles.grid}>
        {field('shoulder', 'Shoulder Width', 30, 65)}
        {field('chest',    'Chest Circumference', 60, 160)}
        {field('waist',    'Waist Circumference', 50, 150)}
        {field('height',   'Height', 120, 230)}
        {field('hip',      'Hip', 60, 175, 'cm', true)}
        {field('inseam',   'Inseam', 60, 110, 'cm', true)}
      </View>

      {/* Style picker */}
      <Text style={styles.label}>Style Preference</Text>
      <View style={styles.chipRow}>
        {STYLE_OPTIONS.map(o => (
          <TouchableOpacity
            key={o.value}
            style={[styles.chip, m.styles === o.value && styles.chipActive]}
            onPress={() => setM(prev => ({ ...prev, styles: o.value }))}
          >
            <Text style={[styles.chipText, m.styles === o.value && styles.chipTextActive]}>
              {o.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Occasion picker */}
      <Text style={[styles.label, { marginTop: 12 }]}>Occasion</Text>
      <View style={styles.chipRow}>
        {OCCASION_OPTIONS.map(o => (
          <TouchableOpacity
            key={o.value}
            style={[styles.chip, m.occasion === o.value && styles.chipActive]}
            onPress={() => setM(prev => ({ ...prev, occasion: o.value }))}
          >
            <Text style={[styles.chipText, m.occasion === o.value && styles.chipTextActive]}>
              {o.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Live preview */}
      <View style={styles.preview}>
        <Text style={styles.previewTitle}>BODY PROPORTIONS PREVIEW</Text>
        <MetricBar label="Shoulder" value={num('shoulder')} max={65} />
        <MetricBar label="Chest"    value={chest}           max={160} />
        <MetricBar label="Waist"    value={waist}           max={150} />
        <MetricBar label="Height"   value={num('height')}   max={230} />
        <View style={styles.previewRow}>
          <Text style={styles.previewStat}>
            Size: <Text style={styles.highlight}>{estimateSize(chest)}</Text>
          </Text>
          <Text style={styles.previewStat}>
            Type: <Text style={styles.highlight}>{estimateBodyType(chest, waist, shoulder)}</Text>
          </Text>
        </View>
      </View>

      <View style={styles.footer}>
        <TouchableOpacity style={styles.backBtn} onPress={onBack}>
          <Text style={styles.backText}>← Back</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.nextBtn, !valid && styles.nextBtnDisabled]}
          disabled={!valid}
          onPress={() => onNext(m)}
        >
          <Text style={styles.nextText}>🧠 Analyse & Try On</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll:  { flex: 1, backgroundColor: colors.bg },
  content: { padding: 20, paddingBottom: 40 },
  title:   { fontSize: font.xl, fontWeight: '800', color: colors.text, marginBottom: 6 },
  sub:     { fontSize: font.sm, color: colors.muted, marginBottom: 20, lineHeight: 20 },
  grid:    { flexDirection: 'row', flexWrap: 'wrap', gap: 12, marginBottom: 16 },
  formGroup: { width: '47%' },
  label: { fontSize: font.xs, color: colors.muted, fontWeight: '700', letterSpacing: 0.8, textTransform: 'uppercase', marginBottom: 4 },
  opt:   { fontSize: font.xs, color: colors.muted, textTransform: 'none' },
  inputWrap: { position: 'relative' },
  input: {
    backgroundColor: colors.bg, borderWidth: 1, borderColor: colors.border,
    borderRadius: radius.sm, paddingVertical: 10, paddingHorizontal: 12,
    paddingRight: 36, color: colors.text, fontSize: font.md,
  },
  inputErr: { borderColor: colors.red },
  unit:     { position: 'absolute', right: 10, top: 12, fontSize: font.xs, color: colors.muted },
  errMsg:   { fontSize: font.xs, color: colors.red, marginTop: 2 },

  chipRow:  { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: 8 },
  chip: {
    paddingVertical: 7, paddingHorizontal: 14,
    borderRadius: radius.full, borderWidth: 1, borderColor: colors.border,
    backgroundColor: colors.card,
  },
  chipActive: { borderColor: colors.accent, backgroundColor: 'rgba(124,111,255,0.15)' },
  chipText:   { fontSize: font.sm, color: colors.muted, fontWeight: '600' },
  chipTextActive: { color: colors.accent },

  preview: {
    backgroundColor: colors.card, borderRadius: radius.md,
    padding: 16, borderWidth: 1, borderColor: colors.border, marginTop: 16,
  },
  previewTitle: { fontSize: font.xs, color: colors.muted, fontWeight: '700', letterSpacing: 0.8, marginBottom: 10 },
  previewRow:   { flexDirection: 'row', gap: 20, marginTop: 10 },
  previewStat:  { fontSize: font.sm, color: colors.muted },
  highlight:    { color: colors.accent, fontWeight: '700' },

  footer:  { flexDirection: 'row', justifyContent: 'space-between', marginTop: 24, gap: 12 },
  backBtn: {
    flex: 1, paddingVertical: 14, alignItems: 'center',
    backgroundColor: colors.card, borderRadius: radius.md,
    borderWidth: 1, borderColor: colors.border,
  },
  backText: { color: colors.text, fontWeight: '700', fontSize: font.md },
  nextBtn:  { flex: 2, paddingVertical: 14, alignItems: 'center', backgroundColor: colors.accent, borderRadius: radius.md },
  nextBtnDisabled: { opacity: 0.35 },
  nextText: { color: '#fff', fontWeight: '700', fontSize: font.md },
});
