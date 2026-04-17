import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { colors, font } from '../constants/theme';

const STEPS = ['Selfie', 'Measurements', 'AI Processing', 'Try-On'];
const ICONS  = ['📸', '📏', '🧠', '✨'];

interface Props { current: number }

export default function StepNav({ current }: Props) {
  return (
    <View style={styles.row}>
      {STEPS.map((label, i) => {
        const n     = i + 1;
        const done  = n < current;
        const active = n === current;
        return (
          <React.Fragment key={n}>
            <View style={styles.item}>
              <View style={[
                styles.circle,
                active && styles.circleActive,
                done   && styles.circleDone,
              ]}>
                <Text style={[styles.circleText, (active || done) && styles.circleTextActive]}>
                  {done ? '✓' : ICONS[i]}
                </Text>
              </View>
              <Text style={[styles.label, active && styles.labelActive, done && styles.labelDone]}>
                {label}
              </Text>
            </View>
            {i < STEPS.length - 1 && (
              <View style={[styles.connector, done && styles.connectorDone]} />
            )}
          </React.Fragment>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'center',
    paddingVertical: 16,
    paddingHorizontal: 8,
    backgroundColor: colors.bg,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  item: { alignItems: 'center', width: 70 },
  circle: {
    width: 36, height: 36, borderRadius: 18,
    backgroundColor: colors.card,
    borderWidth: 2, borderColor: colors.border,
    alignItems: 'center', justifyContent: 'center',
  },
  circleActive: { borderColor: colors.accent, backgroundColor: 'rgba(124,111,255,0.15)' },
  circleDone:   { borderColor: colors.accent, backgroundColor: colors.accent },
  circleText:     { fontSize: 14 },
  circleTextActive: { color: '#fff' },
  label: { fontSize: font.xs, color: colors.muted, marginTop: 4, textAlign: 'center', fontWeight: '600' },
  labelActive: { color: colors.accent },
  labelDone:   { color: colors.text },
  connector: {
    height: 2, flex: 1, backgroundColor: colors.border,
    marginTop: 17, marginHorizontal: 2,
  },
  connectorDone: { backgroundColor: colors.accent },
});
