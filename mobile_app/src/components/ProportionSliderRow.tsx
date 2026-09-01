import { useEffect, useState } from 'react';
import { StyleSheet, Text, TextInput, View } from 'react-native';

import { clamp } from '../services/bodyScaling';
import { colors, radii, spacing, typography } from '../theme';
import { Slider } from './Slider';

interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  onValueChange: (value: number) => void;
}

function formatSigned(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}`;
}

/** A "Customize body" row: a label, an editable numeric value (type a number
 * to set the slider exactly), and a draggable `Slider` — both control the
 * same value, clamped to `[min, max]`. */
export function ProportionSliderRow({ label, value, min, max, onValueChange }: Props) {
  const [text, setText] = useState(formatSigned(value));
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    if (!editing) setText(formatSigned(value));
  }, [value, editing]);

  function commit() {
    setEditing(false);
    const parsed = Number(text);
    if (Number.isFinite(parsed)) onValueChange(clamp(parsed, [min, max]));
  }

  return (
    <View style={styles.row}>
      <View style={styles.labelRow}>
        <Text style={typography.body}>{label}</Text>
        <TextInput
          style={styles.input}
          value={text}
          onChangeText={setText}
          onFocus={() => setEditing(true)}
          onBlur={commit}
          onSubmitEditing={commit}
          keyboardType="default"
          returnKeyType="done"
          selectTextOnFocus
        />
      </View>
      <Slider value={value} min={min} max={max} onValueChange={onValueChange} />
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    marginTop: spacing.sm,
  },
  labelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  input: {
    ...typography.label,
    minWidth: 56,
    textAlign: 'right',
    paddingVertical: 2,
    paddingHorizontal: spacing.xs,
    borderRadius: radii.sm,
    borderWidth: 1,
    borderColor: colors.border,
  },
});
