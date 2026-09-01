import { useRef, useState } from 'react';
import { LayoutChangeEvent, PanResponder, StyleSheet, View } from 'react-native';

import { clamp } from '../services/bodyScaling';
import { colors, radii, shadow } from '../theme';

interface Props {
  value: number;
  min: number;
  max: number;
  onValueChange: (value: number) => void;
}

const THUMB_SIZE = 22;
const TRACK_HEIGHT = 4;

/** Themed, controlled slider. Drag the thumb to call `onValueChange` with a
 * value in `[min, max]`. Built on `PanResponder` (no native dependency) so it
 * renders identically on web and native, like `AvatarViewer3D`'s drag-to-rotate. */
export function Slider({ value, min, max, onValueChange }: Props) {
  const [trackWidth, setTrackWidth] = useState(0);
  const trackWidthRef = useRef(0);
  const valueRef = useRef(value);
  const startValueRef = useRef(value);
  valueRef.current = value;

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onPanResponderGrant: () => {
        startValueRef.current = valueRef.current;
      },
      onPanResponderMove: (_event, gesture) => {
        const width = trackWidthRef.current;
        if (width <= 0) return;
        const delta = (gesture.dx / width) * (max - min);
        onValueChange(clamp(startValueRef.current + delta, [min, max]));
      },
    })
  ).current;

  function handleLayout(event: LayoutChangeEvent) {
    trackWidthRef.current = event.nativeEvent.layout.width;
    setTrackWidth(event.nativeEvent.layout.width);
  }

  const percent = max > min ? clamp((value - min) / (max - min), [0, 1]) : 0;
  const thumbCenter = percent * trackWidth;

  return (
    <View style={styles.container} onLayout={handleLayout}>
      <View style={styles.track} />
      <View style={[styles.fill, { width: thumbCenter }]} />
      <View style={[styles.thumb, { left: thumbCenter - THUMB_SIZE / 2 }]} {...panResponder.panHandlers} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    height: THUMB_SIZE,
    justifyContent: 'center',
  },
  track: {
    height: TRACK_HEIGHT,
    borderRadius: radii.pill,
    backgroundColor: colors.border,
  },
  fill: {
    position: 'absolute',
    height: TRACK_HEIGHT,
    borderRadius: radii.pill,
    backgroundColor: colors.primary,
  },
  thumb: {
    position: 'absolute',
    width: THUMB_SIZE,
    height: THUMB_SIZE,
    borderRadius: THUMB_SIZE / 2,
    backgroundColor: colors.primary,
    borderWidth: 2,
    borderColor: colors.surface,
    ...shadow,
  },
});
