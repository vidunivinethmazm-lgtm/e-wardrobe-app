import { Ionicons } from '@expo/vector-icons';
import { StyleSheet, Text, View } from 'react-native';

import { spacing } from '../theme';
import type { CapturePose } from '../types';

interface Props {
  pose: CapturePose;
}

const HEAD_WIDTH = 220;
const HEAD_HEIGHT = 300;
const SHOULDER_WIDTH = 320;
const SHOULDER_HEIGHT = 140;

const GUIDE_COLOR = 'rgba(255,255,255,0.85)';

const POSE_CAPTIONS: Record<CapturePose, string> = {
  front: 'Look straight ahead and fill the outline',
  left: 'Turn slightly to show your left profile',
  right: 'Turn slightly to show your right profile',
};

/**
 * Absolutely-positioned overlay shown on top of the camera preview during
 * guided avatar capture: a head/shoulders silhouette guide, plus a directional
 * chevron and caption for the left/right profile poses.
 */
export function CaptureSilhouette({ pose }: Props) {
  return (
    <View style={styles.container} pointerEvents="none">
      <View style={styles.guideArea}>
        <View style={styles.shoulders} />
        <View style={styles.head} />
        {pose !== 'front' ? (
          <Ionicons
            name={pose === 'left' ? 'chevron-back-circle' : 'chevron-forward-circle'}
            size={44}
            color={GUIDE_COLOR}
            style={[styles.chevron, pose === 'left' ? styles.chevronLeft : styles.chevronRight]}
          />
        ) : null}
      </View>
      <Text style={styles.caption}>{POSE_CAPTIONS[pose]}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  guideArea: {
    width: SHOULDER_WIDTH,
    height: HEAD_HEIGHT + SHOULDER_HEIGHT / 2,
    alignItems: 'center',
  },
  head: {
    width: HEAD_WIDTH,
    height: HEAD_HEIGHT,
    borderRadius: HEAD_WIDTH / 2,
    borderWidth: 3,
    borderColor: GUIDE_COLOR,
  },
  shoulders: {
    position: 'absolute',
    bottom: 0,
    alignSelf: 'center',
    width: SHOULDER_WIDTH,
    height: SHOULDER_HEIGHT,
    borderTopLeftRadius: SHOULDER_WIDTH / 2,
    borderTopRightRadius: SHOULDER_WIDTH / 2,
    borderWidth: 3,
    borderBottomWidth: 0,
    borderColor: GUIDE_COLOR,
  },
  chevron: {
    position: 'absolute',
    top: HEAD_HEIGHT / 2 - 22,
  },
  chevronLeft: {
    left: -10,
  },
  chevronRight: {
    right: -10,
  },
  caption: {
    marginTop: spacing.lg,
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
    textAlign: 'center',
    textShadowColor: 'rgba(0,0,0,0.6)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 4,
    paddingHorizontal: spacing.lg,
  },
});
