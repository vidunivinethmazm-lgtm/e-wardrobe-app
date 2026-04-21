import React, { useEffect, useRef, useState } from 'react';
import { View, Text, StyleSheet, Animated, Easing } from 'react-native';
import { colors, font, radius } from '../constants/theme';

const STAGES = [
  { icon: '📏', name: 'Stage 1 — Body Calibration',  desc: 'Validating measurements · Computing avatar scale' },
  { icon: '👁️', name: 'Stage 2 — Face Processing',   desc: 'MediaPipe 468 landmarks · CNN 15 keypoints' },
  { icon: '👔', name: 'Stage 3 — Outfit Matching',    desc: 'NisfaMatchmaking · Colour harmony scoring' },
  { icon: '🎭', name: 'Stage 4 — Avatar Generation',  desc: 'Avatar scaling · Mixamo animation setup' },
];

interface Props { apiPromise: Promise<any> | null; onDone: (result: any) => void; onError: (msg: string) => void; }

export default function ProcessingScreen({ apiPromise, onDone, onError }: Props) {
  const [stageStates, setStageStates] = useState<('pending'|'running'|'done')[]>(
    ['pending','pending','pending','pending']
  );
  const [title, setTitle] = useState('Initialising AI Pipeline…');
  const [sub,   setSub]   = useState('Preparing your personalised virtual try-on');
  const progress = useRef(STAGES.map(() => new Animated.Value(0))).current;
  const pulse = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 1.15, duration: 900, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 1.0,  duration: 900, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
      ])
    ).start();
  }, []);

  useEffect(() => {
    if (!apiPromise) return;
    runAnimations();
    apiPromise.then(onDone).catch(e => onError(e.message));
  }, [apiPromise]);

  async function runAnimations() {
    const DURATIONS = [900, 1500, 1000, 800];
    const TITLES = [
      'Validating body measurements…',
      'Analysing facial landmarks…',
      'Matching outfits from wardrobe…',
      'Generating your 3D avatar…',
    ];

    for (let i = 0; i < 4; i++) {
      setTitle(TITLES[i]);
      setSub(STAGES[i].desc);
      setStageStates(prev => prev.map((s, j) => j === i ? 'running' : s));

      await new Promise<void>(resolve =>
        Animated.timing(progress[i], {
          toValue: 1, duration: DURATIONS[i],
          easing: Easing.out(Easing.ease), useNativeDriver: false,
        }).start(() => resolve())
      );

      setStageStates(prev => prev.map((s, j) => j === i ? 'done' : s));
      await new Promise(r => setTimeout(r, 120));
    }
    setTitle('Finalising recommendations…');
    setSub('Applying colour harmony scoring');
  }

  return (
    <View style={styles.container}>
      <Animated.View style={[styles.pulse, { transform: [{ scale: pulse }] }]}>
        <Text style={styles.pulseIcon}>🧠</Text>
      </Animated.View>

      <Text style={styles.title}>{title}</Text>
      <Text style={styles.sub}>{sub}</Text>

      <View style={styles.stageList}>
        {STAGES.map((stage, i) => {
          const state = stageStates[i];
          return (
            <View key={i} style={[styles.stageRow, state === 'running' && styles.stageRunning, state === 'done' && styles.stageDone]}>
              <Text style={styles.stageIcon}>{stage.icon}</Text>
              <View style={styles.stageInfo}>
                <Text style={styles.stageName}>{stage.name}</Text>
                <Text style={styles.stageDesc}>{stage.desc}</Text>
                <View style={styles.barTrack}>
                  <Animated.View style={[styles.barFill, {
                    width: progress[i].interpolate({ inputRange: [0,1], outputRange: ['0%','100%'] }),
                    backgroundColor: state === 'done' ? colors.green : colors.accent,
                  }]} />
                </View>
              </View>
              <Text style={[styles.stateLabel,
                state === 'running' && styles.stateLabelRunning,
                state === 'done'    && styles.stateLabelDone,
              ]}>
                {state === 'pending' ? 'Pending' : state === 'running' ? 'Running…' : '✓ Done'}
              </Text>
            </View>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg, padding: 24, alignItems: 'center' },
  pulse: {
    width: 100, height: 100, borderRadius: 50,
    backgroundColor: 'rgba(124,111,255,0.2)',
    alignItems: 'center', justifyContent: 'center', marginBottom: 24, marginTop: 16,
  },
  pulseIcon: { fontSize: 44 },
  title: { fontSize: font.lg, fontWeight: '700', color: colors.text, textAlign: 'center' },
  sub:   { fontSize: font.sm, color: colors.muted, textAlign: 'center', marginTop: 6, marginBottom: 24 },
  stageList: { width: '100%', gap: 10 },
  stageRow: {
    flexDirection: 'row', alignItems: 'center', gap: 12,
    backgroundColor: colors.card, borderWidth: 1, borderColor: colors.border,
    borderRadius: radius.md, padding: 14,
  },
  stageRunning: { borderColor: colors.accent, backgroundColor: 'rgba(124,111,255,0.06)' },
  stageDone:    { borderColor: colors.green,  backgroundColor: 'rgba(76,175,80,0.06)' },
  stageIcon: { fontSize: 22, width: 30, textAlign: 'center' },
  stageInfo: { flex: 1 },
  stageName: { fontSize: font.sm, fontWeight: '700', color: colors.text },
  stageDesc: { fontSize: font.xs, color: colors.muted, marginTop: 2 },
  barTrack:  { height: 3, backgroundColor: colors.border, borderRadius: 2, marginTop: 8, overflow: 'hidden' },
  barFill:   { height: '100%', borderRadius: 2 },
  stateLabel:        { fontSize: font.xs, color: colors.muted, fontWeight: '700' },
  stateLabelRunning: { color: colors.accent },
  stateLabelDone:    { color: colors.green },
});
