import React, { useState, useRef } from 'react';
import {
  View, Text, TouchableOpacity, Image, StyleSheet,
  ScrollView, Alert, Platform,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { colors, font, radius } from '../constants/theme';

interface Props {
  onSelfieReady: (uri: string) => void;
  onNext: () => void;
}

export default function SelfieScreen({ onSelfieReady, onNext }: Props) {
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraOpen, setCameraOpen]     = useState(false);
  const [selfieUri, setSelfieUri]       = useState<string | null>(null);
  const cameraRef = useRef<CameraView>(null);

  async function openCamera() {
    if (!permission?.granted) {
      const res = await requestPermission();
      if (!res.granted) { Alert.alert('Camera permission required'); return; }
    }
    setCameraOpen(true);
  }

  async function capture() {
    if (!cameraRef.current) return;
    const photo = await cameraRef.current.takePictureAsync({ quality: 0.8 });
    if (photo?.uri) {
      setSelfieUri(photo.uri);
      onSelfieReady(photo.uri);
      setCameraOpen(false);
    }
  }

  async function pickFromGallery() {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.8, allowsEditing: true, aspect: [1, 1],
    });
    if (!res.canceled && res.assets[0]) {
      const uri = res.assets[0].uri;
      setSelfieUri(uri);
      onSelfieReady(uri);
    }
  }

  if (cameraOpen) {
    return (
      <View style={styles.cameraFull}>
        <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="front" />
        <View style={styles.cameraBar}>
          <TouchableOpacity style={styles.cancelBtn} onPress={() => setCameraOpen(false)}>
            <Text style={styles.cancelText}>✕ Cancel</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.captureBtn} onPress={capture}>
            <View style={styles.captureInner} />
          </TouchableOpacity>
          <View style={{ width: 80 }} />
        </View>
      </View>
    );
  }

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
      <Text style={styles.title}>📸 Capture Your Selfie</Text>
      <Text style={styles.sub}>
        Take a photo or upload one. Face should be clearly visible,
        front-facing with good lighting.
      </Text>

      {selfieUri ? (
        <View style={styles.previewWrap}>
          <Image source={{ uri: selfieUri }} style={styles.preview} />
          <Text style={styles.readyText}>✅ Selfie ready for AI processing</Text>
          <TouchableOpacity style={styles.retakeBtn} onPress={() => setSelfieUri(null)}>
            <Text style={styles.retakeText}>🔄 Retake</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <View style={styles.buttonGroup}>
          <TouchableOpacity style={styles.btnPrimary} onPress={openCamera}>
            <Text style={styles.btnText}>📷  Open Camera</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.btnSecondary} onPress={pickFromGallery}>
            <Text style={styles.btnTextSec}>📂  Upload from Gallery</Text>
          </TouchableOpacity>
        </View>
      )}

      <View style={styles.tips}>
        <Text style={styles.tipsTitle}>TIPS FOR BEST RESULTS</Text>
        {[
          'Look directly at the camera — front-facing angle',
          'Good lighting — avoid harsh shadows on face',
          'Clear background helps face detection accuracy',
          'AI detects 468 facial landmarks for personalisation',
        ].map((tip, i) => (
          <View key={i} style={styles.tip}>
            <View style={styles.dot} />
            <Text style={styles.tipText}>{tip}</Text>
          </View>
        ))}
      </View>

      <TouchableOpacity
        style={[styles.nextBtn, !selfieUri && styles.nextBtnDisabled]}
        disabled={!selfieUri}
        onPress={onNext}
      >
        <Text style={styles.nextText}>Next: Measurements →</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.bg },
  content: { padding: 20, paddingBottom: 40 },
  title:   { fontSize: font.xl, fontWeight: '800', color: colors.text, marginBottom: 6 },
  sub:     { fontSize: font.sm, color: colors.muted, marginBottom: 24, lineHeight: 20 },

  previewWrap: { alignItems: 'center', marginBottom: 24 },
  preview: {
    width: 200, height: 200, borderRadius: 100,
    borderWidth: 3, borderColor: colors.accent,
  },
  readyText: { color: colors.green, fontSize: font.sm, marginTop: 12, fontWeight: '600' },
  retakeBtn: {
    marginTop: 12, paddingVertical: 10, paddingHorizontal: 24,
    backgroundColor: colors.card, borderRadius: radius.md,
    borderWidth: 1, borderColor: colors.border,
  },
  retakeText: { color: colors.text, fontSize: font.sm, fontWeight: '700' },

  buttonGroup: { gap: 12, marginBottom: 24 },
  btnPrimary: {
    backgroundColor: colors.accent, borderRadius: radius.md,
    paddingVertical: 16, alignItems: 'center',
  },
  btnSecondary: {
    backgroundColor: colors.card, borderRadius: radius.md,
    paddingVertical: 16, alignItems: 'center',
    borderWidth: 1, borderColor: colors.border,
  },
  btnText:    { color: '#fff', fontSize: font.md, fontWeight: '700' },
  btnTextSec: { color: colors.text, fontSize: font.md, fontWeight: '700' },

  tips: {
    backgroundColor: colors.card, borderRadius: radius.md,
    padding: 16, borderWidth: 1, borderColor: colors.border, marginBottom: 24,
  },
  tipsTitle: { fontSize: font.xs, color: colors.muted, fontWeight: '700', letterSpacing: 1, marginBottom: 10 },
  tip:     { flexDirection: 'row', alignItems: 'flex-start', gap: 10, marginBottom: 8 },
  dot:     { width: 6, height: 6, borderRadius: 3, backgroundColor: colors.accent, marginTop: 4 },
  tipText: { fontSize: font.sm, color: colors.muted, flex: 1, lineHeight: 18 },

  nextBtn: {
    backgroundColor: colors.accent, borderRadius: radius.md,
    paddingVertical: 16, alignItems: 'center',
  },
  nextBtnDisabled: { opacity: 0.35 },
  nextText: { color: '#fff', fontSize: font.md, fontWeight: '700' },

  cameraFull: { flex: 1, backgroundColor: '#000' },
  cameraBar: {
    position: 'absolute', bottom: 40, left: 0, right: 0,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 30,
  },
  cancelBtn: {
    backgroundColor: 'rgba(0,0,0,0.6)', borderRadius: radius.md,
    paddingVertical: 10, paddingHorizontal: 16,
  },
  cancelText: { color: '#fff', fontWeight: '700' },
  captureBtn: {
    width: 72, height: 72, borderRadius: 36,
    backgroundColor: 'rgba(255,255,255,0.3)',
    alignItems: 'center', justifyContent: 'center',
    borderWidth: 3, borderColor: '#fff',
  },
  captureInner: { width: 56, height: 56, borderRadius: 28, backgroundColor: '#fff' },
});
