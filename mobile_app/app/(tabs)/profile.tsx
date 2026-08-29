import * as ImagePicker from 'expo-image-picker';
import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator, Image, ScrollView, StatusBar, StyleSheet, Text,
  TextInput, TouchableOpacity, View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { useAuth } from '../auth';

const AVATARS = ['👤', '🧑', '👩', '👨', '🧕', '👳', '🧑‍🎓', '👩‍💼', '👨‍💼', '🦸', '🧑‍🎨', '😎'];
const STYLES  = ['Casual', 'Smart casual', 'Formal', 'Streetwear', 'Traditional', 'Minimal', 'Bold'];

// A photo avatar is stored inline as a data: URI; an emoji avatar is just text.
const isPhoto = (a: string) => a.startsWith('data:');
const MAX_AVATAR_CHARS = 900_000;   // ~650 KB image once base64-encoded

export default function ProfileScreen() {
  const auth = useAuth();
  const user = auth.user;

  const [name, setName]       = useState('');
  const [avatar, setAvatar]   = useState('👤');
  const [gender, setGender]   = useState('');
  const [age, setAge]         = useState('');
  const [city, setCity]       = useState('');
  const [style, setStyle]     = useState('');
  const [bio, setBio]         = useState('');

  const [busy, setBusy]       = useState(false);
  const [msg, setMsg]         = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);

  useEffect(() => {
    if (!user) return;
    setName(user.name ?? '');
    setAvatar(user.profile.avatar || '👤');
    setGender(user.profile.gender ?? '');
    setAge(user.profile.age != null ? String(user.profile.age) : '');
    setCity(user.profile.city ?? '');
    setStyle(user.profile.style ?? '');
    setBio(user.profile.bio ?? '');
  }, [user]);

  if (!user) {
    return (
      <SafeAreaView style={styles.safe}>
        <ActivityIndicator style={{ marginTop: 60 }} color="#7C3AED" />
      </SafeAreaView>
    );
  }

  const pickAvatarPhoto = async () => {
    setMsg(null);
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      setMsg({ kind: 'err', text: 'Allow photo access to use a picture.' });
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.5,
      base64: true,
    });
    if (result.canceled || !result.assets?.[0]?.base64) return;
    const a = result.assets[0];
    const uri = `data:${a.mimeType ?? 'image/jpeg'};base64,${a.base64}`;
    if (uri.length > MAX_AVATAR_CHARS) {
      setMsg({ kind: 'err', text: 'That image is too large — pick a smaller one.' });
      return;
    }
    setAvatar(uri);
  };

  const save = async () => {
    setMsg(null);
    const ageNum = age.trim() ? Number(age) : null;
    if (ageNum != null && (Number.isNaN(ageNum) || ageNum < 1 || ageNum > 120)) {
      setMsg({ kind: 'err', text: 'Enter a valid age (1–120).' });
      return;
    }
    setBusy(true);
    try {
      await auth.updateProfile({
        name: name.trim(),
        avatar,
        gender: gender.trim(),
        age: ageNum as any,
        city: city.trim(),
        style: style.trim(),
        bio: bio.trim(),
      });
      setMsg({ kind: 'ok', text: 'Profile saved.' });
    } catch (e: any) {
      setMsg({ kind: 'err', text: e?.message ?? 'Could not save the profile.' });
    } finally {
      setBusy(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <StatusBar barStyle="light-content" backgroundColor="#2D1B69" />
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">

        <View style={styles.header}>
          {isPhoto(avatar)
            ? <Image source={{ uri: avatar }} style={styles.headerAvatarImg} />
            : <Text style={styles.headerAvatar}>{avatar}</Text>}
          <Text style={styles.headerName}>{name || user.email.split('@')[0]}</Text>
          <Text style={styles.headerEmail}>{user.email}</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.section}>PROFILE PICTURE</Text>
          <View style={styles.avatarGrid}>
            <TouchableOpacity
              style={[styles.uploadChip, isPhoto(avatar) && styles.avatarChipOn]}
              onPress={pickAvatarPhoto}
            >
              {isPhoto(avatar)
                ? <Image source={{ uri: avatar }} style={styles.uploadPreview} />
                : <>
                    <Text style={styles.uploadIcon}>📷</Text>
                    <Text style={styles.uploadText}>Upload</Text>
                  </>}
            </TouchableOpacity>

            {AVATARS.map(a => (
              <TouchableOpacity
                key={a}
                style={[styles.avatarChip, avatar === a && styles.avatarChipOn]}
                onPress={() => setAvatar(a)}
              >
                <Text style={styles.avatarEmoji}>{a}</Text>
              </TouchableOpacity>
            ))}
          </View>
          {isPhoto(avatar) && (
            <TouchableOpacity onPress={() => setAvatar('👤')}>
              <Text style={styles.removePhoto}>Remove photo</Text>
            </TouchableOpacity>
          )}

          <Text style={styles.label}>NAME</Text>
          <TextInput style={styles.input} value={name} onChangeText={setName}
            placeholder="Your name" placeholderTextColor="#9CA3AF" />

          <View style={styles.row}>
            <View style={styles.col}>
              <Text style={styles.label}>GENDER</Text>
              <TextInput style={styles.input} value={gender} onChangeText={setGender}
                placeholder="e.g. Female" placeholderTextColor="#9CA3AF" />
            </View>
            <View style={styles.col}>
              <Text style={styles.label}>AGE</Text>
              <TextInput style={styles.input} value={age} onChangeText={setAge}
                placeholder="e.g. 22" placeholderTextColor="#9CA3AF" keyboardType="number-pad" />
            </View>
          </View>

          <Text style={styles.label}>CITY</Text>
          <TextInput style={styles.input} value={city} onChangeText={setCity}
            placeholder="e.g. Colombo" placeholderTextColor="#9CA3AF" />

          <Text style={styles.label}>STYLE PREFERENCE</Text>
          <View style={styles.chipWrap}>
            {STYLES.map(s => (
              <TouchableOpacity
                key={s}
                style={[styles.chip, style === s && styles.chipOn]}
                onPress={() => setStyle(style === s ? '' : s)}
              >
                <Text style={[styles.chipText, style === s && styles.chipTextOn]}>{s}</Text>
              </TouchableOpacity>
            ))}
          </View>

          <Text style={styles.label}>ABOUT YOU</Text>
          <TextInput
            style={[styles.input, styles.textarea]}
            value={bio}
            onChangeText={setBio}
            placeholder="A line or two about your wardrobe goals…"
            placeholderTextColor="#9CA3AF"
            multiline
          />

          {msg && (
            <Text style={[styles.msg, msg.kind === 'ok' ? styles.msgOk : styles.msgErr]}>{msg.text}</Text>
          )}

          <TouchableOpacity style={[styles.save, busy && styles.saveBusy]} onPress={save} disabled={busy}>
            {busy ? <ActivityIndicator color="#FFF" /> : <Text style={styles.saveText}>Save profile</Text>}
          </TouchableOpacity>
        </View>

        <TouchableOpacity style={styles.logout} onPress={() => auth.logout()}>
          <Text style={styles.logoutText}>Sign out</Text>
        </TouchableOpacity>

        <View style={{ height: 40 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe:   { flex: 1, backgroundColor: '#2D1B69' },
  scroll: { backgroundColor: '#F3F0FF', paddingBottom: 24 },

  header:          { backgroundColor: '#2D1B69', alignItems: 'center', paddingTop: 12, paddingBottom: 28 },
  headerAvatar:    { fontSize: 60 },
  headerAvatarImg: { width: 76, height: 76, borderRadius: 38, borderWidth: 2, borderColor: '#C4B5FD', backgroundColor: '#EDE9FE' },
  headerName:      { color: '#FFFFFF', fontSize: 22, fontWeight: '800', marginTop: 6 },
  headerEmail:  { color: '#C4B5FD', fontSize: 13, marginTop: 4 },

  card: {
    backgroundColor: '#FFFFFF', margin: 16, borderRadius: 20, padding: 18,
    shadowColor: '#6D28D9', shadowOpacity: 0.08, shadowRadius: 10, shadowOffset: { width: 0, height: 4 }, elevation: 3,
  },

  section: { fontSize: 10, fontWeight: '800', color: '#7C3AED', letterSpacing: 1.5, marginBottom: 10 },
  label:   { fontSize: 10, fontWeight: '800', color: '#7C3AED', letterSpacing: 1.5, marginBottom: 6, marginTop: 14 },

  input: {
    borderWidth: 1, borderColor: '#E5E7EB', borderRadius: 12, paddingHorizontal: 14, paddingVertical: 11,
    fontSize: 15, color: '#1F2937', backgroundColor: '#FAFAFA',
  },
  textarea: { minHeight: 70, textAlignVertical: 'top' },

  row: { flexDirection: 'row', gap: 12 },
  col: { flex: 1 },

  avatarGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  avatarChip: {
    width: 46, height: 46, borderRadius: 12, alignItems: 'center', justifyContent: 'center',
    backgroundColor: '#F3F0FF', borderWidth: 2, borderColor: 'transparent',
  },
  avatarChipOn: { borderColor: '#7C3AED', backgroundColor: '#EDE9FE' },
  avatarEmoji:  { fontSize: 24 },

  uploadChip: {
    width: 46, height: 46, borderRadius: 12, alignItems: 'center', justifyContent: 'center',
    backgroundColor: '#EDE9FE', borderWidth: 2, borderColor: '#C4B5FD', borderStyle: 'dashed',
    overflow: 'hidden',
  },
  uploadIcon:    { fontSize: 16 },
  uploadText:    { fontSize: 7, fontWeight: '800', color: '#7C3AED', letterSpacing: 0.5 },
  uploadPreview: { width: '100%', height: '100%' },
  removePhoto:   { marginTop: 10, fontSize: 12, fontWeight: '700', color: '#B91C1C' },

  chipWrap:    { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip:        { paddingHorizontal: 12, paddingVertical: 7, borderRadius: 999, backgroundColor: '#F3F0FF', borderWidth: 1, borderColor: '#EDE9FE' },
  chipOn:      { backgroundColor: '#7C3AED', borderColor: '#7C3AED' },
  chipText:    { fontSize: 12, fontWeight: '700', color: '#7C3AED' },
  chipTextOn:  { color: '#FFFFFF' },

  msg:    { marginTop: 16, fontSize: 13, fontWeight: '600' },
  msgOk:  { color: '#047857' },
  msgErr: { color: '#B91C1C' },

  save:     { backgroundColor: '#7C3AED', borderRadius: 14, paddingVertical: 14, alignItems: 'center', marginTop: 20 },
  saveBusy: { opacity: 0.7 },
  saveText: { color: '#FFFFFF', fontSize: 15, fontWeight: '800' },

  logout:     { marginHorizontal: 16, marginTop: 4, paddingVertical: 13, borderRadius: 14, alignItems: 'center', borderWidth: 1, borderColor: '#E5D6F5', backgroundColor: '#FFFFFF' },
  logoutText: { color: '#B91C1C', fontSize: 14, fontWeight: '700' },
});
