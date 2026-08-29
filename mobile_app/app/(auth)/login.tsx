import React, { useState } from 'react';
import {
  ActivityIndicator, KeyboardAvoidingView, Platform, ScrollView,
  StatusBar, StyleSheet, Text, TextInput, TouchableOpacity, View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { authStore } from '../auth';

type Mode = 'login' | 'register';

export default function LoginScreen() {
  const [mode, setMode]         = useState<Mode>('login');
  const [name, setName]         = useState('');
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy]         = useState(false);
  const [error, setError]       = useState<string | null>(null);

  const isRegister = mode === 'register';

  const submit = async () => {
    setError(null);
    if (!email.trim() || !password) {
      setError('Enter your email and password.');
      return;
    }
    if (isRegister && password.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }
    setBusy(true);
    try {
      if (isRegister) {
        await authStore.register(email.trim(), password, name.trim());
      } else {
        await authStore.login(email.trim(), password);
      }
      // The auth gate in app/_layout.tsx redirects into the tabs.
    } catch (e: any) {
      setError(e?.message ?? 'Something went wrong. Try again.');
    } finally {
      setBusy(false);
    }
  };

  const swap = () => {
    setMode(isRegister ? 'login' : 'register');
    setError(null);
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top', 'bottom']}>
      <StatusBar barStyle="light-content" backgroundColor="#2D1B69" />
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
          <View style={styles.brand}>
            <Text style={styles.logo}>👗</Text>
            <Text style={styles.title}>E-Wardrobe AI</Text>
            <Text style={styles.subtitle}>
              {isRegister ? 'Create your account' : 'Welcome back'}
            </Text>
          </View>

          <View style={styles.card}>
            <View style={styles.switchRow}>
              <TouchableOpacity
                style={[styles.switchBtn, !isRegister && styles.switchBtnActive]}
                onPress={() => !isRegister || swap()}
              >
                <Text style={[styles.switchText, !isRegister && styles.switchTextActive]}>Sign in</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.switchBtn, isRegister && styles.switchBtnActive]}
                onPress={() => isRegister || swap()}
              >
                <Text style={[styles.switchText, isRegister && styles.switchTextActive]}>Create account</Text>
              </TouchableOpacity>
            </View>

            {isRegister && (
              <>
                <Text style={styles.label}>NAME</Text>
                <TextInput
                  style={styles.input}
                  value={name}
                  onChangeText={setName}
                  placeholder="Your name"
                  placeholderTextColor="#9CA3AF"
                  autoCapitalize="words"
                />
              </>
            )}

            <Text style={styles.label}>EMAIL</Text>
            <TextInput
              style={styles.input}
              value={email}
              onChangeText={setEmail}
              placeholder="you@example.com"
              placeholderTextColor="#9CA3AF"
              keyboardType="email-address"
              autoCapitalize="none"
              autoCorrect={false}
            />

            <Text style={styles.label}>PASSWORD</Text>
            <TextInput
              style={styles.input}
              value={password}
              onChangeText={setPassword}
              placeholder={isRegister ? 'At least 6 characters' : 'Your password'}
              placeholderTextColor="#9CA3AF"
              secureTextEntry
            />

            {error && <Text style={styles.error}>{error}</Text>}

            <TouchableOpacity
              style={[styles.submit, busy && styles.submitBusy]}
              onPress={submit}
              disabled={busy}
              activeOpacity={0.85}
            >
              {busy
                ? <ActivityIndicator color="#FFF" />
                : <Text style={styles.submitText}>{isRegister ? 'Create account' : 'Sign in'}</Text>}
            </TouchableOpacity>

            <TouchableOpacity onPress={swap} style={styles.altRow}>
              <Text style={styles.altText}>
                {isRegister ? 'Already have an account? ' : "Don't have an account? "}
                <Text style={styles.altLink}>{isRegister ? 'Sign in' : 'Create one'}</Text>
              </Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe:   { flex: 1, backgroundColor: '#2D1B69' },
  scroll: { flexGrow: 1, justifyContent: 'center', padding: 24 },

  brand:    { alignItems: 'center', marginBottom: 28 },
  logo:     { fontSize: 52, marginBottom: 8 },
  title:    { color: '#FFFFFF', fontSize: 28, fontWeight: '800', letterSpacing: -0.5 },
  subtitle: { color: '#C4B5FD', fontSize: 15, marginTop: 6 },

  card: {
    backgroundColor: '#FFFFFF', borderRadius: 22, padding: 22,
    shadowColor: '#000', shadowOpacity: 0.2, shadowRadius: 20, shadowOffset: { width: 0, height: 10 }, elevation: 8,
  },

  switchRow:       { flexDirection: 'row', backgroundColor: '#F3F0FF', borderRadius: 12, padding: 4, marginBottom: 18 },
  switchBtn:       { flex: 1, paddingVertical: 9, borderRadius: 9, alignItems: 'center' },
  switchBtnActive: { backgroundColor: '#7C3AED' },
  switchText:      { fontSize: 13, fontWeight: '700', color: '#7C3AED' },
  switchTextActive:{ color: '#FFFFFF' },

  label: { fontSize: 10, fontWeight: '800', color: '#7C3AED', letterSpacing: 1.5, marginBottom: 6, marginTop: 12 },
  input: {
    borderWidth: 1, borderColor: '#E5E7EB', borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12,
    fontSize: 15, color: '#1F2937', backgroundColor: '#FAFAFA',
  },

  error: { color: '#B91C1C', fontSize: 13, fontWeight: '600', marginTop: 14 },

  submit: {
    backgroundColor: '#7C3AED', borderRadius: 14, paddingVertical: 15, alignItems: 'center', marginTop: 22,
  },
  submitBusy: { opacity: 0.7 },
  submitText: { color: '#FFFFFF', fontSize: 16, fontWeight: '800' },

  altRow:  { marginTop: 18, alignItems: 'center' },
  altText: { fontSize: 13, color: '#6B7280' },
  altLink: { color: '#7C3AED', fontWeight: '700' },
});
