import { useRouter, useSegments } from 'expo-router';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useAuth } from '@/app/auth';

/**
 * Slim app bar shown above every tab screen. Carries the brand, a greeting
 * for the signed-in account, and the quick links that don't need a full
 * bottom-tab slot (Trends for now).
 */
export function TopBar() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const segments = useSegments();
  const auth = useAuth();

  const firstName = (auth.user?.name || auth.user?.email?.split('@')[0] || 'there').split(' ')[0];
  const avatar = auth.user?.profile?.avatar || '👤';
  const onTrends = segments[segments.length - 1] === 'trends';

  return (
    <View style={[styles.bar, { paddingTop: insets.top + 8 }]}>
      <Pressable style={styles.brand} onPress={() => router.push('/(tabs)/profile')} hitSlop={8}>
        {avatar.startsWith('data:')
          ? <Image source={{ uri: avatar }} style={styles.avatarImg} />
          : <Text style={styles.avatar}>{avatar}</Text>}
        <View>
          <Text style={styles.hi}>Hi, {firstName}</Text>
          <Text style={styles.appName}>E-Wardrobe AI</Text>
        </View>
      </Pressable>

      <View style={styles.links}>
        <Pressable
          style={[styles.pill, onTrends && styles.pillActive]}
          onPress={() => router.push('/(tabs)/trends')}
          hitSlop={6}
        >
          <Text style={[styles.pillText, onTrends && styles.pillTextActive]}>📈  Trends</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  bar: {
    backgroundColor: '#2D1B69',
    paddingHorizontal: 16,
    paddingBottom: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  brand:     { flexDirection: 'row', alignItems: 'center', gap: 10 },
  avatar:    { fontSize: 26 },
  avatarImg: { width: 30, height: 30, borderRadius: 15, borderWidth: 1, borderColor: 'rgba(196,181,253,0.5)' },
  hi:     { color: '#C4B5FD', fontSize: 11, fontWeight: '600' },
  appName:{ color: '#FFFFFF', fontSize: 16, fontWeight: '800', letterSpacing: -0.3 },

  links: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  pill: {
    backgroundColor: 'rgba(255,255,255,0.12)',
    borderWidth: 1,
    borderColor: 'rgba(196,181,253,0.35)',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 999,
  },
  pillActive:     { backgroundColor: '#7C3AED', borderColor: '#7C3AED' },
  pillText:       { color: '#EDE9FE', fontSize: 13, fontWeight: '700' },
  pillTextActive: { color: '#FFFFFF' },
});
