import React, { useState } from 'react';
import {
  View, Text, Image, ScrollView,
  TouchableOpacity, StyleSheet,
} from 'react-native';
import OutfitCard from '../components/OutfitCard';
import { colors, font, radius } from '../constants/theme';

interface Props {
  result: any;
  selfieUri: string | null;
  onReset: () => void;
}

export default function TryOnScreen({ result, selfieUri, onReset }: Props) {
  const [selectedOutfit, setSelectedOutfit] = useState(0);
  const [tab, setTab] = useState<'outfits' | 'info'>('outfits');

  const sp   = result?.sizingProfile;
  const fa   = result?.faceAnalysis;
  const sc   = result?.renderPayload?.scaleParams;
  const recs = result?.recommendations ?? [];

  return (
    <View style={styles.container}>
      {/* Avatar / selfie header */}
      <View style={styles.avatarCard}>
        {selfieUri
          ? <Image source={{ uri: selfieUri }} style={styles.selfieCircle} />
          : <View style={[styles.selfieCircle, { backgroundColor: colors.card, alignItems:'center', justifyContent:'center' }]}>
              <Text style={{ fontSize: 40 }}>🧍</Text>
            </View>
        }
        <View style={styles.avatarInfo}>
          <Text style={styles.avatarName}>eWardrobeAI Avatar</Text>
          <Text style={styles.avatarSub}>
            Size: <Text style={styles.highlight}>{sp?.standardSize ?? '—'}</Text>
            {'  ·  '}
            <Text style={styles.highlight}>{sp?.bodyType?.replace(/_/g,' ') ?? '—'}</Text>
          </Text>
          {recs.length > 0 && (
            <Text style={styles.avatarSub}>
              Outfit: <Text style={styles.highlight}>{recs[selectedOutfit]?.name}</Text>
            </Text>
          )}
        </View>
      </View>

      {/* Tab bar */}
      <View style={styles.tabBar}>
        <TouchableOpacity
          style={[styles.tab, tab === 'outfits' && styles.tabActive]}
          onPress={() => setTab('outfits')}
        >
          <Text style={[styles.tabText, tab === 'outfits' && styles.tabTextActive]}>
            👔 Outfits ({recs.length})
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, tab === 'info' && styles.tabActive]}
          onPress={() => setTab('info')}
        >
          <Text style={[styles.tabText, tab === 'info' && styles.tabTextActive]}>
            📊 Analysis
          </Text>
        </TouchableOpacity>
      </View>

      {tab === 'outfits' ? (
        <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
          {recs.length === 0
            ? <Text style={styles.empty}>No outfits available for this profile.</Text>
            : recs.map((rec: any, i: number) => (
                <OutfitCard
                  key={i}
                  outfit={rec}
                  index={i}
                  total={recs.length}
                  active={selectedOutfit === i}
                  onPress={() => setSelectedOutfit(i)}
                />
              ))
          }
          <TouchableOpacity style={styles.resetBtn} onPress={onReset}>
            <Text style={styles.resetText}>🔄  New Try-On</Text>
          </TouchableOpacity>
        </ScrollView>
      ) : (
        <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
          <InfoSection title="Sizing Profile">
            <KV k="Standard Size"  v={sp?.standardSize} highlight />
            <KV k="Body Type"      v={sp?.bodyType?.replace(/_/g,' ')} />
            <KV k="Waist–Hip Ratio" v={sp?.waistHipRatio?.toFixed(3)} />
            <KV k="Torso Length"   v={sp?.torsoLengthCm ? `${sp.torsoLengthCm.toFixed(1)} cm` : undefined} />
          </InfoSection>

          <InfoSection title="Face Analysis">
            <KV k="Dense Landmarks"   v={fa?.landmark468Count} />
            <KV k="Sparse Keypoints"  v={fa?.landmark15Count} />
            <KV k="Inter-Eye Dist"    v={fa?.interEyeDistPx ? `${fa.interEyeDistPx.toFixed(1)} px` : undefined} />
            <KV k="Head Yaw"          v={fa?.yawDeg ? `${fa.yawDeg.toFixed(1)}°` : undefined} />
            <KV k="Face Texture"      v={fa?.hasFaceTexture ? '✅ Applied' : '⚠️ None'} />
          </InfoSection>

          <InfoSection title="Avatar Scale">
            <KV k="Height"    v={sc?.globalY   ? `${(sc.globalY   * 100).toFixed(0)}%` : undefined} />
            <KV k="Chest"     v={sc?.chestX    ? `${(sc.chestX    * 100).toFixed(0)}%` : undefined} />
            <KV k="Waist"     v={sc?.waistX    ? `${(sc.waistX    * 100).toFixed(0)}%` : undefined} />
            <KV k="Shoulders" v={sc?.shoulderX ? `${(sc.shoulderX * 100).toFixed(0)}%` : undefined} />
            <KV k="Legs"      v={sc?.legY      ? `${(sc.legY      * 100).toFixed(0)}%` : undefined} />
          </InfoSection>

          <TouchableOpacity style={styles.resetBtn} onPress={onReset}>
            <Text style={styles.resetText}>🔄  New Try-On</Text>
          </TouchableOpacity>
        </ScrollView>
      )}
    </View>
  );
}

function InfoSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <View style={infoStyles.section}>
      <Text style={infoStyles.sectionTitle}>{title.toUpperCase()}</Text>
      {children}
    </View>
  );
}

function KV({ k, v, highlight }: { k: string; v?: string | number; highlight?: boolean }) {
  return (
    <View style={infoStyles.row}>
      <Text style={infoStyles.key}>{k}</Text>
      <Text style={[infoStyles.val, highlight && infoStyles.valHighlight]}>
        {v ?? '—'}
      </Text>
    </View>
  );
}

const infoStyles = StyleSheet.create({
  section: {
    backgroundColor: colors.card, borderWidth: 1, borderColor: colors.border,
    borderRadius: radius.md, padding: 14, marginBottom: 12,
  },
  sectionTitle: { fontSize: font.xs, color: colors.muted, fontWeight: '700', letterSpacing: 1, marginBottom: 10 },
  row: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 5,
         borderBottomWidth: 1, borderBottomColor: 'rgba(255,255,255,0.04)' },
  key: { fontSize: font.sm, color: colors.muted },
  val: { fontSize: font.sm, fontWeight: '700', color: colors.text },
  valHighlight: { color: colors.accent },
});

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },

  avatarCard: {
    flexDirection: 'row', alignItems: 'center', gap: 14,
    padding: 16, backgroundColor: colors.card,
    borderBottomWidth: 1, borderBottomColor: colors.border,
  },
  selfieCircle: { width: 64, height: 64, borderRadius: 32, borderWidth: 2, borderColor: colors.accent },
  avatarInfo:   { flex: 1 },
  avatarName:   { fontSize: font.md, fontWeight: '700', color: colors.text },
  avatarSub:    { fontSize: font.xs, color: colors.muted, marginTop: 3 },
  highlight:    { color: colors.accent, fontWeight: '700' },

  tabBar: {
    flexDirection: 'row', backgroundColor: colors.surface,
    borderBottomWidth: 1, borderBottomColor: colors.border,
  },
  tab: { flex: 1, paddingVertical: 12, alignItems: 'center' },
  tabActive: { borderBottomWidth: 2, borderBottomColor: colors.accent },
  tabText:       { fontSize: font.sm, color: colors.muted, fontWeight: '600' },
  tabTextActive: { color: colors.accent },

  scroll:        { flex: 1 },
  scrollContent: { padding: 16, paddingBottom: 40 },
  empty: { color: colors.muted, textAlign: 'center', marginTop: 40, fontSize: font.sm },

  resetBtn: {
    marginTop: 8, paddingVertical: 14, alignItems: 'center',
    backgroundColor: colors.card, borderRadius: radius.md,
    borderWidth: 1, borderColor: colors.border,
  },
  resetText: { color: colors.text, fontWeight: '700', fontSize: font.md },
});
