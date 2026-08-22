import React, { useState, useEffect } from 'react';
import {
  StyleSheet, View, Text, ScrollView, SafeAreaView,
  StatusBar, TouchableOpacity,
} from 'react-native';
import { wardrobeStore, HistoryEntry, FeedbackMap } from '../store';

export default function HistoryScreen() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    setHistory([...wardrobeStore.history]);
    return wardrobeStore.subscribe(() => {
      setHistory([...wardrobeStore.history]);
    });
  }, []);

  // Collect all liked outfits across all history entries
  const likedOutfits = history.flatMap(h =>
    Object.entries(h.feedback ?? {})
      .filter(([, v]) => v === 'liked')
      .map(([name]) => ({ name, occasion: h.occasion, event_class: h.event_class }))
  );

  const EVENT_COLORS: Record<string, string> = {
    Wedding:     '#7C3AED',
    Party:       '#DB2777',
    Formal:      '#1D4ED8',
    Funeral:     '#374151',
    ColdOutdoor: '#0891B2',
    Casual:      '#16A34A',
  };

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="light-content" backgroundColor="#2D1B69" />

      <View style={styles.header}>
        <Text style={styles.headerEyebrow}>AI-POWERED FASHION</Text>
        <Text style={styles.headerTitle}>History</Text>
        <Text style={styles.headerSubtitle}>Your recent occasion searches</Text>
      </View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false}>

        {/* ── Liked outfits ── */}
        {likedOutfits.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionLabel}>❤️ LIKED OUTFITS</Text>
            {likedOutfits.map((item, i) => (
              <View key={i} style={styles.likedCard}>
                <Text style={styles.likedDot}>👍</Text>
                <View style={{ flex: 1 }}>
                  <Text style={styles.likedName}>{item.name}</Text>
                  <Text style={styles.likedOccasion}>{item.occasion} · {item.event_class}</Text>
                </View>
              </View>
            ))}
          </View>
        )}

        {/* ── Search history ── */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionLabel}>🕐 SEARCH HISTORY</Text>
            {history.length > 0 && (
              <TouchableOpacity onPress={() => wardrobeStore.clearHistory()}>
                <Text style={styles.clearText}>Clear all</Text>
              </TouchableOpacity>
            )}
          </View>

          {history.length === 0 ? (
            <View style={styles.empty}>
              <Text style={styles.emptyIcon}>🔍</Text>
              <Text style={styles.emptyTitle}>No searches yet</Text>
              <Text style={styles.emptyText}>
                Your occasion searches will appear here.{'\n'}Head to Home to get started.
              </Text>
            </View>
          ) : (
            history.map(entry => (
              <View key={entry.id} style={styles.historyCard}>
                <View style={styles.historyTop}>
                  <View style={[styles.eventBadge, { backgroundColor: EVENT_COLORS[entry.event_class] ?? '#7C3AED' }]}>
                    <Text style={styles.eventBadgeText}>{entry.event_class}</Text>
                  </View>
                  <Text style={styles.historyTime}>{entry.time}</Text>
                </View>

                <Text style={styles.historyOccasion}>{entry.occasion}</Text>

                <View style={styles.historyMeta}>
                  <Text style={styles.historyMetaText}>📍 {entry.location}</Text>
                  <Text style={styles.historyMetaText}>🌤 {entry.weather}</Text>
                </View>

                {entry.data.recommendations.length > 0 && (
                  <View style={styles.historyRecs}>
                    <Text style={styles.historyRecsLabel}>Top picks:</Text>
                    {entry.data.recommendations.slice(0, 3).map((r, i) => {
                      const fb = entry.feedback?.[r.outfit];
                      return (
                        <View key={i} style={styles.historyRecRow}>
                          <Text style={styles.historyRecNum}>{i + 1}.</Text>
                          <Text style={styles.historyRecName}>{r.outfit}</Text>
                          {r.price !== undefined && (
                            <Text style={styles.historyRecPrice}>Rs. {r.price.toLocaleString()}</Text>
                          )}
                          {fb && (
                            <Text style={styles.historyRecFb}>{fb === 'liked' ? '👍' : '👎'}</Text>
                          )}
                        </View>
                      );
                    })}
                  </View>
                )}
              </View>
            ))
          )}
        </View>

        <View style={{ height: 40 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe:   { flex: 1, backgroundColor: '#2D1B69' },
  scroll: { flex: 1, backgroundColor: '#F3F0FF' },

  header: { backgroundColor: '#2D1B69', paddingHorizontal: 24, paddingTop: 40, paddingBottom: 36 },
  headerEyebrow:  { color: '#A78BFA', fontSize: 11, fontWeight: '700', letterSpacing: 3, marginBottom: 6 },
  headerTitle:    { color: '#FFFFFF', fontSize: 38, fontWeight: '800', letterSpacing: -1 },
  headerSubtitle: { color: '#C4B5FD', fontSize: 14, marginTop: 6, lineHeight: 20 },

  section:       { margin: 16, marginBottom: 0 },
  sectionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
  sectionLabel:  { fontSize: 10, fontWeight: '800', color: '#7C3AED', letterSpacing: 2, marginBottom: 10 },
  clearText:     { fontSize: 12, color: '#EF4444', fontWeight: '600' },

  likedCard: { flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: '#FFF', borderRadius: 14, padding: 14, marginBottom: 8, borderLeftWidth: 3, borderLeftColor: '#10B981' },
  likedDot:      { fontSize: 18 },
  likedName:     { fontSize: 14, color: '#1F2937', fontWeight: '600' },
  likedOccasion: { fontSize: 11, color: '#6B7280', marginTop: 2 },

  empty:      { backgroundColor: '#FFF', borderRadius: 20, padding: 32, alignItems: 'center', marginBottom: 16 },
  emptyIcon:  { fontSize: 40, marginBottom: 12 },
  emptyTitle: { fontSize: 18, fontWeight: '800', color: '#1F2937', marginBottom: 8 },
  emptyText:  { fontSize: 14, color: '#6B7280', textAlign: 'center', lineHeight: 22 },

  historyCard: { backgroundColor: '#FFF', borderRadius: 20, padding: 16, marginBottom: 14, shadowColor: '#6D28D9', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.07, shadowRadius: 8, elevation: 3 },
  historyTop:  { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  eventBadge:      { borderRadius: 20, paddingHorizontal: 12, paddingVertical: 4 },
  eventBadgeText:  { color: '#fff', fontSize: 11, fontWeight: '700' },
  historyTime:     { fontSize: 12, color: '#9CA3AF' },
  historyOccasion: { fontSize: 16, fontWeight: '700', color: '#1F2937', marginBottom: 8 },
  historyMeta:     { flexDirection: 'row', gap: 12, marginBottom: 10 },
  historyMetaText: { fontSize: 12, color: '#6B7280' },

  historyRecs:      { backgroundColor: '#F9FAFB', borderRadius: 12, padding: 12 },
  historyRecsLabel: { fontSize: 10, fontWeight: '700', color: '#7C3AED', letterSpacing: 1, marginBottom: 8 },
  historyRecRow:    { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 4 },
  historyRecNum:    { fontSize: 12, color: '#9CA3AF', width: 16 },
  historyRecName:   { fontSize: 13, color: '#374151', flex: 1, fontWeight: '500' },
  historyRecPrice:  { fontSize: 12, color: '#92400E', fontWeight: '600' },
  historyRecFb:     { fontSize: 14 },
});
