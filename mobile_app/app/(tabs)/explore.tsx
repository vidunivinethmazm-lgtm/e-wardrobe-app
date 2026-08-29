import React, { useState, useEffect } from 'react';
import {
  StyleSheet, View, Text, ScrollView, SafeAreaView,
  StatusBar, TouchableOpacity, TextInput,
} from 'react-native';
import { wardrobeStore, HistoryEntry } from '../store';

export default function HistoryScreen() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [drafts, setDrafts] = useState<Record<string, string>>({});

  useEffect(() => {
    setHistory([...wardrobeStore.history]);
    wardrobeStore.hydrate();               // pull saved history from the backend
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
                Your occasion searches will appear here.{'\n'}Head to Recommend to get started.
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
                    <Text style={styles.historyRecsLabel}>Top picks — rate them:</Text>
                    {entry.data.recommendations.slice(0, 3).map((r, i) => {
                      const fb = entry.feedback?.[r.outfit];
                      return (
                        <View key={i} style={styles.historyRecRow}>
                          <Text style={styles.historyRecNum}>{i + 1}.</Text>
                          <Text style={styles.historyRecName}>{r.outfit}</Text>
                          <View style={styles.historyFbBtns}>
                            <TouchableOpacity
                              onPress={() => wardrobeStore.setHistoryFeedback(entry.id, r.outfit, 'liked')}
                              style={[styles.fbBtn, fb === 'liked' && styles.fbBtnLikedActive]}
                            >
                              <Text style={styles.fbBtnIcon}>👍</Text>
                            </TouchableOpacity>
                            <TouchableOpacity
                              onPress={() => wardrobeStore.setHistoryFeedback(entry.id, r.outfit, 'skipped')}
                              style={[styles.fbBtn, fb === 'skipped' && styles.fbBtnSkippedActive]}
                            >
                              <Text style={styles.fbBtnIcon}>👎</Text>
                            </TouchableOpacity>
                          </View>
                        </View>
                      );
                    })}
                  </View>
                )}

                {/* ── Free-text feedback ── */}
                <View style={styles.noteBox}>
                  <Text style={styles.noteLabel}>YOUR FEEDBACK</Text>
                  {entry.note ? (
                    <View style={styles.noteSaved}>
                      <Text style={styles.noteSavedText}>“{entry.note}”</Text>
                      <View style={styles.noteSavedActions}>
                        <TouchableOpacity
                          onPress={() => {
                            setDrafts(d => ({ ...d, [entry.id]: entry.note ?? '' }));
                            wardrobeStore.setHistoryNote(entry.id, '');
                          }}
                        >
                          <Text style={styles.noteEdit}>Edit</Text>
                        </TouchableOpacity>
                        <TouchableOpacity onPress={() => wardrobeStore.setHistoryNote(entry.id, '')}>
                          <Text style={styles.noteRemove}>Remove</Text>
                        </TouchableOpacity>
                      </View>
                    </View>
                  ) : (
                    <>
                      <TextInput
                        style={styles.noteInput}
                        placeholder="Type your thoughts on these picks…"
                        placeholderTextColor="#9CA3AF"
                        multiline
                        value={drafts[entry.id] ?? ''}
                        onChangeText={t => setDrafts(d => ({ ...d, [entry.id]: t }))}
                      />
                      <TouchableOpacity
                        style={[styles.noteBtn, !(drafts[entry.id] ?? '').trim() && styles.noteBtnDisabled]}
                        disabled={!(drafts[entry.id] ?? '').trim()}
                        onPress={() => {
                          wardrobeStore.setHistoryNote(entry.id, drafts[entry.id] ?? '');
                          setDrafts(d => {
                            const next = { ...d };
                            delete next[entry.id];
                            return next;
                          });
                        }}
                      >
                        <Text style={styles.noteBtnText}>Add feedback</Text>
                      </TouchableOpacity>
                    </>
                  )}
                </View>
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
  historyRecRow:    { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 6 },
  historyRecNum:    { fontSize: 12, color: '#9CA3AF', width: 16 },
  historyRecName:   { fontSize: 13, color: '#374151', flex: 1, fontWeight: '500' },
  historyFbBtns:    { flexDirection: 'row', gap: 4 },
  fbBtn:            { paddingHorizontal: 7, paddingVertical: 3, borderRadius: 8, backgroundColor: '#F3F4F6' },
  fbBtnLikedActive:   { backgroundColor: '#DCFCE7', borderWidth: 1, borderColor: '#16A34A' },
  fbBtnSkippedActive: { backgroundColor: '#FEE2E2', borderWidth: 1, borderColor: '#EF4444' },
  fbBtnIcon:        { fontSize: 13 },

  noteBox:   { marginTop: 12 },
  noteLabel: { fontSize: 10, fontWeight: '700', color: '#7C3AED', letterSpacing: 1, marginBottom: 6 },
  noteInput: {
    minHeight: 64, borderWidth: 1, borderColor: '#E5E7EB', borderRadius: 12,
    padding: 10, fontSize: 13, color: '#374151', backgroundColor: '#FFF',
    textAlignVertical: 'top',
  },
  noteBtn: {
    marginTop: 8, alignSelf: 'flex-start', backgroundColor: '#7C3AED',
    paddingHorizontal: 16, paddingVertical: 8, borderRadius: 10,
  },
  noteBtnDisabled: { backgroundColor: '#C4B5FD' },
  noteBtnText:     { color: '#FFF', fontSize: 13, fontWeight: '700' },

  noteSaved:        { backgroundColor: '#F5F3FF', borderRadius: 12, padding: 10, borderLeftWidth: 3, borderLeftColor: '#7C3AED' },
  noteSavedText:    { fontSize: 13, color: '#4C1D95', fontStyle: 'italic', lineHeight: 19 },
  noteSavedActions: { flexDirection: 'row', gap: 16, marginTop: 8 },
  noteEdit:         { fontSize: 12, color: '#7C3AED', fontWeight: '700' },
  noteRemove:       { fontSize: 12, color: '#EF4444', fontWeight: '700' },
});
