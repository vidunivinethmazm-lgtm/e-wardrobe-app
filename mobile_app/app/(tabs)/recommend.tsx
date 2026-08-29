import React, { useState, useRef, useEffect } from 'react';
import {
  StyleSheet, TouchableOpacity, ActivityIndicator,
  View, TextInput, Alert, Text, ScrollView, SafeAreaView, StatusBar,
  Animated, PanResponder, Share,
} from 'react-native';
import { Image } from 'expo-image';
import * as Location from 'expo-location';
import { wardrobeStore, HistoryEntry, RecommendationDetail, ApiResponse, FeedbackMap } from '../store';

// ── Constants ────────────────────────────────────────────────────────────────

const RANKS = ['🥇', '🥈', '🥉'];
const SWIPE_THRESHOLD = 100;
const CACHE_TTL = 5 * 60 * 1000;

// Integrated backend mounts the recommendation feature under a prefix.
const RECOMMEND_PATH = '/recommendation/recommend';

// Used when the backend saves an item without a photo, so cards still render.
const PLACEHOLDER_IMG =
  'https://images.unsplash.com/photo-1489987707025-afc232f7ea0f?q=80&w=500';

const COLOR_DOTS: Record<string, string> = {
  white: '#E5E7EB', red: '#EF4444', blue: '#3B82F6', black: '#374151',
  navy: '#1E3A5F', pink: '#F472B6', brown: '#92400E', olive: '#65A30D',
  multicolor: '#7C3AED', gold: '#D97706', purple: '#9333EA', teal: '#0D9488',
  beige: '#D4B48C',
};

const SL_CITIES = [
  'nuwara eliya','ella','horton plains','knuckles','haputale','bandarawela',
  'kandy','matara','galle','colombo','negombo','trincomalee','batticaloa',
  'jaffna','anuradhapura','polonnaruwa','sigiriya','dambulla','ratnapura',
  'hambantota','tangalle','weligama','mirissa','unawatuna','hikkaduwa',
  'bentota','kalutara','chilaw','puttalam','badulla','hatton','kurunegala',
];

// ── Response mapping (integrated backend -> store types) ─────────────────────

function toApiResponse(parsed: any, intent: string, city: string, weatherMain: string): ApiResponse {
  const recommendations: RecommendationDetail[] = (parsed?.recommendations ?? []).map((r: any) => ({
    outfit: r.outfit,
    item_id: r.item_id,
    confidence: typeof r.confidence === 'string'
      ? r.confidence
      : `${Math.round(Number(r.confidence ?? r.score ?? 0) * 100)}%`,
    fabric: r.fabric ?? '',
    color: r.color,
    price: r.price,
    category: r.category,
    image_url: r.image_url || PLACEHOLDER_IMG,
    reason: r.reason,
    combination: r.combination,
    score: r.score,
  }));
  return {
    event_class: parsed?.event_class ?? intent,
    location_detected: parsed?.location_detected ?? city,
    weather: parsed?.weather ?? weatherMain,
    from_cache: parsed?.from_cache,
    recommendations,
  };
}

// ── AnimatedCard (swipe left = skip, swipe right = like) ────────────────────

function AnimatedCard({ children, onLike, onSkip, feedback }: {
  children: React.ReactNode;
  onLike: () => void;
  onSkip: () => void;
  feedback?: 'liked' | 'skipped';
}) {
  const translateX = useRef(new Animated.Value(0)).current;

  const panResponder = useRef(PanResponder.create({
    onMoveShouldSetPanResponder: (_, { dx, dy }) =>
      !feedback && Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 8,
    onPanResponderMove: (_, { dx }) => {
      if (!feedback) translateX.setValue(dx);
    },
    onPanResponderRelease: (_, { dx }) => {
      if (dx > SWIPE_THRESHOLD) {
        Animated.spring(translateX, { toValue: 0, useNativeDriver: true }).start();
        onLike();
      } else if (dx < -SWIPE_THRESHOLD) {
        Animated.spring(translateX, { toValue: 0, useNativeDriver: true }).start();
        onSkip();
      } else {
        Animated.spring(translateX, { toValue: 0, useNativeDriver: true }).start();
      }
    },
  })).current;

  const likeOpacity = translateX.interpolate({
    inputRange: [0, SWIPE_THRESHOLD], outputRange: [0, 0.9], extrapolate: 'clamp',
  });
  const skipOpacity = translateX.interpolate({
    inputRange: [-SWIPE_THRESHOLD, 0], outputRange: [0.9, 0], extrapolate: 'clamp',
  });
  const rotate = translateX.interpolate({
    inputRange: [-SWIPE_THRESHOLD, SWIPE_THRESHOLD], outputRange: ['-4deg', '4deg'], extrapolate: 'clamp',
  });

  return (
    <Animated.View {...panResponder.panHandlers} style={{ transform: [{ translateX }, { rotate }] }}>
      <Animated.View pointerEvents="none" style={[styles.swipeOverlay, styles.likeOverlay, { opacity: likeOpacity }]}>
        <Text style={styles.swipeOverlayText}>👍 LIKE</Text>
      </Animated.View>
      <Animated.View pointerEvents="none" style={[styles.swipeOverlay, styles.skipOverlay, { opacity: skipOpacity }]}>
        <Text style={styles.swipeOverlayText}>👎 SKIP</Text>
      </Animated.View>
      {children}
    </Animated.View>
  );
}

// ── Recommendations Screen ───────────────────────────────────────────────────

export default function HomeScreen() {
  const [data, setData]               = useState<ApiResponse | null>(null);
  const [loading, setLoading]         = useState(false);
  const [userIntent, setUserIntent]   = useState('');
  const [backendStatus, setBackendStatus] = useState('');
  const [errorMessage, setErrorMessage]   = useState('');
  const [history, setHistory]         = useState<HistoryEntry[]>([]);
  const [feedback, setFeedback]       = useState<FeedbackMap>({});

  const cache = useRef<Map<string, { data: ApiResponse; ts: number }>>(new Map());
  const backendHosts = ['10.0.2.2', '10.123.244.242', '127.0.0.1', 'localhost'];

  // Sync with store
  useEffect(() => {
    const unsub = wardrobeStore.subscribe(() => {
      setHistory([...wardrobeStore.history]);
      setFeedback({ ...wardrobeStore.feedback });
    });
    return unsub;
  }, []);

  // ── Local fallback wardrobe ───────────────────────────────────────────────
  const localWardrobe: RecommendationDetail[] = [
    {
      outfit: 'White Linen Summer Dress', fabric: 'Linen', color: 'white', price: 3500, category: 'dress',
      image_url: 'https://images.unsplash.com/photo-1595777457583-95e059d581b8?q=80&w=500',
      confidence: '87%',
      reason: 'Linen is breathable and lightweight · NLP model: Linen suits casual occasions · GNN: high style compatibility',
      combination: 'Pair with strappy sandals and a small clutch',
    },
    {
      outfit: 'Silk Banarasi Saree', fabric: 'Silk', color: 'red', price: 12000, category: 'saree',
      image_url: 'https://images.unsplash.com/photo-1610030469983-98e059d581b8?q=80&w=500',
      confidence: '91%',
      reason: 'Traditional choice for Sri Lankan weddings · Red is an auspicious colour for weddings · GNN: high style compatibility',
      combination: 'Complete the look with Woolen Pashmina Shawl',
    },
    {
      outfit: 'Cotton Embroidered Kurti', fabric: 'Cotton', color: 'blue', price: 2500, category: 'top',
      image_url: 'https://images.unsplash.com/photo-1581044777550-4cfa60707c03?q=80&w=500',
      confidence: '83%',
      reason: 'Cotton keeps you cool in tropical heat · Versatile for casual and semi-formal events · GNN: moderate style compatibility',
      combination: 'Pair with Linen Tailored Trouser Suit for a polished finish',
    },
  ];

  // ── Cache helpers ─────────────────────────────────────────────────────────
  const getCached = (key: string): ApiResponse | null => {
    const entry = cache.current.get(key);
    if (entry && Date.now() - entry.ts < CACHE_TTL) return { ...entry.data, from_cache: true };
    return null;
  };
  const setCache = (key: string, d: ApiResponse) => cache.current.set(key, { data: d, ts: Date.now() });

  // ── Feedback ──────────────────────────────────────────────────────────────
  const handleFeedback = (outfit: string, action: 'liked' | 'skipped') => {
    const updated = { ...feedback, [outfit]: action };
    setFeedback(updated);
    wardrobeStore.setFeedback(outfit, action);
    wardrobeStore.saveCurrentFeedbackToLatestHistory(updated);
  };

  // ── Share ─────────────────────────────────────────────────────────────────
  const handleShare = async (item: RecommendationDetail, eventClass: string) => {
    try {
      await Share.share({
        message:
          `👗 E-Wardrobe AI Recommendation\n\n` +
          `✨ ${item.outfit}\n` +
          `🧵 Fabric: ${item.fabric}\n` +
          (item.color ? `🎨 Colour: ${item.color}\n` : '') +
          `🎯 ${item.confidence} match for ${eventClass}\n\n` +
          `💡 ${(item.reason ?? '').split(' · ')[0]}\n` +
          (item.combination ? `👗 Style tip: ${item.combination}` : ''),
      });
    } catch { /* user dismissed share sheet */ }
  };

  // ── Main API call ─────────────────────────────────────────────────────────
  const getMatchmaking = async (overrideIntent?: string) => {
    const intent = overrideIntent ?? userIntent;
    if (!intent.trim()) return Alert.alert('Input Needed', 'Please describe your occasion.');
    if (overrideIntent) setUserIntent(overrideIntent);

    setLoading(true);
    setData(null);
    wardrobeStore.saveCurrentFeedbackToLatestHistory(feedback);
    setFeedback({});
    wardrobeStore.resetFeedback();
    setBackendStatus('');
    setErrorMessage('');

    try {
      let latitude = 6.9271, longitude = 79.8612, locationSource = 'Default (Colombo)';
      const lowerIntent = intent.toLowerCase();
      const detectedDestination = SL_CITIES.find(c => lowerIntent.includes(c)) ?? null;

      if (!detectedDestination) {
        try {
          const { status } = await Location.requestForegroundPermissionsAsync();
          if (status === 'granted') {
            const locationPromise = Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Low });
            const timeoutPromise  = new Promise<null>((_, reject) => setTimeout(() => reject(new Error('GPS timeout')), 3000));
            const loc = await Promise.race([locationPromise, timeoutPromise]) as Awaited<typeof locationPromise>;
            latitude = loc.coords.latitude;
            longitude = loc.coords.longitude;
            locationSource = 'GPS';
          }
        } catch { setBackendStatus('GPS unavailable, using default location (Colombo)'); }
      }

      const API_KEY   = '64d7c88e4cd6807d9121d8611ed30457';
      const weatherUrl = detectedDestination
        ? `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(detectedDestination)},LK&appid=${API_KEY}&units=metric`
        : `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&appid=${API_KEY}&units=metric`;

      const wCtrl = new AbortController();
      const wTimer = setTimeout(() => wCtrl.abort(), 5000);
      const weatherRes  = await fetch(weatherUrl, { signal: wCtrl.signal });
      clearTimeout(wTimer);
      const weatherData = await weatherRes.json();
      if (!weatherRes.ok || !weatherData?.weather?.[0]) throw new Error('Weather API failed');

      const city        = weatherData.name || detectedDestination || locationSource;
      const weatherMain = weatherData.weather[0].main;
      const humidity    = weatherData.main.humidity;
      const temperature = weatherData.main.temp;

      setBackendStatus(`📍 ${city} — ${temperature.toFixed(0)}°C, ${weatherMain}`);

      const cacheKey = `${intent.toLowerCase().trim()}_${city.toLowerCase()}`;
      const cached = getCached(cacheKey);
      if (cached) {
        setData(cached);
        wardrobeStore.addHistory({
          id: Date.now().toString(), occasion: intent,
          event_class: cached.event_class, location: city,
          weather: weatherMain, time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
          data: cached,
        });
        setBackendStatus(`⚡ ${city} — ${temperature.toFixed(0)}°C (from cache)`);
        return;
      }

      const query = `user_input=${encodeURIComponent(intent)}&city=${encodeURIComponent(city)}&weather=${encodeURIComponent(weatherMain)}&humidity=${humidity}&temperature=${temperature}`;
      let result: ApiResponse | null = null, lastError: string | null = null;

      for (const host of backendHosts) {
        const ctrl  = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 5000);
        try {
          const response = await fetch(`http://${host}:8000${RECOMMEND_PATH}?${query}`, { signal: ctrl.signal });
          clearTimeout(timer);
          if (!response.ok) { lastError = `${host}: ${response.status}`; continue; }
          const parsed = await response.json();
          if (parsed?.recommendations?.length) {
            const normalized = toApiResponse(parsed, intent, city, weatherMain);
            result = normalized;
            setCache(cacheKey, normalized);
            setBackendStatus(`✅ Connected — ${city}`);
            break;
          }
          lastError = `${host}: empty response`;
        } catch (e: any) { clearTimeout(timer); lastError = `${host}: ${e.message}`; }
      }

      if (result) {
        setData(result);
        wardrobeStore.addHistory({
          id: Date.now().toString(), occasion: intent,
          event_class: result.event_class, location: city,
          weather: weatherMain, time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
          data: result,
        });
      } else {
        const msg = lastError || 'No backend reachable.';
        setErrorMessage(msg);
        throw new Error(msg);
      }
    } catch {
      const fallback: ApiResponse = {
        event_class: userIntent || 'Casual', location_detected: 'Local',
        weather: 'Unknown', recommendations: localWardrobe,
      };
      setData(fallback);
      wardrobeStore.addHistory({
        id: Date.now().toString(), occasion: intent ?? userIntent,
        event_class: fallback.event_class, location: 'Local',
        weather: 'Unknown', time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        data: fallback,
      });
      Alert.alert('Offline Mode', 'Using local recommendations.');
    } finally { setLoading(false); }
  };

  const likedOutfits = Object.entries(feedback).filter(([, v]) => v === 'liked').map(([k]) => k);

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="light-content" backgroundColor="#2D1B69" />
      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false}>

        {/* ── Header ── */}
        <View style={styles.header}>
          <Text style={styles.headerEyebrow}>AI-POWERED FASHION</Text>
          <Text style={styles.headerTitle}>E-Wardrobe</Text>
          <Text style={styles.headerSubtitle}>Smart outfit recommendations for every occasion</Text>
        </View>

        {/* ── Search Section ── */}
        <View style={styles.searchSection}>

          {/* Recent searches */}
          {history.length > 0 && (
            <View style={styles.recentBlock}>
              <Text style={styles.sectionLabel}>RECENT SEARCHES</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                <View style={styles.chipRow}>
                  {history.slice(0, 5).map(h => (
                    <TouchableOpacity key={h.id} style={styles.chip} onPress={() => getMatchmaking(h.occasion)}>
                      <Text style={styles.chipText} numberOfLines={1}>{h.occasion}</Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </ScrollView>
            </View>
          )}

          {/* Occasion input */}
          <Text style={[styles.sectionLabel, { marginTop: 16 }]}>{"WHAT'S YOUR OCCASION?"}</Text>
          <TextInput
            style={styles.mainInput}
            placeholder="e.g. Wedding in Matara, Trip to Ella…"
            value={userIntent}
            onChangeText={setUserIntent}
            placeholderTextColor="#9CA3AF"
          />

          {backendStatus ? (
            <View style={[styles.statusPill, data?.from_cache && styles.cachePill]}>
              <Text style={styles.statusPillText}>{backendStatus}</Text>
            </View>
          ) : null}
          {errorMessage ? <Text style={styles.errorText}>{errorMessage}</Text> : null}

          <TouchableOpacity
            style={[styles.button, loading && styles.buttonDisabled]}
            onPress={() => getMatchmaking()}
            disabled={loading}
          >
            {loading
              ? <ActivityIndicator color="#fff" />
              : <Text style={styles.buttonText}>✨  Get AI Recommendations</Text>}
          </TouchableOpacity>
        </View>

        {/* ── Results ── */}
        {data && (
          <View style={styles.resultsSection}>
            <View style={styles.resultsTitleRow}>
              <Text style={styles.sectionLabel}>TOP PICKS FOR YOU</Text>
              {data.from_cache && (
                <View style={styles.cachedBadge}>
                  <Text style={styles.cachedBadgeText}>⚡ CACHED</Text>
                </View>
              )}
            </View>

            <View style={styles.eventRow}>
              <View style={styles.eventBadge}>
                <Text style={styles.eventBadgeText}>{data.event_class}</Text>
              </View>
              <View style={styles.weatherBadge}>
                <Text style={styles.weatherBadgeText}>🌤 {data.weather}</Text>
              </View>
              <View style={styles.locationBadge}>
                <Text style={styles.locationBadgeText}>📍 {data.location_detected}</Text>
              </View>
            </View>

            <Text style={styles.swipeHint}>← swipe card to skip  ·  swipe right to like →</Text>

            {data.recommendations.map((item, index) => (
              <AnimatedCard
                key={item.item_id ?? `${item.outfit}-${index}`}
                onLike={() => handleFeedback(item.outfit, 'liked')}
                onSkip={() => handleFeedback(item.outfit, 'skipped')}
                feedback={feedback[item.outfit]}
              >
                <View style={[
                  styles.card,
                  feedback[item.outfit] === 'liked' && styles.cardLiked,
                  feedback[item.outfit] === 'skipped' && styles.cardSkipped,
                ]}>

                  {/* Image */}
                  <View style={styles.imageWrapper}>
                    <Image source={{ uri: item.image_url }} style={styles.cardImage} contentFit="cover" transition={400} />
                    <View style={styles.rankBadge}>
                      <Text style={styles.rankEmoji}>{RANKS[index] ?? `#${index + 1}`}</Text>
                      <Text style={styles.rankText}>RANK {index + 1}</Text>
                    </View>
                    <View style={styles.confidenceBadge}>
                      <Text style={styles.confidenceText}>{item.confidence}</Text>
                      <Text style={styles.confidenceLabel}>MATCH</Text>
                    </View>
                    {feedback[item.outfit] && (
                      <View style={[
                        styles.feedbackBadge,
                        feedback[item.outfit] === 'liked' ? styles.feedbackLiked : styles.feedbackSkipped,
                      ]}>
                        <Text style={styles.feedbackBadgeText}>
                          {feedback[item.outfit] === 'liked' ? '👍 LIKED' : '👎 SKIPPED'}
                        </Text>
                      </View>
                    )}
                  </View>

                  {/* Card body */}
                  <View style={styles.cardBody}>
                    <Text style={styles.outfitName}>{item.outfit}</Text>

                    {/* Tags */}
                    <View style={styles.tagRow}>
                      <View style={styles.fabricTag}>
                        <Text style={styles.fabricTagText}>🧵 {item.fabric}</Text>
                      </View>
                      {item.color && (
                        <View style={styles.colorTag}>
                          <View style={[styles.colorDot, { backgroundColor: COLOR_DOTS[item.color] ?? '#7C3AED' }]} />
                          <Text style={styles.colorTagText}>{item.color}</Text>
                        </View>
                      )}
                    </View>

                    {/* Why this suits you */}
                    <View style={styles.reasonBox}>
                      <Text style={styles.reasonTitle}>💡 WHY THIS SUITS YOU</Text>
                      {(item.reason ?? `NLP: ${item.fabric} evaluated · GNN: style compatibility assessed`)
                        .split(' · ').map((r, i) => (
                          <View key={i} style={styles.reasonRow}>
                            <Text style={styles.reasonDot}>•</Text>
                            <Text style={styles.reasonText}>{r}</Text>
                          </View>
                        ))}
                    </View>

                    {/* Outfit combination */}
                    {item.combination && (
                      <View style={styles.comboBox}>
                        <Text style={styles.comboTitle}>👗 COMPLETE THE LOOK</Text>
                        <Text style={styles.comboText}>{item.combination}</Text>
                      </View>
                    )}

                    {/* Action row */}
                    <View style={styles.actionRow}>
                      <TouchableOpacity
                        style={[styles.actionBtn, styles.skipBtn, feedback[item.outfit] === 'skipped' && styles.actionBtnActive]}
                        onPress={() => handleFeedback(item.outfit, 'skipped')}
                      >
                        <Text style={styles.actionBtnEmoji}>👎</Text>
                        <Text style={styles.actionBtnLabel}>Skip</Text>
                      </TouchableOpacity>

                      <TouchableOpacity style={styles.shareBtn} onPress={() => handleShare(item, data.event_class)}>
                        <Text style={styles.shareBtnText}>↗ Share</Text>
                      </TouchableOpacity>

                      <TouchableOpacity
                        style={[styles.actionBtn, styles.likeBtn, feedback[item.outfit] === 'liked' && styles.actionBtnActive]}
                        onPress={() => handleFeedback(item.outfit, 'liked')}
                      >
                        <Text style={styles.actionBtnEmoji}>👍</Text>
                        <Text style={styles.actionBtnLabel}>Like</Text>
                      </TouchableOpacity>
                    </View>
                  </View>
                </View>
              </AnimatedCard>
            ))}

            {/* Liked summary */}
            {likedOutfits.length > 0 && (
              <View style={styles.likedSummary}>
                <Text style={styles.likedTitle}>❤️ YOUR LIKED OUTFITS</Text>
                {likedOutfits.map(name => (
                  <Text key={name} style={styles.likedItem}>• {name}</Text>
                ))}
              </View>
            )}
          </View>
        )}

        <View style={{ height: 40 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

// ── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  safe:   { flex: 1, backgroundColor: '#2D1B69' },
  scroll: { flex: 1, backgroundColor: '#F3F0FF' },

  header: { backgroundColor: '#2D1B69', paddingHorizontal: 24, paddingTop: 40, paddingBottom: 36 },
  headerEyebrow:  { color: '#A78BFA', fontSize: 11, fontWeight: '700', letterSpacing: 3, marginBottom: 6 },
  headerTitle:    { color: '#FFFFFF', fontSize: 38, fontWeight: '800', letterSpacing: -1 },
  headerSubtitle: { color: '#C4B5FD', fontSize: 14, marginTop: 6, lineHeight: 20 },

  searchSection: {
    backgroundColor: '#FFFFFF', margin: 16, borderRadius: 24, padding: 20,
    shadowColor: '#6D28D9', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.1, shadowRadius: 12, elevation: 6,
  },
  sectionLabel: { fontSize: 10, fontWeight: '800', color: '#7C3AED', letterSpacing: 2, marginBottom: 10 },

  recentBlock:  { marginBottom: 16 },
  chipRow:      { flexDirection: 'row', gap: 8 },
  chip:         { backgroundColor: '#EDE9FE', borderRadius: 20, paddingHorizontal: 14, paddingVertical: 7, maxWidth: 160 },
  chipText:     { fontSize: 12, color: '#6D28D9', fontWeight: '600' },

  mainInput:    { backgroundColor: '#F5F3FF', borderRadius: 14, padding: 16, fontSize: 15, color: '#1F2937', borderWidth: 1.5, borderColor: '#DDD6FE', marginBottom: 10 },
  statusPill:   { backgroundColor: '#EDE9FE', borderRadius: 20, paddingHorizontal: 14, paddingVertical: 6, alignSelf: 'flex-start', marginBottom: 8 },
  cachePill:    { backgroundColor: '#ECFDF5' },
  statusPillText: { fontSize: 11, color: '#6D28D9', fontWeight: '600' },
  errorText:    { fontSize: 12, color: '#DC2626', marginBottom: 8 },
  button:       { backgroundColor: '#7C3AED', borderRadius: 16, padding: 18, alignItems: 'center', shadowColor: '#7C3AED', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.35, shadowRadius: 10, elevation: 8 },
  buttonDisabled: { opacity: 0.6 },
  buttonText:   { color: '#FFFFFF', fontWeight: '800', fontSize: 16, letterSpacing: 0.5 },

  resultsSection:  { paddingHorizontal: 16 },
  resultsTitleRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 },
  cachedBadge:     { backgroundColor: '#ECFDF5', borderRadius: 12, paddingHorizontal: 10, paddingVertical: 4 },
  cachedBadgeText: { fontSize: 10, color: '#065F46', fontWeight: '700', letterSpacing: 1 },

  eventRow:         { flexDirection: 'row', gap: 8, marginBottom: 12, flexWrap: 'wrap' },
  eventBadge:       { backgroundColor: '#7C3AED', borderRadius: 20, paddingHorizontal: 12, paddingVertical: 5 },
  eventBadgeText:   { color: '#fff', fontSize: 12, fontWeight: '700' },
  weatherBadge:     { backgroundColor: '#ECFDF5', borderRadius: 20, paddingHorizontal: 12, paddingVertical: 5 },
  weatherBadgeText: { color: '#065F46', fontSize: 12, fontWeight: '600' },
  locationBadge:    { backgroundColor: '#FFF7ED', borderRadius: 20, paddingHorizontal: 12, paddingVertical: 5 },
  locationBadgeText:{ color: '#92400E', fontSize: 12, fontWeight: '600' },

  swipeHint: { fontSize: 11, color: '#9CA3AF', textAlign: 'center', marginBottom: 14, letterSpacing: 0.3 },

  // Swipe overlays
  swipeOverlay: {
    position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
    zIndex: 10, borderRadius: 24, justifyContent: 'center', alignItems: 'center',
  },
  likeOverlay:       { backgroundColor: 'rgba(16, 185, 129, 0.85)' },
  skipOverlay:       { backgroundColor: 'rgba(239, 68, 68, 0.85)' },
  swipeOverlayText:  { fontSize: 28, fontWeight: '800', color: '#fff', letterSpacing: 2 },

  // Card
  card:       { backgroundColor: '#FFFFFF', borderRadius: 24, marginBottom: 20, overflow: 'hidden', shadowColor: '#6D28D9', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.12, shadowRadius: 16, elevation: 8 },
  cardLiked:  { borderWidth: 2, borderColor: '#10B981' },
  cardSkipped:{ borderWidth: 2, borderColor: '#EF4444', opacity: 0.75 },

  imageWrapper:     { position: 'relative' },
  cardImage:        { width: '100%', height: 260, backgroundColor: '#EDE9FE' },
  rankBadge:        { position: 'absolute', top: 14, left: 14, backgroundColor: 'rgba(255,255,255,0.95)', borderRadius: 14, paddingHorizontal: 10, paddingVertical: 6, alignItems: 'center' },
  rankEmoji:        { fontSize: 20 },
  rankText:         { fontSize: 8, fontWeight: '800', color: '#7C3AED', letterSpacing: 1 },
  confidenceBadge:  { position: 'absolute', top: 14, right: 14, backgroundColor: '#7C3AED', borderRadius: 14, paddingHorizontal: 12, paddingVertical: 6, alignItems: 'center' },
  confidenceText:   { fontSize: 18, fontWeight: '800', color: '#FFFFFF' },
  confidenceLabel:  { fontSize: 8, fontWeight: '700', color: '#C4B5FD', letterSpacing: 1 },
  feedbackBadge:    { position: 'absolute', bottom: 14, left: '50%', transform: [{ translateX: -50 }], borderRadius: 20, paddingHorizontal: 16, paddingVertical: 6 },
  feedbackLiked:    { backgroundColor: 'rgba(16, 185, 129, 0.9)' },
  feedbackSkipped:  { backgroundColor: 'rgba(239, 68, 68, 0.9)' },
  feedbackBadgeText:{ color: '#fff', fontWeight: '800', fontSize: 13 },

  cardBody:    { padding: 18 },
  outfitName:  { fontSize: 20, fontWeight: '800', color: '#1F2937', marginBottom: 12, lineHeight: 26 },

  tagRow:       { flexDirection: 'row', gap: 8, marginBottom: 14, flexWrap: 'wrap' },
  fabricTag:    { backgroundColor: '#F3F4F6', borderRadius: 20, paddingHorizontal: 12, paddingVertical: 6 },
  fabricTagText:{ fontSize: 12, color: '#374151', fontWeight: '600' },
  colorTag:     { flexDirection: 'row', alignItems: 'center', gap: 5, backgroundColor: '#F9FAFB', borderRadius: 20, paddingHorizontal: 12, paddingVertical: 6, borderWidth: 1, borderColor: '#E5E7EB' },
  colorDot:     { width: 12, height: 12, borderRadius: 6, borderWidth: 1, borderColor: 'rgba(0,0,0,0.1)' },
  colorTagText: { fontSize: 12, color: '#374151', fontWeight: '600' },

  reasonBox:   { backgroundColor: '#FAFAFA', borderRadius: 16, padding: 14, borderLeftWidth: 3, borderLeftColor: '#7C3AED', marginBottom: 12 },
  reasonTitle: { fontSize: 10, fontWeight: '800', color: '#7C3AED', letterSpacing: 1.5, marginBottom: 10 },
  reasonRow:   { flexDirection: 'row', gap: 6, marginBottom: 5 },
  reasonDot:   { fontSize: 12, color: '#7C3AED', marginTop: 1 },
  reasonText:  { fontSize: 12, color: '#4B5563', lineHeight: 18, flex: 1 },

  comboBox:   { backgroundColor: '#F0FDF4', borderRadius: 16, padding: 14, borderLeftWidth: 3, borderLeftColor: '#10B981', marginBottom: 12 },
  comboTitle: { fontSize: 10, fontWeight: '800', color: '#065F46', letterSpacing: 1.5, marginBottom: 6 },
  comboText:  { fontSize: 13, color: '#065F46', lineHeight: 18 },

  actionRow:       { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 4 },
  actionBtn:       { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: '#F3F4F6', borderRadius: 20, paddingHorizontal: 18, paddingVertical: 10, borderWidth: 1.5, borderColor: 'transparent' },
  skipBtn:         { borderColor: '#FCA5A5' },
  likeBtn:         { borderColor: '#6EE7B7' },
  actionBtnActive: { backgroundColor: '#EDE9FE', borderColor: '#7C3AED' },
  actionBtnEmoji:  { fontSize: 16 },
  actionBtnLabel:  { fontSize: 13, fontWeight: '700', color: '#374151' },
  shareBtn:        { backgroundColor: '#EDE9FE', borderRadius: 20, paddingHorizontal: 18, paddingVertical: 10 },
  shareBtnText:    { fontSize: 13, fontWeight: '700', color: '#7C3AED' },

  likedSummary: { backgroundColor: '#FFF', borderRadius: 20, padding: 18, marginBottom: 16, borderWidth: 1.5, borderColor: '#FCA5A5' },
  likedTitle:   { fontSize: 12, fontWeight: '800', color: '#DC2626', letterSpacing: 1, marginBottom: 10 },
  likedItem:    { fontSize: 14, color: '#374151', marginBottom: 6, lineHeight: 20 },
});
