import { Image } from 'expo-image';
import {
  StyleSheet, TouchableOpacity, ActivityIndicator,
  View, TextInput, Alert, Text, ScrollView, SafeAreaView, StatusBar,
} from 'react-native';
import React, { useState } from 'react';
import * as Location from 'expo-location';

// --- TYPES ---
interface RecommendationDetail {
  outfit: string;
  confidence: string;
  fabric: string;
  image_url: string;
  reason?: string;
}

interface ApiResponse {
  event_class: string;
  location_detected: string;
  weather: string;
  recommendations: RecommendationDetail[];
  logic_summary?: string;
}

const RANKS = ['🥇', '🥈', '🥉'];

export default function HomeScreen() {
  const [data, setData]               = useState<ApiResponse | null>(null);
  const [loading, setLoading]         = useState(false);
  const [userIntent, setUserIntent]   = useState('');
  const [backendHost] = useState('10.123.244.242');
  const [backendStatus, setBackendStatus] = useState('');
  const [errorMessage, setErrorMessage]   = useState('');

  const defaultHosts = [backendHost, '10.0.2.2', '127.0.0.1', 'localhost'];
  const backendHosts = Array.from(new Set(defaultHosts.filter(Boolean)));

  const localWardrobe = [
    {
      outfit: 'White Linen Summer Dress', fabric: 'Linen',
      image_url: 'https://images.unsplash.com/photo-1595777457583-95e059d581b8?q=80&w=500',
      confidence: '87%', score: 0.87,
      reason: 'Linen is breathable and lightweight · NLP model: Linen suits casual occasions · GNN: high style compatibility',
    },
    {
      outfit: 'Silk Banarasi Saree', fabric: 'Silk',
      image_url: 'https://images.unsplash.com/photo-1610030469983-98e550d6193c?q=80&w=500',
      confidence: '91%', score: 0.91,
      reason: 'Traditional choice for Sri Lankan weddings · Silk is a premium formal fabric · GNN: high style compatibility',
    },
    {
      outfit: 'Cotton Embroidered Kurti', fabric: 'Cotton',
      image_url: 'https://images.unsplash.com/photo-1581044777550-4cfa60707c03?q=80&w=500',
      confidence: '83%', score: 0.83,
      reason: 'Cotton keeps you cool in tropical heat · Versatile for casual and semi-formal events · GNN: moderate style compatibility',
    },
  ];

  const getMatchmaking = async () => {
    if (!userIntent.trim()) return Alert.alert('Input Needed', 'Please describe your event.');

    setLoading(true);
    setData(null);
    setBackendStatus('');
    setErrorMessage('');

    try {
      let latitude = 6.9271, longitude = 79.8612, locationSource = 'Default (Colombo)';
      try {
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (status === 'granted') {
          const locationPromise = Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Low });
          const timeoutPromise  = new Promise<null>((_, reject) => setTimeout(() => reject(new Error('GPS timeout')), 8000));
          const location = await Promise.race([locationPromise, timeoutPromise]) as Awaited<typeof locationPromise>;
          latitude = location.coords.latitude;
          longitude = location.coords.longitude;
          locationSource = 'GPS';
        }
      } catch { setBackendStatus('GPS unavailable, using default location (Colombo)'); }

      const SL_CITIES = [
        'nuwara eliya','ella','horton plains','knuckles','haputale','bandarawela',
        'kandy','matara','galle','colombo','negombo','trincomalee','batticaloa',
        'jaffna','anuradhapura','polonnaruwa','sigiriya','dambulla','ratnapura',
        'hambantota','tangalle','weligama','mirissa','unawatuna','hikkaduwa',
        'bentota','kalutara','chilaw','puttalam','badulla','hatton','kurunegala',
      ];
      const lowerIntent = userIntent.toLowerCase();
      const detectedDestination = SL_CITIES.find(c => lowerIntent.includes(c)) ?? null;

      const API_KEY  = '64d7c88e4cd6807d9121d8611ed30457';
      const weatherUrl = detectedDestination
        ? `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(detectedDestination)},LK&appid=${API_KEY}&units=metric`
        : `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&appid=${API_KEY}&units=metric`;

      const weatherRes  = await fetch(weatherUrl);
      const weatherData = await weatherRes.json();
      if (!weatherRes.ok || !weatherData?.weather?.[0]) throw new Error('Weather API failed');

      const city        = weatherData.name || detectedDestination || locationSource;
      const weatherMain = weatherData.weather[0].main;
      const humidity    = weatherData.main.humidity;
      const temperature = weatherData.main.temp;

      if (detectedDestination) setBackendStatus(`📍 ${city} — ${temperature.toFixed(0)}°C, ${weatherMain}`);

      const query = `user_input=${encodeURIComponent(userIntent)}&city=${encodeURIComponent(city)}&weather=${encodeURIComponent(weatherMain)}&humidity=${humidity}&temperature=${temperature}`;
      let result: ApiResponse | null = null, lastError: string | null = null;

      for (const host of backendHosts) {
        setBackendStatus(`Connecting to ${host}…`);
        try {
          const response = await fetch(`http://${host}:8000/recommend?${query}`);
          if (!response.ok) { lastError = `${host}: ${response.status}`; continue; }
          const parsed = await response.json();
          if (parsed?.recommendations?.length) { result = parsed; setBackendStatus(`✅ Connected — ${city}`); break; }
          lastError = `${host}: empty response`;
        } catch (e: any) { lastError = `${host}: ${e.message}`; }
      }

      if (result) { setData(result); }
      else { const msg = lastError || 'No backend reachable.'; setErrorMessage(msg); throw new Error(msg); }

    } catch {
      setData({ event_class: userIntent || 'Casual', location_detected: 'Local', weather: 'Unknown', recommendations: localWardrobe });
      Alert.alert('Connection Error', 'Using local recommendations.');
    } finally { setLoading(false); }
  };

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="light-content" backgroundColor="#2D1B69" />
      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false}>

        {/* ── HEADER ── */}
        <View style={styles.header}>
          <Text style={styles.headerEyebrow}>AI-POWERED FASHION</Text>
          <Text style={styles.headerTitle}>E-Wardrobe</Text>
          <Text style={styles.headerSubtitle}>Smart outfit recommendations for every occasion</Text>
        </View>

        {/* ── SEARCH SECTION ── */}
        <View style={styles.searchSection}>
          <Text style={styles.sectionLabel}>{"WHAT'S YOUR OCCASION?"}</Text>
          <TextInput
            style={styles.mainInput}
            placeholder="e.g. Wedding in Matara, Trip to Ella…"
            value={userIntent}
            onChangeText={setUserIntent}
            placeholderTextColor="#9CA3AF"
          />

          {backendStatus ? (
            <View style={styles.statusPill}>
              <Text style={styles.statusPillText}>{backendStatus}</Text>
            </View>
          ) : null}
          {errorMessage ? (
            <Text style={styles.errorText}>{errorMessage}</Text>
          ) : null}

          <TouchableOpacity style={[styles.button, loading && styles.buttonDisabled]} onPress={getMatchmaking} disabled={loading}>
            {loading
              ? <ActivityIndicator color="#fff" />
              : <Text style={styles.buttonText}>✨  Get AI Recommendations</Text>}
          </TouchableOpacity>
        </View>

        {/* ── CARDS ── */}
        {data && (
          <View style={styles.resultsSection}>
            <Text style={styles.sectionLabel}>TOP PICKS FOR YOU</Text>
            <View style={styles.eventRow}>
              <View style={styles.eventBadge}><Text style={styles.eventBadgeText}>{data.event_class}</Text></View>
              <View style={styles.weatherBadge}><Text style={styles.weatherBadgeText}>🌤 {data.weather}</Text></View>
              <View style={styles.locationBadge}><Text style={styles.locationBadgeText}>📍 {data.location_detected}</Text></View>
            </View>

            {data.recommendations.map((item, index) => (
              <View key={index} style={styles.card}>

                {/* Image with rank overlay */}
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
                </View>

                {/* Card body */}
                <View style={styles.cardBody}>
                  <Text style={styles.outfitName}>{item.outfit}</Text>

                  <View style={styles.tagRow}>
                    <View style={styles.fabricTag}>
                      <Text style={styles.fabricTagText}>🧵 {item.fabric}</Text>
                    </View>
                    <View style={styles.intentTag}>
                      <Text style={styles.intentTagText}>{data.event_class}</Text>
                    </View>
                  </View>

                  <View style={styles.reasonBox}>
                    <Text style={styles.reasonTitle}>💡 WHY THIS SUITS YOU</Text>
                    {(item.reason || `NLP model: ${item.fabric} evaluated for this event · GNN: style compatibility assessed`).split(' · ').map((r, i) => (
                      <View key={i} style={styles.reasonRow}>
                        <Text style={styles.reasonDot}>•</Text>
                        <Text style={styles.reasonText}>{r}</Text>
                      </View>
                    ))}
                  </View>
                </View>
              </View>
            ))}
          </View>
        )}

        <View style={{ height: 40 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe:   { flex: 1, backgroundColor: '#2D1B69' },
  scroll: { flex: 1, backgroundColor: '#F3F0FF' },

  // Header
  header: { backgroundColor: '#2D1B69', paddingHorizontal: 24, paddingTop: 40, paddingBottom: 36 },
  headerEyebrow: { color: '#A78BFA', fontSize: 11, fontWeight: '700', letterSpacing: 3, marginBottom: 6 },
  headerTitle:   { color: '#FFFFFF', fontSize: 38, fontWeight: '800', letterSpacing: -1 },
  headerSubtitle:{ color: '#C4B5FD', fontSize: 14, marginTop: 6, lineHeight: 20 },

  // Search section
  searchSection: { backgroundColor: '#FFFFFF', margin: 16, borderRadius: 24, padding: 20, shadowColor: '#6D28D9', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.1, shadowRadius: 12, elevation: 6 },
  sectionLabel:  { fontSize: 10, fontWeight: '800', color: '#7C3AED', letterSpacing: 2, marginBottom: 12 },
  mainInput:  { backgroundColor: '#F5F3FF', borderRadius: 14, padding: 16, fontSize: 15, color: '#1F2937', borderWidth: 1.5, borderColor: '#DDD6FE', marginBottom: 10 },
  statusPill: { backgroundColor: '#EDE9FE', borderRadius: 20, paddingHorizontal: 14, paddingVertical: 6, alignSelf: 'flex-start', marginBottom: 8 },
  statusPillText: { fontSize: 11, color: '#6D28D9', fontWeight: '600' },
  errorText:  { fontSize: 12, color: '#DC2626', marginBottom: 8 },
  button:     { backgroundColor: '#7C3AED', borderRadius: 16, padding: 18, alignItems: 'center', shadowColor: '#7C3AED', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.35, shadowRadius: 10, elevation: 8 },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: '#FFFFFF', fontWeight: '800', fontSize: 16, letterSpacing: 0.5 },

  // Results
  resultsSection: { paddingHorizontal: 16 },
  eventRow: { flexDirection: 'row', gap: 8, marginBottom: 16, flexWrap: 'wrap' },
  eventBadge:    { backgroundColor: '#7C3AED', borderRadius: 20, paddingHorizontal: 12, paddingVertical: 5 },
  eventBadgeText:{ color: '#fff', fontSize: 12, fontWeight: '700' },
  weatherBadge:  { backgroundColor: '#ECFDF5', borderRadius: 20, paddingHorizontal: 12, paddingVertical: 5 },
  weatherBadgeText: { color: '#065F46', fontSize: 12, fontWeight: '600' },
  locationBadge: { backgroundColor: '#FFF7ED', borderRadius: 20, paddingHorizontal: 12, paddingVertical: 5 },
  locationBadgeText: { color: '#92400E', fontSize: 12, fontWeight: '600' },

  // Card
  card: { backgroundColor: '#FFFFFF', borderRadius: 24, marginBottom: 20, overflow: 'hidden', shadowColor: '#6D28D9', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.12, shadowRadius: 16, elevation: 8 },
  imageWrapper: { position: 'relative' },
  cardImage: { width: '100%', height: 280, backgroundColor: '#EDE9FE' },
  rankBadge: { position: 'absolute', top: 14, left: 14, backgroundColor: 'rgba(255,255,255,0.95)', borderRadius: 14, paddingHorizontal: 10, paddingVertical: 6, alignItems: 'center' },
  rankEmoji: { fontSize: 20 },
  rankText:  { fontSize: 8, fontWeight: '800', color: '#7C3AED', letterSpacing: 1 },
  confidenceBadge: { position: 'absolute', top: 14, right: 14, backgroundColor: '#7C3AED', borderRadius: 14, paddingHorizontal: 12, paddingVertical: 6, alignItems: 'center' },
  confidenceText:  { fontSize: 18, fontWeight: '800', color: '#FFFFFF' },
  confidenceLabel: { fontSize: 8, fontWeight: '700', color: '#C4B5FD', letterSpacing: 1 },

  // Card body
  cardBody:   { padding: 18 },
  outfitName: { fontSize: 20, fontWeight: '800', color: '#1F2937', marginBottom: 12, lineHeight: 26 },
  tagRow:     { flexDirection: 'row', gap: 8, marginBottom: 14 },
  fabricTag:  { backgroundColor: '#F3F4F6', borderRadius: 20, paddingHorizontal: 12, paddingVertical: 6 },
  fabricTagText: { fontSize: 12, color: '#374151', fontWeight: '600' },
  intentTag:  { backgroundColor: '#EDE9FE', borderRadius: 20, paddingHorizontal: 12, paddingVertical: 6 },
  intentTagText: { fontSize: 12, color: '#6D28D9', fontWeight: '600' },

  // Reason box
  reasonBox:   { backgroundColor: '#FAFAFA', borderRadius: 16, padding: 14, borderLeftWidth: 3, borderLeftColor: '#7C3AED' },
  reasonTitle: { fontSize: 10, fontWeight: '800', color: '#7C3AED', letterSpacing: 1.5, marginBottom: 10 },
  reasonRow:   { flexDirection: 'row', gap: 6, marginBottom: 5 },
  reasonDot:   { fontSize: 12, color: '#7C3AED', marginTop: 1 },
  reasonText:  { fontSize: 12, color: '#4B5563', lineHeight: 18, flex: 1 },
});
