import React, { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";

import axios from "axios";

import { RECOMMENDATION_URL } from "../../constants/api";

export default function RecommendScreen() {
  const [userInput, setUserInput] = useState("");
  const [city, setCity] = useState("Colombo");
  const [weather, setWeather] = useState("Humid");
  const [temperature, setTemperature] = useState("28");

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const getRecommendations = async () => {
    if (!userInput.trim()) {
      Alert.alert("Describe the occasion", "e.g. \"office meeting\", \"a friend's wedding in Kandy\".");
      return;
    }

    try {
      setLoading(true);
      setResult(null);

      const response = await axios.get(`${RECOMMENDATION_URL}/recommend`, {
        params: {
          user_input: userInput,
          city,
          weather,
          temperature: Number(temperature) || 28,
        },
      });

      setResult(response.data);
    } catch (error: any) {
      console.log("Recommend error:", error);
      Alert.alert("Recommendation failed", error.message || "Check the backend server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="light-content" />

      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Outfit Recommendations</Text>
        <Text style={styles.subtitle}>
          Best looks for an occasion, ranked from the clothes in your wardrobe
        </Text>

        <View style={styles.card}>
          <Text style={styles.label}>Occasion</Text>
          <TextInput
            style={[styles.input, styles.multiline]}
            placeholder="e.g. traditional wedding in Kandy this evening"
            placeholderTextColor="#94a3b8"
            value={userInput}
            onChangeText={setUserInput}
            multiline
          />

          <View style={styles.row}>
            <View style={styles.rowItem}>
              <Text style={styles.label}>City</Text>
              <TextInput
                style={styles.input}
                placeholderTextColor="#94a3b8"
                value={city}
                onChangeText={setCity}
              />
            </View>
            <View style={styles.rowItem}>
              <Text style={styles.label}>Temp (°C)</Text>
              <TextInput
                style={styles.input}
                placeholderTextColor="#94a3b8"
                keyboardType="numeric"
                value={temperature}
                onChangeText={setTemperature}
              />
            </View>
          </View>

          <Text style={styles.label}>Weather</Text>
          <TextInput
            style={styles.input}
            placeholderTextColor="#94a3b8"
            value={weather}
            onChangeText={setWeather}
          />

          <TouchableOpacity
            style={[styles.primaryButton, loading && styles.disabledButton]}
            onPress={getRecommendations}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#ffffff" />
            ) : (
              <Text style={styles.buttonText}>Get Recommendations</Text>
            )}
          </TouchableOpacity>
        </View>

        {result && (
          <>
            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Reading</Text>
              <Text style={styles.metaLine}>Event type: <Text style={styles.metaStrong}>{result.event_class}</Text></Text>
              <Text style={styles.metaLine}>Location: {result.location_detected}</Text>
              <Text style={styles.metaLine}>
                Source:{" "}
                <Text style={styles.metaStrong}>
                  {result.wardrobe_source === "user_wardrobe"
                    ? `your wardrobe (${result.items_considered} items)`
                    : "demo wardrobe — add 3+ items to use yours"}
                </Text>
              </Text>
              <Text style={styles.logic}>{result.logic_summary}</Text>
            </View>

            {(result.recommendations || []).map((r: any, idx: number) => (
              <View key={`${r.outfit}-${idx}`} style={styles.recCard}>
                {r.image_url ? (
                  <Image source={{ uri: r.image_url }} style={styles.recImage} />
                ) : null}
                <View style={styles.recHeader}>
                  <Text style={styles.recName}>{r.outfit}</Text>
                  <Text style={styles.recScore}>{r.confidence}</Text>
                </View>
                <Text style={styles.recMeta}>
                  {[r.fabric, r.color, r.category].filter(Boolean).join(" · ")}
                </Text>
                <Text style={styles.recReason}>{r.reason}</Text>
                {r.combination ? (
                  <Text style={styles.recCombo}>{r.combination}</Text>
                ) : null}
              </View>
            ))}
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#080C18" },
  container: { padding: 18, paddingBottom: 48, maxWidth: 640, width: "100%", alignSelf: "center" },

  title: { fontSize: 30, fontWeight: "800", color: "#F8FAFC", textAlign: "center", letterSpacing: -0.5, marginTop: 24 },
  subtitle: { fontSize: 14, color: "#8A97AD", textAlign: "center", lineHeight: 20, marginTop: 10, marginBottom: 22, paddingHorizontal: 12 },

  card: {
    backgroundColor: "#111A2E", borderRadius: 20, padding: 20, marginBottom: 16,
    borderWidth: 1, borderColor: "#1F2A44",
  },

  label: {
    color: "#9AA7BD", fontSize: 11, fontWeight: "600", textTransform: "uppercase",
    letterSpacing: 0.6, marginBottom: 6, marginTop: 4,
  },
  input: {
    backgroundColor: "#0B1220", color: "#F1F5F9", borderWidth: 1, borderColor: "#26324C",
    borderRadius: 12, paddingVertical: 13, paddingHorizontal: 14, marginBottom: 10, fontSize: 15,
  },
  multiline: { height: 80, textAlignVertical: "top" },
  row: { flexDirection: "row", gap: 12 },
  rowItem: { flex: 1 },

  primaryButton: {
    backgroundColor: "#6366F1", paddingVertical: 15, borderRadius: 14, alignItems: "center", marginTop: 10,
  },
  disabledButton: { backgroundColor: "#334155" },
  buttonText: { color: "#ffffff", fontSize: 15, fontWeight: "700", letterSpacing: 0.3 },

  sectionTitle: { fontSize: 16, fontWeight: "800", color: "#F1F5F9", marginBottom: 10, letterSpacing: -0.2 },
  metaLine: { color: "#AEB9CC", fontSize: 13, lineHeight: 20 },
  metaStrong: { color: "#C7D2FE", fontWeight: "700" },
  logic: { color: "#8A97AD", fontSize: 12, marginTop: 8, lineHeight: 18 },

  recCard: {
    backgroundColor: "#0B1220", borderRadius: 16, padding: 15, marginBottom: 12,
    borderWidth: 1, borderColor: "#1F2A44",
  },
  recImage: { width: "100%", height: 170, borderRadius: 12, resizeMode: "cover", backgroundColor: "#1E293B", marginBottom: 12 },
  recHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", gap: 10 },
  recName: { color: "#F1F5F9", fontSize: 16, fontWeight: "800", flex: 1, letterSpacing: -0.2 },
  recScore: { color: "#34D399", fontSize: 16, fontWeight: "800" },
  recMeta: { color: "#818CF8", fontSize: 12, fontWeight: "500", marginTop: 4, textTransform: "capitalize" },
  recReason: { color: "#AEB9CC", fontSize: 13, lineHeight: 19, marginTop: 8 },
  recCombo: { color: "#E2E8F0", fontSize: 13, lineHeight: 19, marginTop: 8, fontStyle: "italic" },
});
