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
  TouchableOpacity,
  View,
} from "react-native";

import axios from "axios";

import { CLASSIFICATION_URL, WARDROBE_URL } from "../../constants/api";

const API_URL = CLASSIFICATION_URL;

export default function TrendsScreen() {
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<any>(null);
  const [eventSuggestions, setEventSuggestions] = useState<any[]>([]);

  const loadTrendAnalysis = async () => {
    try {
      setLoading(true);

      const [wardrobeRes, eventsRes] = await Promise.all([
        axios.get(`${WARDROBE_URL}/`),
        axios.get(`${WARDROBE_URL}/schedule`),
      ]);

      const wardrobeItems = wardrobeRes.data.map((d: any) => ({
        id: d.item_id,
        type: d.type,
        color: d.color,
        gender: d.gender,
        season: d.season,
        material: d.material,
        processedImageUrl: d.processed_image_url,
        trendAnalysis: d.trend_analysis,
      }));
      const events = eventsRes.data.map((e: any) => ({
        id: e.event_id,
        eventName: e.event_name,
        eventDate: e.event_date,
        eventTime: e.event_time,
      }));

      const analysisResponse = await axios.post(`${API_URL}/wardrobe/trend-analysis`, {
        items: wardrobeItems,
      });

      const suggestions = await Promise.all(
        events.slice(0, 5).map(async (event: any) => {
          const response = await axios.post(`${API_URL}/schedule/suggestion`, {
            items: wardrobeItems,
            event,
          });

          return {
            event,
            recommendation: response.data,
          };
        })
      );

      setAnalysis(analysisResponse.data);
      setEventSuggestions(suggestions);
    } catch (error) {
      console.log("Trend analysis error:", error);
      Alert.alert("Trend analysis failed", "Check backend server and internet connection.");
    } finally {
      setLoading(false);
    }
  };

  const maxChartValue = Math.max(
    1,
    ...(analysis?.chart_data || []).map((item: any) => item.value)
  );

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" />

      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Wardrobe Trends</Text>
        <Text style={styles.subtitle}>
          Saved item analysis, fashion status, and event outfit suggestions
        </Text>

        <TouchableOpacity
          style={[styles.primaryButton, loading && styles.disabledButton]}
          onPress={loadTrendAnalysis}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#ffffff" />
          ) : (
            <Text style={styles.buttonText}>Analyze Saved Wardrobe</Text>
          )}
        </TouchableOpacity>

        {analysis && (
          <>
            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Fashion Status Summary</Text>

              <View style={styles.scoreRow}>
                <View style={styles.scoreBox}>
                  <Text style={styles.scoreValue}>{analysis.summary.total_items}</Text>
                  <Text style={styles.scoreLabel}>Items</Text>
                </View>
                <View style={styles.scoreBox}>
                  <Text style={styles.scoreValue}>{analysis.summary.average_score}%</Text>
                  <Text style={styles.scoreLabel}>Avg Trend</Text>
                </View>
              </View>

              {analysis.chart_data.map((bar: any) => (
                <View key={bar.label} style={styles.chartRow}>
                  <Text style={styles.chartLabel}>{bar.label}</Text>
                  <View style={styles.chartTrack}>
                    <View
                      style={[
                        styles.chartFill,
                        {
                          width: `${(bar.value / maxChartValue) * 100}%`,
                          backgroundColor: bar.color,
                        },
                      ]}
                    />
                  </View>
                  <Text style={styles.chartValue}>{bar.value}</Text>
                </View>
              ))}
            </View>

            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Trending Suggestions</Text>

              {analysis.suggestions.map((suggestion: string) => (
                <View key={suggestion} style={styles.suggestionBox}>
                  <Text style={styles.suggestionText}>{suggestion}</Text>
                </View>
              ))}
            </View>

            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Saved Item Trend Results</Text>

              {analysis.items.map((item: any) => (
                <View key={item.id || `${item.type}-${item.color}`} style={styles.itemCard}>
                  {item.processedImageUrl && (
                    <Image source={{ uri: item.processedImageUrl }} style={styles.itemImage} />
                  )}

                  <View style={styles.itemHeader}>
                    <Text style={styles.itemTitle}>
                      {item.color} {item.type}
                    </Text>
                    <Text style={[styles.statusBadge, statusStyle(item.status)]}>
                      {item.status}
                    </Text>
                  </View>

                  <View style={styles.progressTrack}>
                    <View style={[styles.progressFill, { width: `${item.score}%` }]} />
                  </View>

                  <Text style={styles.itemInfo}>Trend score: {item.score}%</Text>
                  <Text style={styles.itemInfo}>Keyword: {item.keyword}</Text>
                  <Text style={styles.itemSuggestion}>{item.suggestion}</Text>
                </View>
              ))}
            </View>

            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Best Looks For Scheduled Dates</Text>

              {eventSuggestions.length === 0 && (
                <Text style={styles.emptyText}>No scheduled events found.</Text>
              )}

              {eventSuggestions.map(({ event, recommendation }) => (
                <View key={event.id} style={styles.eventCard}>
                  <Text style={styles.eventTitle}>{event.eventName}</Text>
                  <Text style={styles.eventMeta}>
                    {event.eventDate} at {event.eventTime}
                  </Text>
                  <Text style={styles.itemSuggestion}>{recommendation.suggestion}</Text>
                </View>
              ))}
            </View>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function statusStyle(status: string) {
  if (status === "trending") {
    return { color: "#22c55e", borderColor: "#22c55e" };
  }
  if (status === "outdated") {
    return { color: "#f97316", borderColor: "#f97316" };
  }
  return { color: "#38bdf8", borderColor: "#38bdf8" };
}

/* Tailwind violet / slate palette - matches the Recommend screen. */
const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: "#F3F0FF",
  },

  container: {
    padding: 20,
    paddingBottom: 40,
  },

  title: {
    fontSize: 32,
    fontWeight: "800",
    color: "#2D1B69",
    textAlign: "center",
    marginTop: 20,
  },

  subtitle: {
    fontSize: 14,
    color: "#6B7280",
    textAlign: "center",
    marginBottom: 22,
    marginTop: 8,
  },

  primaryButton: {
    backgroundColor: "#7C3AED",
    paddingVertical: 14,
    borderRadius: 16,
    alignItems: "center",
    marginBottom: 18,
    shadowColor: "#7C3AED",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.35,
    shadowRadius: 10,
    elevation: 3,
  },

  disabledButton: {
    backgroundColor: "#C4B5FD",
  },

  buttonText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "800",
  },

  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 24,
    padding: 18,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: "#EDE9FE",
    shadowColor: "#6D28D9",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
    elevation: 4,
  },

  sectionTitle: {
    color: "#1F2937",
    fontSize: 20,
    fontWeight: "800",
    marginBottom: 14,
  },

  scoreRow: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 18,
  },

  scoreBox: {
    flex: 1,
    backgroundColor: "#F5F3FF",
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: "#EDE9FE",
  },

  scoreValue: {
    color: "#1F2937",
    fontSize: 24,
    fontWeight: "900",
  },

  scoreLabel: {
    color: "#6B7280",
    fontSize: 13,
    marginTop: 4,
  },

  chartRow: {
    alignItems: "center",
    flexDirection: "row",
    gap: 10,
    marginBottom: 12,
  },

  chartLabel: {
    color: "#4B5563",
    fontSize: 13,
    width: 72,
  },

  chartTrack: {
    flex: 1,
    height: 12,
    backgroundColor: "#EDE9FE",
    borderRadius: 12,
    overflow: "hidden",
  },

  chartFill: {
    height: "100%",
    borderRadius: 12,
  },

  chartValue: {
    color: "#1F2937",
    fontSize: 13,
    fontWeight: "800",
    width: 24,
    textAlign: "right",
  },

  suggestionBox: {
    backgroundColor: "#FAFAFA",
    borderRadius: 14,
    padding: 14,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: "#EDE9FE",
    borderLeftWidth: 3,
    borderLeftColor: "#7C3AED",
  },

  suggestionText: {
    color: "#4B5563",
    fontSize: 14,
    lineHeight: 20,
  },

  itemCard: {
    backgroundColor: "#F9FAFB",
    borderRadius: 16,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#EDE9FE",
  },

  itemImage: {
    width: "100%",
    height: 170,
    borderRadius: 14,
    resizeMode: "contain",
    backgroundColor: "#EDE9FE",
    marginBottom: 12,
  },

  itemHeader: {
    alignItems: "center",
    flexDirection: "row",
    gap: 10,
    justifyContent: "space-between",
    marginBottom: 10,
  },

  itemTitle: {
    color: "#1F2937",
    flex: 1,
    fontSize: 17,
    fontWeight: "800",
    textTransform: "capitalize",
  },

  statusBadge: {
    borderRadius: 12,
    borderWidth: 1,
    fontSize: 12,
    fontWeight: "800",
    paddingHorizontal: 10,
    paddingVertical: 4,
    textTransform: "capitalize",
  },

  progressTrack: {
    height: 10,
    backgroundColor: "#EDE9FE",
    borderRadius: 10,
    overflow: "hidden",
    marginBottom: 10,
  },

  progressFill: {
    height: "100%",
    backgroundColor: "#7C3AED",
    borderRadius: 10,
  },

  itemInfo: {
    color: "#4B5563",
    fontSize: 13,
    marginBottom: 4,
  },

  itemSuggestion: {
    color: "#374151",
    fontSize: 13,
    lineHeight: 19,
    marginTop: 6,
  },

  eventCard: {
    backgroundColor: "#F9FAFB",
    borderRadius: 16,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#EDE9FE",
  },

  eventTitle: {
    color: "#1F2937",
    fontSize: 17,
    fontWeight: "800",
    marginBottom: 4,
  },

  eventMeta: {
    color: "#7C3AED",
    fontSize: 13,
  },

  emptyText: {
    color: "#6B7280",
    fontSize: 14,
  },
});
