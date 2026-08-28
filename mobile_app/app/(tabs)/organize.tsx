import React, { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

import axios from "axios";

import { ORGANIZATION_URL } from "../../constants/api";

const SECTION_ORDER = ["A", "B", "C", "D"];

export default function OrganizeScreen() {
  const [loading, setLoading] = useState(false);
  const [rowBusy, setRowBusy] = useState<string | null>(null);
  const [items, setItems] = useState<any[]>([]);
  const [layout, setLayout] = useState<any>(null);
  const [insights, setInsights] = useState<any>(null);

  const load = async () => {
    try {
      setLoading(true);
      const [orgRes, layoutRes, insightRes] = await Promise.all([
        axios.get(`${ORGANIZATION_URL}/items/organized`),
        axios.get(`${ORGANIZATION_URL}/wardrobe/layout`),
        axios.get(`${ORGANIZATION_URL}/items/insights`),
      ]);
      setItems(orgRes.data);
      setLayout(layoutRes.data);
      setInsights(insightRes.data);
    } catch (error: any) {
      console.log("Organize error:", error);
      Alert.alert("Could not organize", error.message || "Check the backend server.");
    } finally {
      setLoading(false);
    }
  };

  const act = async (itemId: string, action: "wear" | "wash") => {
    try {
      setRowBusy(itemId + action);
      await axios.post(`${ORGANIZATION_URL}/items/${action}/${itemId}`);
      await load();
    } catch (error: any) {
      console.log("Wear/wash error:", error);
      Alert.alert("Action failed", error.message || "Try again.");
    } finally {
      setRowBusy(null);
    }
  };

  const grouped = SECTION_ORDER.map((sec) => ({
    section: sec,
    items: items.filter((i) => i.wardrobe_section === sec),
  })).filter((g) => g.items.length > 0);

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="light-content" />

      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Wardrobe Organization</Text>
        <Text style={styles.subtitle}>
          Your saved clothes clustered into zones, with wear &amp; wash tracking
        </Text>

        <TouchableOpacity
          style={[styles.primaryButton, loading && styles.disabledButton]}
          onPress={load}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#ffffff" />
          ) : (
            <Text style={styles.buttonText}>Organize My Wardrobe</Text>
          )}
        </TouchableOpacity>

        {insights && (
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Insights</Text>
            <View style={styles.statRow}>
              <View style={styles.statBox}>
                <Text style={styles.statValue}>{insights.dirty_count}</Text>
                <Text style={styles.statLabel}>Need washing</Text>
              </View>
              <View style={styles.statBox}>
                <Text style={styles.statValue}>{insights.overused?.length ?? 0}</Text>
                <Text style={styles.statLabel}>Overused</Text>
              </View>
              <View style={styles.statBox}>
                <Text style={styles.statValue}>{insights.underused?.length ?? 0}</Text>
                <Text style={styles.statLabel}>Underused</Text>
              </View>
            </View>
          </View>
        )}

        {layout && (
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Wardrobe Layout ({layout.total_items} items)</Text>
            {SECTION_ORDER.map((k) => {
              const s = layout.sections[k];
              return (
                <View key={k} style={styles.layoutRow}>
                  <Text style={styles.layoutSec}>{k}</Text>
                  <View style={styles.layoutInfo}>
                    <Text style={styles.layoutLabel}>{s.label}</Text>
                    <Text style={styles.layoutLoc}>{s.location}</Text>
                  </View>
                  <Text style={styles.layoutCount}>{s.item_count}</Text>
                </View>
              );
            })}
          </View>
        )}

        {grouped.map((g) => (
          <View key={g.section} style={styles.card}>
            <Text style={styles.sectionTitle}>
              Section {g.section} · {g.items[0].section_label}
            </Text>

            {g.items.map((item) => (
              <View key={item.item_id} style={styles.itemCard}>
                <View style={styles.itemHeader}>
                  <Text style={styles.itemName}>{item.name}</Text>
                  <Text
                    style={[
                      styles.badge,
                      item.status === "Dirty" ? styles.badgeDirty : styles.badgeClean,
                    ]}
                  >
                    {item.status}
                  </Text>
                </View>
                <Text style={styles.itemMeta}>{item.position_label}</Text>
                <Text style={styles.itemMeta}>
                  Worn {item.total_wear_count}× · {item.current_cycle_wears}/{item.max_wears_before_wash} this cycle
                </Text>

                <View style={styles.actions}>
                  <TouchableOpacity
                    style={[styles.wearBtn, rowBusy === item.item_id + "wear" && styles.disabledButton]}
                    onPress={() => act(item.item_id, "wear")}
                    disabled={!!rowBusy}
                  >
                    <Text style={styles.actionText}>Wore it</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.washBtn, rowBusy === item.item_id + "wash" && styles.disabledButton]}
                    onPress={() => act(item.item_id, "wash")}
                    disabled={!!rowBusy}
                  >
                    <Text style={styles.actionText}>Washed it</Text>
                  </TouchableOpacity>
                </View>
              </View>
            ))}
          </View>
        ))}

        {items.length === 0 && !loading && (
          <Text style={styles.emptyText}>
            Save some clothes in the Wardrobe tab, then organize them here.
          </Text>
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

  primaryButton: { backgroundColor: "#8B5CF6", paddingVertical: 15, borderRadius: 14, alignItems: "center", marginBottom: 16 },
  disabledButton: { backgroundColor: "#334155" },
  buttonText: { color: "#ffffff", fontSize: 15, fontWeight: "700", letterSpacing: 0.3 },

  card: { backgroundColor: "#111A2E", borderRadius: 20, padding: 20, marginBottom: 16, borderWidth: 1, borderColor: "#1F2A44" },
  sectionTitle: { fontSize: 16, fontWeight: "800", color: "#F1F5F9", marginBottom: 12, letterSpacing: -0.2 },

  statRow: { flexDirection: "row", gap: 10 },
  statBox: { flex: 1, backgroundColor: "#0B1220", borderRadius: 14, padding: 14, borderWidth: 1, borderColor: "#1F2A44" },
  statValue: { color: "#F1F5F9", fontSize: 22, fontWeight: "900" },
  statLabel: { color: "#8A97AD", fontSize: 11, marginTop: 4, textTransform: "uppercase", letterSpacing: 0.5 },

  layoutRow: { flexDirection: "row", alignItems: "center", gap: 12, paddingVertical: 8 },
  layoutSec: { color: "#C7D2FE", fontSize: 18, fontWeight: "900", width: 24, textAlign: "center" },
  layoutInfo: { flex: 1 },
  layoutLabel: { color: "#F1F5F9", fontSize: 14, fontWeight: "700" },
  layoutLoc: { color: "#8A97AD", fontSize: 12 },
  layoutCount: { color: "#818CF8", fontSize: 16, fontWeight: "800" },

  itemCard: { backgroundColor: "#0B1220", borderRadius: 14, padding: 14, marginBottom: 10, borderWidth: 1, borderColor: "#1F2A44" },
  itemHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", gap: 10 },
  itemName: { color: "#F1F5F9", fontSize: 15, fontWeight: "800", flex: 1, letterSpacing: -0.2 },
  badge: { borderRadius: 10, borderWidth: 1, fontSize: 11, fontWeight: "800", paddingHorizontal: 8, paddingVertical: 3 },
  badgeClean: { color: "#34D399", borderColor: "#1C7F5C" },
  badgeDirty: { color: "#FCA5A5", borderColor: "#7F1D1D" },
  itemMeta: { color: "#AEB9CC", fontSize: 12, marginTop: 5 },

  actions: { flexDirection: "row", gap: 10, marginTop: 12 },
  wearBtn: { flex: 1, backgroundColor: "#6366F1", paddingVertical: 11, borderRadius: 12, alignItems: "center" },
  washBtn: { flex: 1, backgroundColor: "#1E293B", paddingVertical: 11, borderRadius: 12, alignItems: "center", borderWidth: 1, borderColor: "#2E3B54" },
  actionText: { color: "#ffffff", fontSize: 14, fontWeight: "700" },

  emptyText: { color: "#8A97AD", fontSize: 14, textAlign: "center", marginTop: 20, lineHeight: 20 },
});
