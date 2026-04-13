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

import * as ImagePicker from "expo-image-picker";
import axios from "axios";

import {
  addDoc,
  collection,
  getDocs,
  orderBy,
  query,
  serverTimestamp,
} from "firebase/firestore";

import { db } from "../../firebaseConfig";

const API_URL = "http://10.0.2.2:8000";

export default function App() {
  const [image, setImage] = useState<any>(null);
  const [prediction, setPrediction] = useState<any>(null);

  const [processedImageUrl, setProcessedImageUrl] = useState<string | null>(null);
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null);

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const [savedItemId, setSavedItemId] = useState<string | null>(null);

  const [eventName, setEventName] = useState("");
  const [eventDate, setEventDate] = useState("");
  const [eventTime, setEventTime] = useState("");
  const [eventNotes, setEventNotes] = useState("");

  const [wardrobeItems, setWardrobeItems] = useState<any[]>([]);
  const [scheduledEvents, setScheduledEvents] = useState<any[]>([]);

  const pickImage = async () => {
    setPrediction(null);
    setProcessedImageUrl(null);
    setOriginalImageUrl(null);
    setSavedItemId(null);

    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      Alert.alert("Permission required", "Please allow gallery access.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0]);
    }
  };

  const predictImage = async () => {
    if (!image) {
      Alert.alert("No image", "Please select an image first.");
      return;
    }

    try {
      setLoading(true);

      const formData = new FormData();

      formData.append("file", {
        uri: image.uri,
        name: "wardrobe_image.jpg",
        type: "image/jpeg",
      } as any);

      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setPrediction(response.data.prediction);

      if (response.data.original_image_url) {
        setOriginalImageUrl(`${API_URL}${response.data.original_image_url}`);
      }

      if (response.data.processed_image_url) {
        setProcessedImageUrl(`${API_URL}${response.data.processed_image_url}`);
      }
    } catch (error) {
      console.log("Prediction error:", error);
      Alert.alert(
        "Prediction failed",
        "Check backend server, IP address, and WiFi connection."
      );
    } finally {
      setLoading(false);
    }
  };

  const saveWardrobeItem = async () => {
    if (!prediction) {
      Alert.alert("No prediction", "Please predict clothing first.");
      return;
    }

    try {
      setSaving(true);

      const savePromise = addDoc(collection(db, "wardrobe_items"), {
        type: prediction.type,
        typeConfidence: prediction.type_confidence,
        color: prediction.color,
        colorConfidence: prediction.color_confidence,
        gender: prediction.gender,
        genderConfidence: prediction.gender_confidence,
        season: prediction.season,
        seasonConfidence: prediction.season_confidence,
        originalImageUrl,
        processedImageUrl,
        createdAt: serverTimestamp(),
      });

      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error("Firebase save timeout")), 15000)
      );

      const docRef: any = await Promise.race([savePromise, timeoutPromise]);

      setSavedItemId(docRef.id);

      Alert.alert("Saved", "Wardrobe item saved to Firebase.");
    } catch (error: any) {
      console.log("Firebase save error:", error);
      Alert.alert("Save failed", error.message || "Could not save item.");
    } finally {
      setSaving(false);
    }
  };

  const saveScheduleEvent = async () => {
    if (!savedItemId) {
      Alert.alert(
        "Save item first",
        "Please save the wardrobe item before scheduling."
      );
      return;
    }

    if (!eventName || !eventDate || !eventTime) {
      Alert.alert("Missing details", "Enter event name, date, and time.");
      return;
    }

    try {
      await addDoc(collection(db, "scheduled_events"), {
        wardrobeItemId: savedItemId,
        eventName,
        eventDate,
        eventTime,
        notes: eventNotes,
        clothingType: prediction?.type,
        clothingColor: prediction?.color,
        clothingGender: prediction?.gender,
        clothingSeason: prediction?.season,
        processedImageUrl,
        createdAt: serverTimestamp(),
      });

      setEventName("");
      setEventDate("");
      setEventTime("");
      setEventNotes("");

      Alert.alert("Scheduled", "Dressing event saved to Firebase.");
    } catch (error) {
      console.log("Schedule error:", error);
      Alert.alert("Schedule failed", "Could not save event.");
    }
  };

  const loadSavedDetails = async () => {
    try {
      const q = query(
        collection(db, "wardrobe_items"),
        orderBy("createdAt", "desc")
      );

      const snapshot = await getDocs(q);

      const items = snapshot.docs.map((doc) => ({
        id: doc.id,
        ...doc.data(),
      }));

      setWardrobeItems(items);
    } catch (error) {
      console.log("Load wardrobe error:", error);
      Alert.alert("Error", "Could not load saved wardrobe details.");
    }
  };

  const loadScheduledEvents = async () => {
    try {
      const q = query(
        collection(db, "scheduled_events"),
        orderBy("createdAt", "desc")
      );

      const snapshot = await getDocs(q);

      const events = snapshot.docs.map((doc) => ({
        id: doc.id,
        ...doc.data(),
      }));

      setScheduledEvents(events);
    } catch (error) {
      console.log("Load events error:", error);
      Alert.alert("Error", "Could not load scheduled events.");
    }
  };

  const ConfidenceBar = ({ label, value }: any) => {
    const percentage = Math.round((value || 0) * 100);

    return (
      <View style={styles.confidenceBox}>
        <View style={styles.confidenceTop}>
          <Text style={styles.confidenceLabel}>{label}</Text>
          <Text style={styles.confidenceValue}>{percentage}%</Text>
        </View>

        <View style={styles.progressTrack}>
          <View style={[styles.progressFill, { width: `${percentage}%` }]} />
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="light-content" />

      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>E-Wardrobe AI</Text>
        <Text style={styles.subtitle}>
          AI clothing prediction, wardrobe storage, and dressing schedule
        </Text>

        <View style={styles.card}>
          <Text style={styles.sectionTitle}>Upload Clothing Image</Text>

          <TouchableOpacity style={styles.uploadBox} onPress={pickImage}>
            {image ? (
              <Image source={{ uri: image.uri }} style={styles.previewImage} />
            ) : (
              <Text style={styles.uploadText}>Tap to choose image</Text>
            )}
          </TouchableOpacity>

          <TouchableOpacity style={styles.primaryButton} onPress={pickImage}>
            <Text style={styles.buttonText}>Choose Image</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[
              styles.predictButton,
              (!image || loading) && styles.disabledButton,
            ]}
            onPress={predictImage}
            disabled={!image || loading}
          >
            {loading ? (
              <ActivityIndicator color="#ffffff" />
            ) : (
              <Text style={styles.buttonText}>Predict Clothing</Text>
            )}
          </TouchableOpacity>
        </View>

        {processedImageUrl && (
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Background Removed Image</Text>
            <Image source={{ uri: processedImageUrl }} style={styles.previewImage} />
          </View>
        )}

        {prediction && (
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>AI Prediction Results</Text>

            <View style={styles.resultGrid}>
              <View style={styles.resultBox}>
                <Text style={styles.resultLabel}>Type</Text>
                <Text style={styles.resultValue}>{prediction.type}</Text>
              </View>

              <View style={styles.resultBox}>
                <Text style={styles.resultLabel}>Color</Text>
                <Text style={styles.resultValue}>{prediction.color}</Text>
              </View>

              <View style={styles.resultBox}>
                <Text style={styles.resultLabel}>Gender</Text>
                <Text style={styles.resultValue}>{prediction.gender}</Text>
              </View>

              <View style={styles.resultBox}>
                <Text style={styles.resultLabel}>Season</Text>
                <Text style={styles.resultValue}>{prediction.season}</Text>
              </View>
            </View>

            <ConfidenceBar label="Type Confidence" value={prediction.type_confidence} />
            <ConfidenceBar label="Color Confidence" value={prediction.color_confidence} />
            <ConfidenceBar label="Gender Confidence" value={prediction.gender_confidence} />
            <ConfidenceBar label="Season Confidence" value={prediction.season_confidence} />

            <TouchableOpacity
              style={[styles.saveButton, saving && styles.disabledButton]}
              onPress={saveWardrobeItem}
              disabled={saving}
            >
              <Text style={styles.buttonText}>
                {saving ? "Saving..." : "Save to Wardrobe"}
              </Text>
            </TouchableOpacity>

            {savedItemId && (
              <Text style={styles.savedText}>Saved Item ID: {savedItemId}</Text>
            )}
          </View>
        )}

        {savedItemId && (
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Schedule Dressing Event</Text>

            <TextInput
              style={styles.input}
              placeholder="Event name"
              placeholderTextColor="#94a3b8"
              value={eventName}
              onChangeText={setEventName}
            />

            <TextInput
              style={styles.input}
              placeholder="Date: 2026-05-20"
              placeholderTextColor="#94a3b8"
              value={eventDate}
              onChangeText={setEventDate}
            />

            <TextInput
              style={styles.input}
              placeholder="Time: 18:30"
              placeholderTextColor="#94a3b8"
              value={eventTime}
              onChangeText={setEventTime}
            />

            <TextInput
              style={[styles.input, styles.notesInput]}
              placeholder="Notes"
              placeholderTextColor="#94a3b8"
              value={eventNotes}
              onChangeText={setEventNotes}
              multiline
            />

            <TouchableOpacity style={styles.scheduleButton} onPress={saveScheduleEvent}>
              <Text style={styles.buttonText}>Save Schedule</Text>
            </TouchableOpacity>
          </View>
        )}

        <View style={styles.card}>
          <Text style={styles.sectionTitle}>Saved Wardrobe Items</Text>

          <TouchableOpacity style={styles.primaryButton} onPress={loadSavedDetails}>
            <Text style={styles.buttonText}>View Saved Wardrobe Items</Text>
          </TouchableOpacity>

          {wardrobeItems.map((item) => (
            <View key={item.id} style={styles.savedCard}>
              {item.processedImageUrl && (
                <Image source={{ uri: item.processedImageUrl }} style={styles.smallImage} />
              )}

              <Text style={styles.savedTitle}>{item.type}</Text>
              <Text style={styles.savedInfo}>Color: {item.color}</Text>
              <Text style={styles.savedInfo}>Gender: {item.gender}</Text>
              <Text style={styles.savedInfo}>Season: {item.season}</Text>
              <Text style={styles.savedInfo}>Item ID: {item.id}</Text>
            </View>
          ))}
        </View>

        <View style={styles.card}>
          <Text style={styles.sectionTitle}>Scheduled Dressing Events</Text>

          <TouchableOpacity style={styles.scheduleButton} onPress={loadScheduledEvents}>
            <Text style={styles.buttonText}>View Scheduled Events</Text>
          </TouchableOpacity>

          {scheduledEvents.map((event) => (
            <View key={event.id} style={styles.savedCard}>
              {event.processedImageUrl && (
                <Image source={{ uri: event.processedImageUrl }} style={styles.smallImage} />
              )}

              <Text style={styles.savedTitle}>{event.eventName}</Text>
              <Text style={styles.savedInfo}>Date: {event.eventDate}</Text>
              <Text style={styles.savedInfo}>Time: {event.eventTime}</Text>
              <Text style={styles.savedInfo}>Dress: {event.clothingType}</Text>
              <Text style={styles.savedInfo}>Color: {event.clothingColor}</Text>
              <Text style={styles.savedInfo}>Notes: {event.notes}</Text>
            </View>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: "#0f172a",
  },

  container: {
    padding: 20,
    paddingBottom: 40,
  },

  title: {
    fontSize: 34,
    fontWeight: "800",
    color: "#ffffff",
    textAlign: "center",
    marginTop: 20,
  },

  subtitle: {
    fontSize: 14,
    color: "#cbd5e1",
    textAlign: "center",
    marginBottom: 25,
    marginTop: 8,
  },

  card: {
    backgroundColor: "#1e293b",
    borderRadius: 22,
    padding: 18,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: "#334155",
  },

  sectionTitle: {
    fontSize: 20,
    fontWeight: "700",
    color: "#ffffff",
    marginBottom: 15,
  },

  uploadBox: {
    height: 300,
    borderRadius: 18,
    borderWidth: 2,
    borderStyle: "dashed",
    borderColor: "#64748b",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0f172a",
    overflow: "hidden",
    marginBottom: 15,
  },

  uploadText: {
    color: "#94a3b8",
    fontSize: 16,
  },

  previewImage: {
    width: "100%",
    height: 300,
    borderRadius: 18,
    resizeMode: "contain",
    backgroundColor: "#ffffff",
  },

  primaryButton: {
    backgroundColor: "#2563eb",
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: "center",
    marginBottom: 12,
  },

  predictButton: {
    backgroundColor: "#16a34a",
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: "center",
  },

  saveButton: {
    backgroundColor: "#f97316",
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: "center",
    marginTop: 16,
  },

  scheduleButton: {
    backgroundColor: "#7c3aed",
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: "center",
    marginTop: 8,
  },

  disabledButton: {
    backgroundColor: "#475569",
  },

  buttonText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "700",
  },

  resultGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
    marginBottom: 20,
  },

  resultBox: {
    width: "47%",
    backgroundColor: "#0f172a",
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: "#334155",
  },

  resultLabel: {
    color: "#94a3b8",
    fontSize: 13,
    marginBottom: 6,
  },

  resultValue: {
    color: "#ffffff",
    fontSize: 18,
    fontWeight: "800",
  },

  confidenceBox: {
    marginBottom: 14,
  },

  confidenceTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 6,
  },

  confidenceLabel: {
    color: "#cbd5e1",
    fontSize: 14,
  },

  confidenceValue: {
    color: "#ffffff",
    fontSize: 14,
    fontWeight: "700",
  },

  progressTrack: {
    height: 10,
    backgroundColor: "#334155",
    borderRadius: 10,
    overflow: "hidden",
  },

  progressFill: {
    height: "100%",
    backgroundColor: "#38bdf8",
    borderRadius: 10,
  },

  input: {
    backgroundColor: "#0f172a",
    color: "#ffffff",
    borderWidth: 1,
    borderColor: "#334155",
    borderRadius: 14,
    padding: 14,
    marginBottom: 12,
    fontSize: 15,
  },

  notesInput: {
    height: 90,
    textAlignVertical: "top",
  },

  savedText: {
    color: "#22c55e",
    marginTop: 12,
    fontSize: 13,
  },

  savedCard: {
    backgroundColor: "#0f172a",
    borderRadius: 16,
    padding: 14,
    marginTop: 12,
    borderWidth: 1,
    borderColor: "#334155",
  },

  smallImage: {
    width: "100%",
    height: 180,
    borderRadius: 14,
    resizeMode: "contain",
    backgroundColor: "#ffffff",
    marginBottom: 10,
  },

  savedTitle: {
    color: "#ffffff",
    fontSize: 18,
    fontWeight: "800",
    marginBottom: 6,
  },

  savedInfo: {
    color: "#cbd5e1",
    fontSize: 14,
    marginBottom: 4,
  },
});