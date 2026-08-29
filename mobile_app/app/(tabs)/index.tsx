import React, { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  Platform,
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

import { CLASSIFICATION_URL, WARDROBE_URL } from "../../constants/api";

const API_URL = CLASSIFICATION_URL;

export default function App() {
  const [image, setImage] = useState<any>(null);
  const [backImage, setBackImage] = useState<any>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [trendAnalysis, setTrendAnalysis] = useState<any>(null);

  const [processedImageUrl, setProcessedImageUrl] = useState<string | null>(null);
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null);
  const [backImageUrl, setBackImageUrl] = useState<string | null>(null);

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);


  const [eventName, setEventName] = useState("");
  const [eventDate, setEventDate] = useState("");
  const [eventTime, setEventTime] = useState("");
  const [eventNotes, setEventNotes] = useState("");

  const [wardrobeItems, setWardrobeItems] = useState<any[]>([]);
  const [scheduledEvents, setScheduledEvents] = useState<any[]>([]);

  const [editingId, setEditingId] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [rowBusyId, setRowBusyId] = useState<string | null>(null);
  const [editForm, setEditForm] = useState({
    type: "",
    color: "",
    gender: "",
    season: "",
    material: "",
  });

  const [editEventId, setEditEventId] = useState<string | null>(null);
  const [delEventId, setDelEventId] = useState<string | null>(null);
  const [eventRowBusy, setEventRowBusy] = useState<string | null>(null);
  const [eventForm, setEventForm] = useState({
    eventName: "",
    eventDate: "",
    eventTime: "",
    notes: "",
  });

  const pickImage = async () => {
    setPrediction(null);
    setTrendAnalysis(null);
    setProcessedImageUrl(null);
    setOriginalImageUrl(null);
    setBackImage(null);
    setBackImageUrl(null);

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

  const pickBackImage = async () => {
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
      setBackImage(result.assets[0]);
      setPrediction(null);
      setTrendAnalysis(null);
      setProcessedImageUrl(null);
      setOriginalImageUrl(null);
      setBackImageUrl(null);
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

      if (Platform.OS === "web") {
        const frontBlob = await (await fetch(image.uri)).blob();
        formData.append("file", frontBlob, "wardrobe_image.jpg");

        if (backImage) {
          const backBlob = await (await fetch(backImage.uri)).blob();
          formData.append("back_file", backBlob, "wardrobe_back_image.jpg");
        }
      } else {
        formData.append("file", {
          uri: image.uri,
          name: "wardrobe_image.jpg",
          type: "image/jpeg",
        } as any);

        if (backImage) {
          formData.append("back_file", {
            uri: backImage.uri,
            name: "wardrobe_back_image.jpg",
            type: "image/jpeg",
          } as any);
        }
      }

      const response = await axios.post(
        `${API_URL}/predict`,
        formData,
        Platform.OS === "web"
          ? undefined
          : { headers: { "Content-Type": "multipart/form-data" } }
      );

      setPrediction(response.data.prediction);
      setTrendAnalysis(response.data.trend_analysis);

      if (response.data.original_image_url) {
        setOriginalImageUrl(`${API_URL}${response.data.original_image_url}`);
      }

      if (response.data.back_processed_image_url) {
        setBackImageUrl(`${API_URL}${response.data.back_processed_image_url}`);
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

  // Fires the dressing-event save for a just-saved wardrobe item, using the
  // current schedule-form fields. Returns the trend suggestion (or null).
  const postScheduleEvent = async (item: any) => {
    let suggestion: any = null;
    try {
      const r = await axios.post(`${API_URL}/schedule/suggestion`, {
        items: [{
          id: item.id,
          type: item.type,
          color: item.color,
          gender: item.gender,
          season: item.season,
          material: item.material,
          processedImageUrl: item.processedImageUrl,
        }],
        event: { eventName, eventDate, eventTime },
      });
      suggestion = r.data;
    } catch (suggestionError) {
      console.log("Schedule suggestion error:", suggestionError);
    }

    await axios.post(`${WARDROBE_URL}/schedule`, {
      wardrobe_item_id: item.id,
      event_name: eventName,
      event_date: eventDate,
      event_time: eventTime,
      notes: eventNotes,
      clothing_type: item.type,
      clothing_color: item.color,
      processed_image_url: item.processedImageUrl,
      trend_suggestion: suggestion,
    });

    return suggestion;
  };

  const saveWardrobeItem = async () => {
    if (!prediction) {
      Alert.alert("No prediction", "Please predict clothing first.");
      return;
    }

    try {
      setSaving(true);

      const response = await axios.post(`${WARDROBE_URL}/`, {
        prediction: { ...prediction, trend_analysis: trendAnalysis },
        images: {
          original_image_url: originalImageUrl,
          processed_image_url: processedImageUrl,
          back_image_url: backImageUrl,
          back_processed_image_url: backImageUrl,
        },
      });

      const snapshot = {
        id: response.data.item_id,
        type: prediction.type,
        color: prediction.color,
        gender: prediction.gender,
        season: prediction.season,
        material: prediction.material,
        processedImageUrl,
      };

      // Collapse the prediction result now that it's saved.
      setPrediction(null);
      setTrendAnalysis(null);
      setProcessedImageUrl(null);
      setOriginalImageUrl(null);
      setBackImageUrl(null);
      setImage(null);
      setBackImage(null);

      // If the dressing-event form was filled in, save that event too.
      const scheduleComplete =
        !!(eventName.trim() && eventDate.trim() && eventTime.trim());
      let message = "Added to your wardrobe.";
      if (scheduleComplete) {
        try {
          const suggestion = await postScheduleEvent(snapshot);
          message = suggestion?.suggestion
            ? `Saved to your wardrobe and scheduled.\n\nBest for this date: ${suggestion.suggestion}`
            : "Saved to your wardrobe and scheduled.";
        } catch (scheduleError) {
          console.log("Schedule save error:", scheduleError);
          message = "Saved to your wardrobe. The dressing event could not be saved - try again from the Scheduled Events section.";
        }
      }

      // Collapse and reset the schedule form.
      setEventName("");
      setEventDate("");
      setEventTime("");
      setEventNotes("");

      Alert.alert("Saved", message);
    } catch (error: any) {
      console.log("Save error:", error);
      Alert.alert("Save failed", error.message || "Could not save item.");
    } finally {
      setSaving(false);
    }
  };

  const loadSavedDetails = async () => {
    try {
      const response = await axios.get(`${WARDROBE_URL}/`);

      const items = response.data.map((d: any) => ({
        id: d.item_id,
        type: d.type,
        color: d.color,
        gender: d.gender,
        season: d.season,
        material: d.material,
        originalImageUrl: d.original_image_url,
        processedImageUrl: d.processed_image_url,
        backImageUrl: d.back_processed_image_url || d.back_image_url,
        trendAnalysis: d.trend_analysis,
      }));

      setWardrobeItems(items);
    } catch (error) {
      console.log("Load wardrobe error:", error);
      Alert.alert("Error", "Could not load saved wardrobe details.");
    }
  };

  const startEditItem = (item: any) => {
    setEditingId(item.id);
    setEditForm({
      type: item.type ?? "",
      color: item.color ?? "",
      gender: item.gender ?? "",
      season: item.season ?? "",
      material: item.material ?? "",
    });
  };

  const cancelEditItem = () => {
    setEditingId(null);
  };

  const saveEditItem = async (itemId: string) => {
    try {
      setRowBusyId(itemId);

      await axios.patch(`${WARDROBE_URL}/${itemId}`, {
        type: editForm.type,
        color: editForm.color,
        gender: editForm.gender,
        season: editForm.season,
        material: editForm.material,
      });

      setWardrobeItems((prev) =>
        prev.map((it) => (it.id === itemId ? { ...it, ...editForm } : it))
      );

      setEditingId(null);
      Alert.alert("Updated", "Wardrobe item updated.");
    } catch (error) {
      console.log("Update wardrobe error:", error);
      Alert.alert("Update failed", "Could not update this item.");
    } finally {
      setRowBusyId(null);
    }
  };

  const requestDeleteItem = (itemId: string) => {
    setConfirmDeleteId(itemId);
  };

  const cancelDeleteItem = () => {
    setConfirmDeleteId(null);
  };

  const deleteWardrobeItem = async (itemId: string) => {
    try {
      setRowBusyId(itemId);

      await axios.delete(`${WARDROBE_URL}/${itemId}`);

      setWardrobeItems((prev) => prev.filter((it) => it.id !== itemId));
      setConfirmDeleteId(null);

      if (editingId === itemId) {
        setEditingId(null);
      }
    } catch (error: any) {
      console.log("Delete wardrobe error:", error);
      const message =
        error?.message || "Could not delete this item.";
      if (Platform.OS === "web" && typeof window !== "undefined") {
        window.alert("Delete failed: " + message);
      } else {
        Alert.alert("Delete failed", message);
      }
    } finally {
      setRowBusyId(null);
    }
  };

  const loadScheduledEvents = async () => {
    try {
      const response = await axios.get(`${WARDROBE_URL}/schedule`);

      const events = response.data.map((e: any) => ({
        id: e.event_id,
        eventName: e.event_name,
        eventDate: e.event_date,
        eventTime: e.event_time,
        notes: e.notes,
        clothingType: e.clothing_type,
        clothingColor: e.clothing_color,
        processedImageUrl: e.processed_image_url,
        trendSuggestion: e.trend_suggestion,
      }));

      setScheduledEvents(events);
    } catch (error) {
      console.log("Load events error:", error);
      Alert.alert("Error", "Could not load scheduled events.");
    }
  };

  const startEditEvent = (ev: any) => {
    setEditEventId(ev.id);
    setEventForm({
      eventName: ev.eventName ?? "",
      eventDate: ev.eventDate ?? "",
      eventTime: ev.eventTime ?? "",
      notes: ev.notes ?? "",
    });
  };

  const cancelEditEvent = () => setEditEventId(null);

  const saveEditEvent = async (id: string) => {
    try {
      setEventRowBusy(id);
      await axios.patch(`${WARDROBE_URL}/schedule/${id}`, {
        event_name: eventForm.eventName,
        event_date: eventForm.eventDate,
        event_time: eventForm.eventTime,
        notes: eventForm.notes,
      });
      setScheduledEvents((prev) =>
        prev.map((e) => (e.id === id ? { ...e, ...eventForm } : e))
      );
      setEditEventId(null);
      Alert.alert("Updated", "Dressing event updated.");
    } catch (error) {
      console.log("Update event error:", error);
      Alert.alert("Update failed", "Could not update this event.");
    } finally {
      setEventRowBusy(null);
    }
  };

  const requestDeleteEvent = (id: string) => setDelEventId(id);
  const cancelDeleteEvent = () => setDelEventId(null);

  const deleteEvent = async (id: string) => {
    try {
      setEventRowBusy(id);
      await axios.delete(`${WARDROBE_URL}/schedule/${id}`);
      setScheduledEvents((prev) => prev.filter((e) => e.id !== id));
      setDelEventId(null);
      if (editEventId === id) setEditEventId(null);
    } catch (error: any) {
      console.log("Delete event error:", error);
      const msg = error?.message || "Could not delete this event.";
      if (Platform.OS === "web" && typeof window !== "undefined") {
        window.alert("Delete failed: " + msg);
      } else {
        Alert.alert("Delete failed", msg);
      }
    } finally {
      setEventRowBusy(null);
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
      <StatusBar barStyle="dark-content" />

      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>E-Wardrobe AI</Text>
        <Text style={styles.subtitle}>
          AI clothing prediction, wardrobe storage, and dressing schedule
        </Text>

        <View style={styles.card}>
          <Text style={styles.sectionTitle}>Upload Clothing Images</Text>

          <Text style={styles.imageSideLabel}>Front side (used for prediction)</Text>

          <TouchableOpacity style={styles.uploadBox} onPress={pickImage}>
            {image ? (
              <Image source={{ uri: image.uri }} style={styles.previewImage} />
            ) : (
              <Text style={styles.uploadText}>Tap to choose image</Text>
            )}
          </TouchableOpacity>

          <TouchableOpacity style={styles.primaryButton} onPress={pickImage}>
            <Text style={styles.buttonText}>Choose Front Image</Text>
          </TouchableOpacity>

          <Text style={styles.imageSideLabel}>Back side (optional, save only)</Text>

          <TouchableOpacity style={styles.uploadBox} onPress={pickBackImage}>
            {backImage ? (
              <Image source={{ uri: backImage.uri }} style={styles.previewImage} />
            ) : (
              <Text style={styles.uploadText}>Tap to add back image</Text>
            )}
          </TouchableOpacity>

          <TouchableOpacity style={styles.secondaryButton} onPress={pickBackImage}>
            <Text style={styles.buttonText}>Choose Back Image (Optional)</Text>
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
            <Text style={styles.sectionTitle}>Background Removed Images</Text>
            <Text style={styles.imageSideLabel}>Front side</Text>
            <Image source={{ uri: processedImageUrl }} style={styles.previewImage} />
            {backImageUrl && (
              <>
                <Text style={styles.imageSideLabel}>Back side</Text>
                <Image source={{ uri: backImageUrl }} style={styles.previewImage} />
              </>
            )}
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

              <View style={styles.resultBox}>
                <Text style={styles.resultLabel}>Material</Text>
                <Text style={styles.resultValue}>{prediction.material}</Text>
              </View>
            </View>

            <ConfidenceBar label="Type Confidence" value={prediction.type_confidence} />
            <ConfidenceBar label="Color Confidence" value={prediction.color_confidence} />
            <ConfidenceBar label="Gender Confidence" value={prediction.gender_confidence} />
            <ConfidenceBar label="Season Confidence" value={prediction.season_confidence} />
            <ConfidenceBar label="Material Confidence" value={prediction.material_confidence} />

            {trendAnalysis?.matches?.length > 0 && (
              <View style={styles.trendPanel}>
                <Text style={styles.trendTitle}>Trend-Aware Suggestions</Text>
                <Text style={styles.trendSource}>{trendAnalysis.source}</Text>

                {trendAnalysis.matches.map((trend: any) => (
                  <View key={trend.keyword} style={styles.trendItem}>
                    <View style={styles.trendHeader}>
                      <Text style={styles.trendKeyword}>{trend.keyword}</Text>
                      <Text style={styles.trendScore}>{trend.score}%</Text>
                    </View>
                    <Text style={styles.trendMeta}>
                      Matched: {trend.matched_on.join(", ")}
                    </Text>
                    <Text style={styles.trendSuggestion}>{trend.suggestion}</Text>
                  </View>
                ))}
              </View>
            )}

            <TouchableOpacity
              style={[styles.saveButton, saving && styles.disabledButton]}
              onPress={saveWardrobeItem}
              disabled={saving}
            >
              <Text style={styles.buttonText}>
                {saving
                  ? "Saving..."
                  : eventName.trim() && eventDate.trim() && eventTime.trim()
                  ? "Save to Wardrobe & Schedule"
                  : "Save to Wardrobe"}
              </Text>
            </TouchableOpacity>
          </View>
        )}

        {prediction && (
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Schedule Dressing Event (optional)</Text>

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

            <Text style={styles.savedInfo}>
              Fill event name, date and time to save this dressing event together with the item. Leave blank to save the item only.
            </Text>
          </View>
        )}

        <View style={styles.card}>
          <Text style={styles.sectionTitle}>Saved Wardrobe Items</Text>

          <TouchableOpacity
            style={styles.primaryButton}
            onPress={wardrobeItems.length > 0 ? () => setWardrobeItems([]) : loadSavedDetails}
          >
            <Text style={styles.buttonText}>
              {wardrobeItems.length > 0 ? "Hide Wardrobe Items" : "View Saved Wardrobe Items"}
            </Text>
          </TouchableOpacity>

          {wardrobeItems.map((item) => {
            const isEditing = editingId === item.id;
            const isBusy = rowBusyId === item.id;

            return (
              <View key={item.id} style={styles.savedCard}>
                {item.processedImageUrl && (
                  <Image source={{ uri: item.processedImageUrl }} style={styles.smallImage} />
                )}
                {item.backImageUrl && (
                  <>
                    <Text style={styles.savedInfo}>Back side:</Text>
                    <Image source={{ uri: item.backImageUrl }} style={styles.smallImage} />
                  </>
                )}

                {isEditing ? (
                  <>
                    <Text style={styles.editLabel}>Type</Text>
                    <TextInput
                      style={styles.input}
                      value={editForm.type}
                      onChangeText={(t) => setEditForm((f) => ({ ...f, type: t }))}
                      placeholder="Type"
                      placeholderTextColor="#94a3b8"
                    />

                    <Text style={styles.editLabel}>Color</Text>
                    <TextInput
                      style={styles.input}
                      value={editForm.color}
                      onChangeText={(t) => setEditForm((f) => ({ ...f, color: t }))}
                      placeholder="Color"
                      placeholderTextColor="#94a3b8"
                    />

                    <Text style={styles.editLabel}>Gender</Text>
                    <TextInput
                      style={styles.input}
                      value={editForm.gender}
                      onChangeText={(t) => setEditForm((f) => ({ ...f, gender: t }))}
                      placeholder="Gender"
                      placeholderTextColor="#94a3b8"
                    />

                    <Text style={styles.editLabel}>Season</Text>
                    <TextInput
                      style={styles.input}
                      value={editForm.season}
                      onChangeText={(t) => setEditForm((f) => ({ ...f, season: t }))}
                      placeholder="Season"
                      placeholderTextColor="#94a3b8"
                    />

                    <Text style={styles.editLabel}>Fabric</Text>
                    <TextInput
                      style={styles.input}
                      value={editForm.material}
                      onChangeText={(t) => setEditForm((f) => ({ ...f, material: t }))}
                      placeholder="Fabric"
                      placeholderTextColor="#94a3b8"
                    />

                    <View style={styles.cardActions}>
                      <TouchableOpacity
                        style={[
                          styles.saveButton,
                          styles.cardActionButton,
                          isBusy && styles.disabledButton,
                        ]}
                        onPress={() => saveEditItem(item.id)}
                        disabled={isBusy}
                      >
                        <Text style={styles.buttonText}>
                          {isBusy ? "Saving..." : "Save Changes"}
                        </Text>
                      </TouchableOpacity>

                      <TouchableOpacity
                        style={[styles.secondaryButton, styles.cardActionButton]}
                        onPress={cancelEditItem}
                        disabled={isBusy}
                      >
                        <Text style={styles.buttonText}>Cancel</Text>
                      </TouchableOpacity>
                    </View>
                  </>
                ) : (
                  <>
                    <Text style={styles.savedTitle}>{item.type}</Text>
                    <Text style={styles.savedInfo}>Color: {item.color}</Text>
                    <Text style={styles.savedInfo}>Gender: {item.gender}</Text>
                    <Text style={styles.savedInfo}>Season: {item.season}</Text>
                    {item.material ? (
                      <Text style={styles.savedInfo}>Fabric: {item.material}</Text>
                    ) : null}
                    {item.trendAnalysis?.matches?.length > 0 && (
                      <Text style={styles.savedInfo}>
                        Trend: {item.trendAnalysis.matches[0].keyword}
                      </Text>
                    )}
                    <Text style={styles.savedInfo}>Item ID: {item.id}</Text>

                    {confirmDeleteId === item.id ? (
                      <>
                        <Text style={styles.confirmText}>
                          Delete this item permanently?
                        </Text>
                        <View style={styles.cardActions}>
                          <TouchableOpacity
                            style={[
                              styles.deleteButton,
                              styles.cardActionButton,
                              isBusy && styles.disabledButton,
                            ]}
                            onPress={() => deleteWardrobeItem(item.id)}
                            disabled={isBusy}
                          >
                            <Text style={styles.buttonText}>
                              {isBusy ? "Deleting..." : "Confirm Delete"}
                            </Text>
                          </TouchableOpacity>

                          <TouchableOpacity
                            style={[styles.secondaryButton, styles.cardActionButton]}
                            onPress={cancelDeleteItem}
                            disabled={isBusy}
                          >
                            <Text style={styles.buttonText}>Keep</Text>
                          </TouchableOpacity>
                        </View>
                      </>
                    ) : (
                      <View style={styles.cardActions}>
                        <TouchableOpacity
                          style={[styles.editButton, styles.cardActionButton]}
                          onPress={() => startEditItem(item)}
                          disabled={isBusy}
                        >
                          <Text style={styles.buttonText}>Edit</Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                          style={[styles.deleteButton, styles.cardActionButton]}
                          onPress={() => requestDeleteItem(item.id)}
                          disabled={isBusy}
                        >
                          <Text style={styles.buttonText}>Delete</Text>
                        </TouchableOpacity>
                      </View>
                    )}
                  </>
                )}
              </View>
            );
          })}
        </View>

        <View style={styles.card}>
          <Text style={styles.sectionTitle}>Scheduled Dressing Events</Text>

          <TouchableOpacity
            style={styles.scheduleButton}
            onPress={scheduledEvents.length > 0 ? () => setScheduledEvents([]) : loadScheduledEvents}
          >
            <Text style={styles.buttonText}>
              {scheduledEvents.length > 0 ? "Hide Scheduled Events" : "View Scheduled Events"}
            </Text>
          </TouchableOpacity>

          {scheduledEvents.map((event) => {
            const isEditing = editEventId === event.id;
            const isBusy = eventRowBusy === event.id;

            return (
              <View key={event.id} style={styles.savedCard}>
                {event.processedImageUrl && (
                  <Image source={{ uri: event.processedImageUrl }} style={styles.smallImage} />
                )}

                {isEditing ? (
                  <>
                    <Text style={styles.editLabel}>Event name</Text>
                    <TextInput
                      style={styles.input}
                      value={eventForm.eventName}
                      onChangeText={(t) => setEventForm((f) => ({ ...f, eventName: t }))}
                      placeholder="Event name"
                      placeholderTextColor="#94a3b8"
                    />

                    <Text style={styles.editLabel}>Date</Text>
                    <TextInput
                      style={styles.input}
                      value={eventForm.eventDate}
                      onChangeText={(t) => setEventForm((f) => ({ ...f, eventDate: t }))}
                      placeholder="Date: 2026-05-20"
                      placeholderTextColor="#94a3b8"
                    />

                    <Text style={styles.editLabel}>Time</Text>
                    <TextInput
                      style={styles.input}
                      value={eventForm.eventTime}
                      onChangeText={(t) => setEventForm((f) => ({ ...f, eventTime: t }))}
                      placeholder="Time: 18:30"
                      placeholderTextColor="#94a3b8"
                    />

                    <Text style={styles.editLabel}>Notes</Text>
                    <TextInput
                      style={[styles.input, styles.notesInput]}
                      value={eventForm.notes}
                      onChangeText={(t) => setEventForm((f) => ({ ...f, notes: t }))}
                      placeholder="Notes"
                      placeholderTextColor="#94a3b8"
                      multiline
                    />

                    <View style={styles.cardActions}>
                      <TouchableOpacity
                        style={[
                          styles.saveButton,
                          styles.cardActionButton,
                          isBusy && styles.disabledButton,
                        ]}
                        onPress={() => saveEditEvent(event.id)}
                        disabled={isBusy}
                      >
                        <Text style={styles.buttonText}>
                          {isBusy ? "Saving..." : "Save Changes"}
                        </Text>
                      </TouchableOpacity>

                      <TouchableOpacity
                        style={[styles.secondaryButton, styles.cardActionButton]}
                        onPress={cancelEditEvent}
                        disabled={isBusy}
                      >
                        <Text style={styles.buttonText}>Cancel</Text>
                      </TouchableOpacity>
                    </View>
                  </>
                ) : (
                  <>
                    <Text style={styles.savedTitle}>{event.eventName}</Text>
                    <Text style={styles.savedInfo}>Date: {event.eventDate}</Text>
                    <Text style={styles.savedInfo}>Time: {event.eventTime}</Text>
                    <Text style={styles.savedInfo}>Dress: {event.clothingType}</Text>
                    <Text style={styles.savedInfo}>Color: {event.clothingColor}</Text>
                    <Text style={styles.savedInfo}>Notes: {event.notes}</Text>
                    {event.trendSuggestion?.suggestion && (
                      <Text style={styles.savedInfo}>
                        Suggestion: {event.trendSuggestion.suggestion}
                      </Text>
                    )}

                    {delEventId === event.id ? (
                      <>
                        <Text style={styles.confirmText}>
                          Delete this event permanently?
                        </Text>
                        <View style={styles.cardActions}>
                          <TouchableOpacity
                            style={[
                              styles.deleteButton,
                              styles.cardActionButton,
                              isBusy && styles.disabledButton,
                            ]}
                            onPress={() => deleteEvent(event.id)}
                            disabled={isBusy}
                          >
                            <Text style={styles.buttonText}>
                              {isBusy ? "Deleting..." : "Confirm Delete"}
                            </Text>
                          </TouchableOpacity>

                          <TouchableOpacity
                            style={[styles.secondaryButton, styles.cardActionButton]}
                            onPress={cancelDeleteEvent}
                            disabled={isBusy}
                          >
                            <Text style={styles.buttonText}>Keep</Text>
                          </TouchableOpacity>
                        </View>
                      </>
                    ) : (
                      <View style={styles.cardActions}>
                        <TouchableOpacity
                          style={[styles.editButton, styles.cardActionButton]}
                          onPress={() => startEditEvent(event)}
                          disabled={isBusy}
                        >
                          <Text style={styles.buttonText}>Edit</Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                          style={[styles.deleteButton, styles.cardActionButton]}
                          onPress={() => requestDeleteEvent(event.id)}
                          disabled={isBusy}
                        >
                          <Text style={styles.buttonText}>Delete</Text>
                        </TouchableOpacity>
                      </View>
                    )}
                  </>
                )}
              </View>
            );
          })}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

/* Tailwind violet / slate palette - matches the Recommend screen. */
const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: "#F3F0FF",
  },

  container: {
    padding: 18,
    paddingBottom: 48,
    maxWidth: 640,
    width: "100%",
    alignSelf: "center",
  },

  title: {
    fontSize: 32,
    fontWeight: "800",
    color: "#2D1B69",
    textAlign: "center",
    letterSpacing: -0.6,
    marginTop: 24,
  },

  subtitle: {
    fontSize: 14,
    color: "#6B7280",
    textAlign: "center",
    lineHeight: 20,
    marginBottom: 26,
    marginTop: 10,
    paddingHorizontal: 12,
  },

  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 24,
    padding: 20,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#EDE9FE",
    shadowColor: "#6D28D9",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
    elevation: 4,
  },

  sectionTitle: {
    fontSize: 17,
    fontWeight: "800",
    color: "#1F2937",
    letterSpacing: -0.2,
    marginBottom: 16,
  },

  uploadBox: {
    height: 280,
    borderRadius: 16,
    borderWidth: 1.5,
    borderStyle: "dashed",
    borderColor: "#DDD6FE",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#F5F3FF",
    overflow: "hidden",
    marginBottom: 14,
  },

  uploadText: {
    color: "#9CA3AF",
    fontSize: 15,
    fontWeight: "500",
  },

  previewImage: {
    width: "100%",
    height: 280,
    borderRadius: 16,
    resizeMode: "contain",
    backgroundColor: "#EDE9FE",
  },

  primaryButton: {
    backgroundColor: "#7C3AED",
    paddingVertical: 15,
    borderRadius: 16,
    alignItems: "center",
    marginBottom: 12,
    shadowColor: "#7C3AED",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.35,
    shadowRadius: 10,
    elevation: 3,
  },

  secondaryButton: {
    backgroundColor: "#8B5CF6",
    paddingVertical: 15,
    borderRadius: 16,
    alignItems: "center",
    marginBottom: 20,
    borderWidth: 1,
    borderColor: "#DDD6FE",
  },

  imageSideLabel: {
    color: "#7C3AED",
    fontSize: 12,
    fontWeight: "800",
    textTransform: "uppercase",
    letterSpacing: 1.5,
    marginBottom: 10,
  },

  predictButton: {
    backgroundColor: "#7C3AED",
    paddingVertical: 15,
    borderRadius: 16,
    alignItems: "center",
    shadowColor: "#7C3AED",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.32,
    shadowRadius: 10,
    elevation: 3,
  },

  saveButton: {
    backgroundColor: "#10B981",
    paddingVertical: 15,
    borderRadius: 16,
    alignItems: "center",
    marginTop: 18,
    shadowColor: "#10B981",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
    elevation: 3,
  },

  scheduleButton: {
    backgroundColor: "#8B5CF6",
    paddingVertical: 15,
    borderRadius: 16,
    alignItems: "center",
    marginTop: 8,
    shadowColor: "#8B5CF6",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.32,
    shadowRadius: 10,
    elevation: 3,
  },

  disabledButton: {
    backgroundColor: "#C4B5FD",
    shadowOpacity: 0,
    elevation: 0,
  },

  buttonText: {
    color: "#ffffff",
    fontSize: 15,
    fontWeight: "800",
    letterSpacing: 0.3,
  },

  resultGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginBottom: 22,
  },

  resultBox: {
    width: "48%",
    backgroundColor: "#F5F3FF",
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: "#EDE9FE",
  },

  resultLabel: {
    color: "#7C3AED",
    fontSize: 11,
    fontWeight: "800",
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: 6,
  },

  resultValue: {
    color: "#1F2937",
    fontSize: 17,
    fontWeight: "800",
    letterSpacing: -0.2,
  },

  confidenceBox: {
    marginBottom: 14,
  },

  confidenceTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 7,
  },

  confidenceLabel: {
    color: "#4B5563",
    fontSize: 13,
    fontWeight: "500",
  },

  confidenceValue: {
    color: "#7C3AED",
    fontSize: 13,
    fontWeight: "700",
  },

  progressTrack: {
    height: 8,
    backgroundColor: "#EDE9FE",
    borderRadius: 999,
    overflow: "hidden",
  },

  progressFill: {
    height: "100%",
    backgroundColor: "#7C3AED",
    borderRadius: 999,
  },

  trendPanel: {
    backgroundColor: "#FAFAFA",
    borderRadius: 16,
    padding: 16,
    marginTop: 8,
    marginBottom: 2,
    borderWidth: 1,
    borderColor: "#EDE9FE",
    borderLeftWidth: 3,
    borderLeftColor: "#7C3AED",
  },

  trendTitle: {
    color: "#7C3AED",
    fontSize: 16,
    fontWeight: "800",
    letterSpacing: -0.2,
    marginBottom: 4,
  },

  trendSource: {
    color: "#9CA3AF",
    fontSize: 11,
    marginBottom: 14,
  },

  trendItem: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 13,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: "#EDE9FE",
  },

  trendHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 10,
    marginBottom: 5,
  },

  trendKeyword: {
    color: "#1F2937",
    flex: 1,
    fontSize: 14,
    fontWeight: "800",
    textTransform: "capitalize",
  },

  trendScore: {
    color: "#10B981",
    fontSize: 14,
    fontWeight: "800",
  },

  trendMeta: {
    color: "#7C3AED",
    fontSize: 11,
    fontWeight: "500",
    marginBottom: 6,
  },

  trendSuggestion: {
    color: "#4B5563",
    fontSize: 13,
    lineHeight: 19,
  },

  eventSuggestionBox: {
    backgroundColor: "#F0FDF4",
    borderRadius: 14,
    padding: 15,
    marginTop: 14,
    borderWidth: 1,
    borderColor: "#10B981",
  },

  eventSuggestionTitle: {
    color: "#065F46",
    fontSize: 14,
    fontWeight: "800",
    letterSpacing: 0.2,
    marginBottom: 6,
  },

  eventSuggestionText: {
    color: "#065F46",
    fontSize: 13,
    lineHeight: 19,
  },

  input: {
    backgroundColor: "#F5F3FF",
    color: "#1F2937",
    borderWidth: 1.5,
    borderColor: "#DDD6FE",
    borderRadius: 14,
    paddingVertical: 13,
    paddingHorizontal: 14,
    marginBottom: 12,
    fontSize: 15,
  },

  notesInput: {
    height: 96,
    textAlignVertical: "top",
  },

  savedText: {
    color: "#065F46",
    fontWeight: "600",
    marginTop: 12,
    fontSize: 13,
  },

  savedCard: {
    backgroundColor: "#F9FAFB",
    borderRadius: 14,
    padding: 15,
    marginTop: 12,
    borderWidth: 1,
    borderColor: "#EDE9FE",
  },

  smallImage: {
    width: "100%",
    height: 180,
    borderRadius: 12,
    resizeMode: "contain",
    backgroundColor: "#EDE9FE",
    marginBottom: 12,
  },

  savedTitle: {
    color: "#1F2937",
    fontSize: 17,
    fontWeight: "800",
    letterSpacing: -0.2,
    marginBottom: 8,
  },

  savedInfo: {
    color: "#4B5563",
    fontSize: 13,
    lineHeight: 20,
    marginBottom: 3,
  },

  cardActions: {
    flexDirection: "row",
    gap: 10,
    marginTop: 14,
  },

  cardActionButton: {
    flex: 1,
    marginTop: 0,
    marginBottom: 0,
  },

  editButton: {
    backgroundColor: "#7C3AED",
    paddingVertical: 12,
    borderRadius: 14,
    alignItems: "center",
  },

  deleteButton: {
    backgroundColor: "#EF4444",
    paddingVertical: 12,
    borderRadius: 14,
    alignItems: "center",
  },

  editLabel: {
    color: "#7C3AED",
    fontSize: 11,
    fontWeight: "800",
    textTransform: "uppercase",
    letterSpacing: 1,
    marginTop: 4,
    marginBottom: 6,
  },

  confirmText: {
    color: "#DC2626",
    fontSize: 13,
    fontWeight: "600",
    marginTop: 12,
  },
});
