import { Platform } from "react-native";

// One parent server (backend/main.py) fronts every feature under a prefix.
// Android emulator reaches the host machine at 10.0.2.2.
export const API_BASE =
  Platform.OS === "web" ? "http://127.0.0.1:8000" : "http://10.0.2.2:8000";

export const CLASSIFICATION_URL = `${API_BASE}/classification`;
export const RECOMMENDATION_URL = `${API_BASE}/recommendation`;
export const ORGANIZATION_URL = `${API_BASE}/organization`;
export const WARDROBE_URL = `${API_BASE}/wardrobe`;
