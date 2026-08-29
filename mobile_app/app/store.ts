import { RECOMMENDATION_URL } from '../constants/api';

const HISTORY_URL = `${RECOMMENDATION_URL}/history`;

export interface RecommendationDetail {
  outfit: string;
  item_id?: string;
  confidence: string;
  fabric: string;
  color?: string;
  price?: number;
  category?: string;
  image_url: string;
  reason?: string;
  combination?: string;
  score?: number;
}

export interface ApiResponse {
  event_class: string;
  location_detected: string;
  weather: string;
  recommendations: RecommendationDetail[];
  from_cache?: boolean;
}

export interface HistoryEntry {
  id: string;
  occasion: string;
  event_class: string;
  location: string;
  weather: string;
  time: string;
  data: ApiResponse;
  feedback?: FeedbackMap;
  note?: string;
}

export type FeedbackMap = Record<string, 'liked' | 'skipped'>;

type Listener = () => void;

class WardrobeStore {
  history: HistoryEntry[] = [];
  feedback: FeedbackMap = {};
  private listeners = new Set<Listener>();

  // Pull the saved search history from the backend (MongoDB). Best-effort:
  // if the server can't be reached the in-memory history is left untouched.
  async hydrate() {
    try {
      const res = await fetch(HISTORY_URL);
      if (!res.ok) return;
      const rows = await res.json();
      if (!Array.isArray(rows)) return;
      this.history = rows.map((d: any): HistoryEntry => ({
        id: d.rec_id ?? d.id ?? Date.now().toString(),
        occasion: d.occasion ?? '',
        event_class: d.event_class ?? '',
        location: d.location ?? '',
        weather: d.weather ?? '',
        time: d.time ?? '',
        data: d.data ?? { event_class: '', location_detected: '', weather: '', recommendations: [] },
        feedback: d.feedback ?? {},
        note: d.note ?? undefined,
      }));
      this.notify();
    } catch { /* offline - keep whatever is in memory */ }
  }

  addHistory(entry: HistoryEntry) {
    this.history = [entry, ...this.history.slice(0, 9)];
    this.notify();
    this.persistHistory(entry.id);
  }

  private async persistHistory(localId: string) {
    const entry = this.history.find(h => h.id === localId);
    if (!entry) return;
    try {
      const res = await fetch(HISTORY_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          occasion: entry.occasion, event_class: entry.event_class,
          location: entry.location, weather: entry.weather, time: entry.time,
          data: entry.data, feedback: entry.feedback ?? {}, note: entry.note ?? null,
        }),
      });
      if (!res.ok) return;
      const saved = await res.json();
      const serverId: string | undefined = saved?.rec_id;
      if (serverId) {
        this.history = this.history.map(h => (h.id === localId ? { ...h, id: serverId } : h));
        this.notify();
      }
    } catch { /* offline - entry stays local only */ }
  }

  private async patchHistory(entryId: string, path: 'feedback' | 'note', body: unknown) {
    if (!entryId.startsWith('r_')) return;   // never made it to the server
    try {
      await fetch(`${HISTORY_URL}/${entryId}/${path}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    } catch { /* offline - local state is still updated */ }
  }

  setFeedback(outfit: string, action: 'liked' | 'skipped') {
    this.feedback = { ...this.feedback, [outfit]: action };
    this.notify();
  }

  resetFeedback() {
    this.feedback = {};
    this.notify();
  }

  saveCurrentFeedbackToLatestHistory(feedback: FeedbackMap) {
    if (this.history.length > 0 && Object.keys(feedback).length > 0) {
      const top = this.history[0];
      this.history = [{ ...top, feedback }, ...this.history.slice(1)];
      this.notify();
      this.patchHistory(top.id, 'feedback', { feedback });
    }
  }

  // Toggle like / skip on a single outfit inside a specific history entry.
  setHistoryFeedback(entryId: string, outfit: string, action: 'liked' | 'skipped') {
    let sent: 'liked' | 'skipped' | 'none' = action;
    this.history = this.history.map(h => {
      if (h.id !== entryId) return h;
      const fb: FeedbackMap = { ...(h.feedback ?? {}) };
      if (fb[outfit] === action) { delete fb[outfit]; sent = 'none'; }
      else fb[outfit] = action;
      return { ...h, feedback: fb };
    });
    this.notify();
    this.patchHistory(entryId, 'feedback', { outfit, action: sent });
  }

  // Free-text feedback the user types for a whole history entry.
  setHistoryNote(entryId: string, note: string) {
    this.history = this.history.map(h =>
      h.id === entryId ? { ...h, note: note.trim() || undefined } : h
    );
    this.notify();
    this.patchHistory(entryId, 'note', { note: note.trim() });
  }

  clearHistory() {
    this.history = [];
    this.feedback = {};
    this.notify();
    fetch(HISTORY_URL, { method: 'DELETE' }).catch(() => {});
  }

  subscribe(fn: Listener) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private notify() {
    this.listeners.forEach(fn => fn());
  }
}

export const wardrobeStore = new WardrobeStore();
