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
}

export type FeedbackMap = Record<string, 'liked' | 'skipped'>;

type Listener = () => void;

class WardrobeStore {
  history: HistoryEntry[] = [];
  feedback: FeedbackMap = {};
  private listeners = new Set<Listener>();

  addHistory(entry: HistoryEntry) {
    this.history = [entry, ...this.history.slice(0, 9)];
    this.notify();
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
      this.history = [{ ...this.history[0], feedback }, ...this.history.slice(1)];
      this.notify();
    }
  }

  // Toggle like / skip on a single outfit inside a specific history entry.
  setHistoryFeedback(entryId: string, outfit: string, action: 'liked' | 'skipped') {
    this.history = this.history.map(h => {
      if (h.id !== entryId) return h;
      const fb: FeedbackMap = { ...(h.feedback ?? {}) };
      if (fb[outfit] === action) delete fb[outfit];
      else fb[outfit] = action;
      return { ...h, feedback: fb };
    });
    this.notify();
  }

  clearHistory() {
    this.history = [];
    this.feedback = {};
    this.notify();
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
