import axios from 'axios';
import { useEffect, useState } from 'react';
import { Platform } from 'react-native';
import { AUTH_URL } from '../constants/api';

// ── token persistence ───────────────────────────────────────────────────────
// Web keeps the session across reloads via localStorage. On native we hold it
// in memory for the session (add @react-native-async-storage/async-storage
// later if native persistence is needed).

const TOKEN_KEY = 'ewardrobe.token';
let memoryToken: string | null = null;

const storage = {
  get(): string | null {
    if (Platform.OS === 'web') {
      try { return window.localStorage.getItem(TOKEN_KEY); } catch { return memoryToken; }
    }
    return memoryToken;
  },
  set(token: string | null) {
    memoryToken = token;
    if (Platform.OS === 'web') {
      try {
        if (token) window.localStorage.setItem(TOKEN_KEY, token);
        else window.localStorage.removeItem(TOKEN_KEY);
      } catch { /* private mode - memory only */ }
    }
  },
};

// ── types ───────────────────────────────────────────────────────────────────

export interface Profile {
  gender: string;
  age: number | null;
  city: string;
  style: string;
  avatar: string;
  bio: string;
}

export interface User {
  user_id: string;
  email: string;
  name: string;
  created_at?: string;
  profile: Profile;
}

type Listener = () => void;

async function readError(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === 'string') return body.detail;
    if (Array.isArray(body?.detail) && body.detail[0]?.msg) return body.detail[0].msg;
  } catch { /* no JSON body */ }
  return fallback;
}

// ── store ───────────────────────────────────────────────────────────────────

class AuthStore {
  token: string | null = null;
  user: User | null = null;
  ready = false;                       // finished the initial token check
  private listeners = new Set<Listener>();

  get isAuthed() { return !!this.token && !!this.user; }

  async init() {
    const token = storage.get();
    if (!token) { this.ready = true; this.notify(); return; }
    this.token = token;
    this.applyAxiosAuth();
    try {
      const res = await fetch(`${AUTH_URL}/me`, { headers: this.authHeaders() });
      if (res.ok) {
        this.user = (await res.json()).user;
      } else {
        this.token = null; storage.set(null); this.applyAxiosAuth();
      }
    } catch {
      // backend unreachable - keep the token, let the user retry later
    }
    this.ready = true;
    this.notify();
  }

  authHeaders(): Record<string, string> {
    return this.token ? { Authorization: `Bearer ${this.token}` } : {};
  }

  async register(email: string, password: string, name: string) {
    const res = await fetch(`${AUTH_URL}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, name }),
    });
    if (!res.ok) throw new Error(await readError(res, 'Could not create the account'));
    const data = await res.json();
    this.setSession(data.token, data.user);
  }

  async login(email: string, password: string) {
    const res = await fetch(`${AUTH_URL}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) throw new Error(await readError(res, 'Could not sign in'));
    const data = await res.json();
    this.setSession(data.token, data.user);
  }

  async updateProfile(changes: Partial<Profile> & { name?: string }) {
    const res = await fetch(`${AUTH_URL}/profile`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', ...this.authHeaders() },
      body: JSON.stringify(changes),
    });
    if (!res.ok) throw new Error(await readError(res, 'Could not save the profile'));
    this.user = (await res.json()).user;
    this.notify();
  }

  async logout() {
    try {
      await fetch(`${AUTH_URL}/logout`, { method: 'POST', headers: this.authHeaders() });
    } catch { /* ignore - clear locally regardless */ }
    this.setSession(null, null);
  }

  private setSession(token: string | null, user: User | null) {
    this.token = token;
    this.user = user;
    storage.set(token);
    this.applyAxiosAuth();
    this.notify();
  }

  // Keep the axios default header in sync so every feature request (wardrobe,
  // recommendation, organization, trends) is scoped to this account.
  private applyAxiosAuth() {
    if (this.token) axios.defaults.headers.common.Authorization = `Bearer ${this.token}`;
    else delete axios.defaults.headers.common.Authorization;
  }

  subscribe(fn: Listener) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private notify() {
    this.listeners.forEach(fn => fn());
  }
}

export const authStore = new AuthStore();

// React hook: re-render on any auth change.
export function useAuth() {
  const [, force] = useState(0);
  useEffect(() => authStore.subscribe(() => force(n => n + 1)), []);
  return authStore;
}
