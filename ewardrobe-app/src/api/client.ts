export const API_BASE = 'http://10.236.188.16:8000';

export interface TryOnPayload {
  selfie:            { uri: string; name: string; type: string };
  shoulder_width_cm: string;
  chest_cm:          string;
  waist_cm:          string;
  height_cm:         string;
  hip_cm?:           string;
  inseam_cm?:        string;
  styles:            string;
  occasion:          string;
  top_k:             number;
}

export async function callTryOn(payload: TryOnPayload) {
  const fd = new FormData();
  fd.append('selfie', payload.selfie as any);
  fd.append('shoulder_width_cm', payload.shoulder_width_cm);
  fd.append('chest_cm',          payload.chest_cm);
  fd.append('waist_cm',          payload.waist_cm);
  fd.append('height_cm',         payload.height_cm);
  if (payload.hip_cm)    fd.append('hip_cm',    payload.hip_cm);
  if (payload.inseam_cm) fd.append('inseam_cm', payload.inseam_cm);
  fd.append('styles',   payload.styles);
  fd.append('occasion', payload.occasion);
  fd.append('top_k',    String(payload.top_k));

  const res = await fetch(`${API_BASE}/api/tryon`, { method: 'POST', body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail?.error || `Server error ${res.status}`);
  }
  return res.json();
}

export async function fetchWardrobeSummary() {
  const res = await fetch(`${API_BASE}/api/wardrobe/summary`);
  if (!res.ok) throw new Error('Failed to fetch wardrobe');
  return res.json();
}

export async function runStageAccuracy(stage: number) {
  const res = await fetch(`${API_BASE}/api/accuracy/stage/${stage}`, { method: 'POST' });
  if (!res.ok) throw new Error(`Stage ${stage} eval failed`);
  return res.json();
}
