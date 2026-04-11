/**
 * eWardrobeAI — Main Application State Machine
 *
 * Flow:  STEP_1 (Selfie) → STEP_2 (Measurements) → STEP_3 (AI Processing) → STEP_4 (Try-On)
 *
 * Key responsibilities:
 *  - Drive the 4-step wizard
 *  - Submit selfie + measurements to backend
 *  - Animate stage progress during AI processing
 *  - Populate Try-On UI with pipeline response
 *  - Bridge outfit selection ↔ 3D renderer
 */

'use strict';

const API = 'http://localhost:8000';

/* ── App State ──────────────────────────────────────────────────────────── */
const State = {
  currentStep:     1,
  selfieFile:      null,         // File object
  selfieDataUrl:   null,         // base64 preview
  pipelineResult:  null,         // full API response
  selectedOutfit:  0,            // index into recommendations array
  activeAnimation: 'idle',
};

/* ── Accuracy Dashboard Redirect ────────────────────────────────────────── */
function openAccuracyDashboard() {
  // Persist the selfie so the accuracy page can display it
  if (State.selfieDataUrl) {
    try { sessionStorage.setItem('ewai_selfie', State.selfieDataUrl); } catch(_) {}
  }
  // Open accuracy dashboard with autorun flag
  window.location.href = '/accuracy?autorun=all';
}

/* ── Step Navigation ────────────────────────────────────────────────────── */
function goToStep(n) {
  document.querySelectorAll('.step-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.step-item').forEach((it, i) => {
    it.classList.toggle('active', i + 1 === n);
    it.classList.toggle('done',   i + 1 < n);
  });
  document.getElementById(`step-${n}`).classList.add('active');
  State.currentStep = n;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ── Step 1: Selfie Confirmed ───────────────────────────────────────────── */
function onSelfieReady(file, dataUrl) {
  State.selfieFile    = file;
  State.selfieDataUrl = dataUrl;

  // Show thumbnail
  const thumb = document.getElementById('selfie-thumb');
  thumb.src   = dataUrl;
  document.getElementById('selfie-confirmed').style.display = 'block';
  document.getElementById('upload-zone').style.display      = 'none';
  document.getElementById('retake-btn').style.display       = 'inline-flex';

  // Enable next
  document.getElementById('step1-next').disabled = false;
  flashSuccess();
}

function retakeSelfie() {
  State.selfieFile  = null;
  State.selfieDataUrl = null;
  document.getElementById('selfie-confirmed').style.display = 'none';
  document.getElementById('upload-zone').style.display      = '';
  document.getElementById('retake-btn').style.display       = 'none';
  document.getElementById('step1-next').disabled            = true;
  stopCamera();
}

/* ── Step 2: Measurement live validation & preview ──────────────────────── */
const BOUNDS = {
  shoulder: [30, 65], chest: [60, 160], waist: [50, 150], height: [120, 230]
};
const SIZE_THRESHOLDS = [[82,'XS'],[88,'S'],[96,'M'],[104,'L'],[112,'XL'],[124,'XXL'],[Infinity,'XXXL']];

function validateMeasure(name, value, min, max) {
  const el  = document.getElementById(`v-${name}`);
  const num = parseFloat(value);
  if (isNaN(num))        { el.className='validation-msg error'; el.textContent='Required'; return; }
  if (num < min)         { el.className='validation-msg error'; el.textContent=`Min ${min} cm`; return; }
  if (num > max)         { el.className='validation-msg error'; el.textContent=`Max ${max} cm`; return; }
  el.className='validation-msg ok'; el.textContent='✓';
  updateMeasurePreview();
}

function updateMeasurePreview() {
  const get = id => parseFloat(document.getElementById(id).value) || 0;
  const shoulder = get('m-shoulder');
  const chest    = get('m-chest');
  const waist    = get('m-waist');
  const height   = get('m-height');

  // Bar widths (normalised relative to max)
  const setBar = (id, val, max) => {
    document.getElementById(`bar-${id}`).style.width = Math.min(100, (val / max) * 100) + '%';
    document.getElementById(`bv-${id}`).textContent  = val + ' cm';
  };
  setBar('shoulder', shoulder, 65);
  setBar('chest',    chest,   160);
  setBar('waist',    waist,   150);
  setBar('height',   height,  230);

  // Estimated size
  const size = SIZE_THRESHOLDS.find(([t]) => chest <= t)?.[1] || 'XXXL';
  document.getElementById('est-size').textContent = size;

  // Body type heuristic
  const hip = waist + 25;
  const wDef = (chest + hip) / 2 - waist;
  const sHRatio = hip / (shoulder * 2.3);
  let bodyType = 'Rectangle';
  if (wDef > 9 && Math.abs(sHRatio - 1) < 0.08) bodyType = 'Hourglass';
  else if (sHRatio < 0.87) bodyType = 'Inv. Triangle';
  else if (sHRatio > 1.13) bodyType = 'Pear';
  document.getElementById('est-bodytype').textContent = bodyType;
}

/* ── Step 3: AI Processing Simulation + API Call ────────────────────────── */
const STAGE_MSGS = [
  { title: 'Validating body measurements…', sub: 'Stage 1 — Body Calibration' },
  { title: 'Analysing facial landmarks…',   sub: 'Stage 2 — MediaPipe (468 pts) + CNN (15 keypoints)' },
  { title: 'Matching outfits from wardrobe…', sub: 'Stage 3 — NisfaMatchmaking + RaveehaOrganisationalDB' },
  { title: 'Generating your 3D avatar…',    sub: 'Stage 4 — Avatar scaling + Mixamo animation setup' },
];

async function startProcessing() {
  if (!State.selfieFile) { alert('Please capture or upload a selfie first.'); return; }

  // Validate measurements
  for (const [name, [min, max]] of Object.entries(BOUNDS)) {
    const val = parseFloat(document.getElementById(`m-${name}`).value);
    if (isNaN(val) || val < min || val > max) {
      alert(`Please fix the ${name} measurement (valid range: ${min}–${max} cm).`);
      return;
    }
  }

  goToStep(3);
  resetStageUI();

  // Draw selfie on landmark canvas
  drawSelfieOnCanvas();

  // Run stage animations alongside the API call
  const apiPromise   = callTryOnAPI();
  const animPromise  = runStageAnimations(apiPromise);

  try {
    const result = await apiPromise;
    await animPromise;          // ensure animations finish
    State.pipelineResult = result;
    populateTryOnUI(result);
    goToStep(4);
    flashSuccess();
  } catch (err) {
    showProcessingError(err.message);
  }
}

async function callTryOnAPI() {
  const fd = new FormData();
  fd.append('selfie',            State.selfieFile);
  fd.append('shoulder_width_cm', document.getElementById('m-shoulder').value);
  fd.append('chest_cm',          document.getElementById('m-chest').value);
  fd.append('waist_cm',          document.getElementById('m-waist').value);
  fd.append('height_cm',         document.getElementById('m-height').value);
  const hip    = document.getElementById('m-hip').value;
  const inseam = document.getElementById('m-inseam').value;
  if (hip)    fd.append('hip_cm',    hip);
  if (inseam) fd.append('inseam_cm', inseam);
  fd.append('styles',        document.getElementById('m-styles').value);
  fd.append('occasion',      document.getElementById('m-occasion').value);
  fd.append('animation_key', State.activeAnimation);
  fd.append('top_k',         5);

  const res = await fetch(`${API}/api/tryon`, { method: 'POST', body: fd });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail?.error || `Server error ${res.status}`);
  }
  return res.json();
}

/* Stage animation driver — runs in parallel with the API call */
async function runStageAnimations(apiPromise) {
  const DURATIONS = [800, 1400, 900, 700];   // ms per stage
  let apiDone     = false;
  apiPromise.then(() => { apiDone = true; }).catch(() => { apiDone = true; });

  for (let i = 0; i < 4; i++) {
    setStageRunning(i + 1);
    const { title, sub } = STAGE_MSGS[i];
    document.getElementById('proc-title').textContent = title;
    document.getElementById('proc-sub').textContent   = sub;

    // Animate progress bar
    const fill = document.getElementById(`sb-${i+1}`);
    const start = performance.now();
    const dur   = DURATIONS[i];
    await new Promise(resolve => {
      function tick(now) {
        const pct = Math.min(100, ((now - start) / dur) * 100);
        fill.style.width = pct + '%';
        if (pct < 100) requestAnimationFrame(tick);
        else resolve();
      }
      requestAnimationFrame(tick);
    });

    setStageComplete(i + 1);
    await sleep(120);
  }

  // Wait for API if still running
  if (!apiDone) {
    document.getElementById('proc-title').textContent = 'Finalising recommendations…';
    document.getElementById('proc-sub').textContent   = 'Applying colour harmony scoring';
    await apiPromise.catch(() => {});
  }
}

function setStageRunning(n) {
  const row = document.getElementById(`stage-${n}`);
  const ss  = document.getElementById(`ss-${n}`);
  row.classList.add('active');
  ss.className   = 'stage-status running';
  ss.textContent = 'Running…';
}
function setStageComplete(n) {
  const row = document.getElementById(`stage-${n}`);
  const ss  = document.getElementById(`ss-${n}`);
  row.classList.remove('active');
  row.classList.add('done');
  ss.className   = 'stage-status done';
  ss.textContent = '✓ Done';
  document.getElementById(`sb-${n}`).style.width = '100%';
}
function resetStageUI() {
  for (let i = 1; i <= 4; i++) {
    const row = document.getElementById(`stage-${i}`);
    const ss  = document.getElementById(`ss-${i}`);
    row.className = 'stage-row';
    ss.className  = 'stage-status pending';
    ss.textContent = 'Pending';
    document.getElementById(`sb-${i}`).style.width = '0%';
  }
}

function drawSelfieOnCanvas() {
  const wrap   = document.getElementById('lm-wrap');
  const canvas = document.getElementById('landmark-canvas');
  if (!State.selfieDataUrl) return;
  wrap.style.display = 'block';
  const ctx = canvas.getContext('2d');
  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, 220, 220);
    // Circular clip
    ctx.save();
    ctx.beginPath();
    ctx.arc(110, 110, 108, 0, Math.PI * 2);
    ctx.clip();
    ctx.drawImage(img, 0, 0, 220, 220);
    ctx.restore();
    // Animate scanning dots
    animateLandmarkScan(ctx);
  };
  img.src = State.selfieDataUrl;
}

let _scanAnimId = null;
function animateLandmarkScan(ctx) {
  let tick = 0;
  function frame() {
    tick++;
    // Draw random green dots simulating landmark detection
    for (let i = 0; i < 3; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r     = 20 + Math.random() * 85;
      const x     = 110 + Math.cos(angle) * r;
      const y     = 110 + Math.sin(angle) * r;
      ctx.beginPath();
      ctx.arc(x, y, 1.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(124,111,255,${0.4 + Math.random() * 0.6})`;
      ctx.fill();
    }
    if (tick < 120) _scanAnimId = requestAnimationFrame(frame);
  }
  if (_scanAnimId) cancelAnimationFrame(_scanAnimId);
  frame();
}

function showProcessingError(msg) {
  document.getElementById('proc-title').textContent = '❌ Processing Failed';
  document.getElementById('proc-sub').textContent   = msg;
  document.querySelector('.ai-pulse').style.background =
    'radial-gradient(circle, rgba(244,67,54,0.3) 0%, transparent 70%)';
}

/* ── Step 4: Populate Try-On UI ─────────────────────────────────────────── */
function populateTryOnUI(data) {
  // Sizing profile
  const sp = data.sizingProfile;
  if (sp) {
    document.getElementById('info-size').textContent  = sp.standardSize;
    document.getElementById('info-body').textContent  = sp.bodyType.replace(/_/g,' ');
    document.getElementById('info-whr').textContent   = sp.waistHipRatio?.toFixed(3) || '—';
    document.getElementById('info-torso').textContent = sp.torsoLengthCm?.toFixed(1) + ' cm';
    document.getElementById('avatar-size').textContent = sp.standardSize;
    document.getElementById('avatar-body').textContent = sp.bodyType.replace(/_/g,' ');
    document.getElementById('est-size').textContent    = sp.standardSize;
    document.getElementById('est-bodytype').textContent= sp.bodyType.replace(/_/g,' ');
  }

  // Face analysis
  const fa = data.faceAnalysis;
  if (fa) {
    document.getElementById('info-468').textContent = fa.landmark468Count;
    document.getElementById('info-15').textContent  = fa.landmark15Count;
    document.getElementById('info-ied').textContent = fa.interEyeDistPx?.toFixed(1) + ' px';
    document.getElementById('info-yaw').textContent = fa.yawDeg?.toFixed(1) + '°';
    document.getElementById('info-tex').textContent = fa.hasFaceTexture ? '✅ Applied' : '⚠️ None';
  }

  // Scale params
  const sc = data.renderPayload?.scaleParams;
  if (sc) {
    document.getElementById('sc-height').textContent   = (sc.globalY * 100).toFixed(0) + '%';
    document.getElementById('sc-chest').textContent    = (sc.chestX  * 100).toFixed(0) + '%';
    document.getElementById('sc-waist').textContent    = (sc.waistX  * 100).toFixed(0) + '%';
    document.getElementById('sc-shoulder').textContent = (sc.shoulderX * 100).toFixed(0) + '%';
    document.getElementById('sc-leg').textContent      = (sc.legY    * 100).toFixed(0) + '%';
  }

  // Face preview
  if (State.selfieDataUrl) {
    const fp    = document.getElementById('face-preview-img');
    fp.src      = State.selfieDataUrl;
    fp.style.display = 'block';
  }

  // Wardrobe status
  loadWardrobeSummary();

  // Outfit list
  renderOutfitList(data.recommendations);

  // Load primary outfit into 3D renderer
  if (data.renderPayload && window.eWardrobeRenderer) {
    setCanvasLoading('Loading 3D avatar…');
    window.eWardrobeRenderer.loadPayload(data.renderPayload, () => hideCanvasLoading());
  }

  // Header status
  loadWardrobeSummary();
}

/* ── Outfit List ─────────────────────────────────────────────────────────── */
function renderOutfitList(recs) {
  const container = document.getElementById('outfit-list');
  if (!recs || !recs.length) {
    container.innerHTML = '<div class="text-muted" style="padding:12px;font-size:0.78rem">No available outfits found.</div>';
    return;
  }

  container.innerHTML = recs.map((rec, i) => `
    <div class="outfit-card ${i === 0 ? 'active' : ''}" onclick="selectOutfit(${i}, this)">
      <div class="outfit-num">Outfit ${i + 1} of ${recs.length}</div>
      <div class="outfit-name">${rec.name}</div>
      <div class="outfit-style">${rec.style.replace(/_/g,' ')} · ${rec.occasion}</div>
      <div class="outfit-items">
        ${rec.items.map(item => `<span class="outfit-tag">${item.name}</span>`).join('')}
      </div>
      <div class="score-bar">
        <div class="score-fill" style="width:${Math.round(rec.score * 100)}%"></div>
      </div>
      <div class="score-label">Match score: ${(rec.score * 100).toFixed(0)}%</div>
    </div>
  `).join('');

  // Show first outfit's garments in info panel
  if (recs.length) showOutfitDetail(recs[0].items);
}

function selectOutfit(idx, el) {
  document.querySelectorAll('.outfit-card').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  State.selectedOutfit = idx;

  const payloads = [State.pipelineResult?.renderPayload,
                    ...(State.pipelineResult?.alternatePayloads || [])];

  if (payloads[idx] && window.eWardrobeRenderer) {
    setCanvasLoading('Changing outfit…');
    window.eWardrobeRenderer.loadPayload(payloads[idx], () => hideCanvasLoading());
  }

  const items = State.pipelineResult?.recommendations[idx]?.items;
  if (items) showOutfitDetail(items);
}

function showOutfitDetail(items) {
  const COLOUR_MAP = {
    white:'#FFFFFF', light_blue:'#ADD8E6', navy:'#001F5B', black:'#1A1A1A',
    charcoal:'#36454F', khaki:'#C3B091', olive:'#808000', emerald:'#50C878',
    terracotta:'#E2725B', dark_indigo:'#1B1464', mid_grey:'#888888',
    cobalt_blue:'#0047AB', burgundy:'#800020', deep_red:'#8B0000',
    champagne:'#F7E7CE', navy_blue:'#001F5B', forest_green:'#228B22',
  };

  document.getElementById('outfit-items-detail').innerHTML = items.map(item => {
    const colour = COLOUR_MAP[item.colours?.[0]] || '#6C63FF';
    return `
      <div class="garment-row">
        <div class="garment-swatch" style="background:${colour};border:1px solid rgba(255,255,255,0.1)"></div>
        <div class="garment-info">
          <div class="gname">${item.name}</div>
          <div class="gcat">${item.category} · <span class="status-badge badge-clean" style="font-size:0.6rem">Clean</span></div>
        </div>
      </div>
    `;
  }).join('');
}

/* ── Animation Controls ─────────────────────────────────────────────────── */
function setAnim(key, el) {
  document.querySelectorAll('.anim-btn').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
  State.activeAnimation = key;
  if (window.eWardrobeRenderer) window.eWardrobeRenderer.setAnimation(key);
}

/* ── Canvas Loading State ───────────────────────────────────────────────── */
function setCanvasLoading(msg) {
  const el = document.getElementById('canvas-loading');
  if (el) { el.style.display = 'flex'; document.getElementById('canvas-loading-text').textContent = msg || 'Loading…'; }
}
function hideCanvasLoading() {
  const el = document.getElementById('canvas-loading');
  if (el) el.style.display = 'none';
}

/* ── Wardrobe Summary ────────────────────────────────────────────────────── */
async function loadWardrobeSummary() {
  try {
    const res  = await fetch(`${API}/api/wardrobe/summary`);
    const data = await res.json();
    const c = data['Clean'] || 0, d = data['Dirty'] || 0, l = data['In Laundry'] || 0;
    document.getElementById('hdr-clean').textContent  = c;
    document.getElementById('hdr-dirty').textContent  = d;
    document.getElementById('ws-clean').textContent   = `${c} Clean`;
    document.getElementById('ws-dirty').textContent   = `${d} Dirty`;
    document.getElementById('ws-laundry').textContent = `${l} Laundry`;
  } catch(_) {}
}

/* ── Misc Helpers ────────────────────────────────────────────────────────── */
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function flashSuccess() {
  const el = document.createElement('div');
  el.className = 'success-flash';
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 700);
}

/* ── Init ────────────────────────────────────────────────────────────────── */
window.addEventListener('load', () => {
  loadWardrobeSummary();
  setInterval(loadWardrobeSummary, 30000);
  updateMeasurePreview();

  // Auto-hide canvas loading once renderer is ready
  setTimeout(() => {
    if (document.getElementById('canvas-loading')?.style.display !== 'none' &&
        State.currentStep !== 4) {
      hideCanvasLoading();
    }
  }, 3000);
});
