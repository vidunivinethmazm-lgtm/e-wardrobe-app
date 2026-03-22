"""
eWardrobeAI Mobile App
Uses Python's built-in http.server — no extra packages needed.

Run:  python mobile_app.py
Open: http://localhost:8501
"""

import os, sys, io, json, base64, time, threading, webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.mobile.face_models import FaceModelRunner
from src.mobile.body_models import BodyModelRunner
from src.mobile.trainer     import accuracy_table, body_results_table, per_keypoint_table, _star_rating
from src.mobile.visualiser  import draw_face_keypoints, draw_body_landmarks, draw_all

# ── Load models once at startup ────────────────────────────────────────────
print("\n[eWardrobeAI] Loading models …")
face_runner = FaceModelRunner()
body_runner = BodyModelRunner()
if face_runner.models_trained():
    face_runner.load()
    print("[eWardrobeAI] Face CNNs loaded ✓")
    print("[eWardrobeAI] Evaluating on validation split …")
    _startup_results = face_runner.evaluate()
    W = 66
    print(f"\n  {'═'*W}")
    print(f"  Face CNN Accuracy — Validation Split (training.csv)")
    print(f"  {'═'*W}")
    accuracy_table(_startup_results)
    best = min(_startup_results, key=lambda r: r.mae_px)
    fast = min(_startup_results, key=lambda r: r.inference_ms)
    print(f"  Best accuracy : {best.model_name}  MAE={best.mae_px:.3f}px  {_star_rating(best.mae_px)}")
    print(f"  Fastest       : {fast.model_name}  {fast.inference_ms:.2f}ms/image")
    per_keypoint_table(_startup_results)
    print(f"  {'═'*W}\n")
else:
    print("[eWardrobeAI] Face CNNs not trained — run: python -m src.mobile.trainer")
print("[eWardrobeAI] Body models ready ✓\n")


# ── Image helpers ──────────────────────────────────────────────────────────

def bytes_to_bgr(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def bgr_to_96(bgr):
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (96, 96))

def img_to_b64(img_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()

def star(mae: float) -> str:
    if mae <= 3.5: return '★★★★★'
    if mae <= 5.0: return '★★★★☆'
    if mae <= 7.0: return '★★★☆☆'
    return '★★☆☆☆'


# ── Terminal report ────────────────────────────────────────────────────────

def print_terminal_report(face_results, body_results, shape, filename):
    W = 66
    print(f"\n{'═'*W}")
    print(f"  eWardrobeAI Mobile — Photo Analysis")
    print(f"  Image : {filename}  ({shape[1]}×{shape[0]} px)")
    print(f"{'═'*W}")

    print(f"\n  {'─'*W}")
    print(f"  FACE DETECTION  ({len(face_results)} models, trained on training.csv)")
    print(f"  {'─'*W}")
    accuracy_table(face_results)
    per_keypoint_table(face_results)

    print(f"\n  {'─'*W}")
    print(f"  BODY CALIBRATION  ({len(body_results)} models)")
    print(f"  {'─'*W}")
    body_results_table(body_results)

    print(f"{'═'*W}")
    print(f"  ✅  Analysis complete\n")


# ── Analyse endpoint ───────────────────────────────────────────────────────

def analyse(image_bytes: bytes, filename: str) -> dict:
    bgr   = bytes_to_bgr(image_bytes)
    if bgr is None:
        return {'error': 'Could not decode image'}

    h, w  = bgr.shape[:2]
    img96 = bgr_to_96(bgr)

    # Face models
    face_preds = face_runner.predict(img96)
    face_acc   = face_runner.evaluate() if face_runner.models_trained() else []
    for fa, fp in zip(face_acc, face_preds):
        fa.inference_ms = fp.inference_ms
        fa.keypoints    = fp.keypoints
    display = face_acc if face_acc else face_preds

    # Body models
    body_results = body_runner.run(bgr)

    # Print to terminal
    print_terminal_report(display, body_results, bgr.shape, filename)

    # Annotated images → base64
    face_img = draw_face_keypoints(bgr, face_preds, w, h)
    body_img = draw_body_landmarks(bgr, body_results)
    all_img  = draw_all(bgr, face_preds, body_results)

    return {
        'face': [
            {
                'name':        r.model_name,
                'description': r.description,
                'mae_px':      r.mae_px,
                'rmse_px':     r.rmse_px,
                'params':      r.param_count,
                'speed_ms':    r.inference_ms,
                'stars':       star(r.mae_px) if r.mae_px else '—',
                'per_kp':      r.per_kp_mae,
            }
            for r in display
        ],
        'body': [
            {
                'name':         r.model_name,
                'description':  r.description,
                'detected':     r.detected,
                'confidence':   r.confidence,
                'shoulder_cm':  r.shoulder_cm,
                'height_cm':    r.height_cm,
                'hip_cm':       r.hip_cm,
                'body_type':    r.body_type,
                'size':         r.standard_size,
                'landmarks':    f"{r.landmarks_found}/{r.total_landmarks}",
                'speed_ms':     r.inference_ms,
            }
            for r in body_results
        ],
        'agreement':  body_runner.agreement(body_results),
        'images': {
            'face':     img_to_b64(face_img),
            'body':     img_to_b64(body_img),
            'combined': img_to_b64(all_img),
        },
        'trained': face_runner.models_trained(),
    }


# ── HTML page ──────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>eWardrobeAI Mobile</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0D0D14;--s:#16161F;--c:#1E1E2A;--b:#2A2A3C;
      --a:#7C6FFF;--a2:#FF6B9D;--a3:#4ECDC4;--t:#E8E8F5;--m:#7777AA;
      --g:#4CAF50;--r:#F44336;--y:#FFC107}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);
     color:var(--t);min-height:100vh}
header{background:rgba(13,13,20,.95);border-bottom:1px solid var(--b);
       padding:14px 20px;display:flex;align-items:center;gap:12px;
       position:sticky;top:0;z-index:100}
.logo{font-size:1.3rem;font-weight:800}
.logo .e{color:var(--a)}.logo .w{color:var(--a2)}.logo .ai{color:var(--a3);
font-size:.7em;vertical-align:super}
.badge{background:var(--a);color:#fff;font-size:.62rem;font-weight:700;
       padding:3px 10px;border-radius:20px}
.container{max-width:960px;margin:0 auto;padding:24px 16px 60px}

/* Upload */
.upload-card{background:var(--c);border:2px dashed var(--b);border-radius:14px;
             padding:32px;text-align:center;cursor:pointer;transition:border-color .2s;
             margin-bottom:20px}
.upload-card:hover{border-color:var(--a)}
.upload-card input{display:none}
.upload-icon{font-size:3rem;margin-bottom:10px}
.upload-label{font-size:.9rem;color:var(--m)}
.upload-label strong{color:var(--a)}
.camera-btn{background:var(--c);border:1px solid var(--b);border-radius:8px;
            padding:10px 20px;color:var(--t);font-size:.82rem;cursor:pointer;
            margin-top:10px;transition:border-color .2s}
.camera-btn:hover{border-color:var(--a)}

/* Spinner */
#spinner{display:none;text-align:center;padding:40px}
.spin{width:48px;height:48px;border-radius:50%;border:4px solid var(--b);
      border-top-color:var(--a);animation:spin .7s linear infinite;margin:0 auto 12px}
@keyframes spin{to{transform:rotate(360deg)}}

/* Tabs */
.tabs{display:flex;gap:8px;margin-bottom:20px;flex-wrap:wrap}
.tab{padding:8px 18px;border-radius:8px;border:1px solid var(--b);
     background:var(--c);color:var(--m);cursor:pointer;font-size:.8rem;
     font-weight:700;transition:all .2s}
.tab:hover,.tab.active{background:var(--a);border-color:var(--a);color:#fff}
.tab-panel{display:none}.tab-panel.active{display:block;animation:fi .3s ease}
@keyframes fi{from{opacity:0;transform:translateY(8px)}to{opacity:1}}

/* Model grid */
.model-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px;margin-bottom:20px}
.model-card{background:var(--c);border:1px solid var(--b);border-radius:12px;padding:18px}
.model-card.best{border-color:var(--g)}
.mc-name{font-size:.85rem;font-weight:700;margin-bottom:10px;display:flex;
          align-items:center;justify-content:space-between}
.mc-badge{font-size:.62rem;font-weight:700;padding:2px 8px;border-radius:20px}
.badge-deep{background:rgba(124,111,255,.2);color:var(--a)}
.badge-light{background:rgba(255,107,157,.2);color:var(--a2)}
.badge-mp{background:rgba(78,205,196,.2);color:var(--a3)}
.badge-fa{background:rgba(255,180,50,.2);color:#FFB432}
.best-tag{font-size:.62rem;background:rgba(76,175,80,.2);color:var(--g);
          padding:2px 8px;border-radius:20px}
.kv{display:flex;justify-content:space-between;align-items:center;
    padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:.78rem}
.kv:last-child{border-bottom:none}
.kv .k{color:var(--m)}.kv .v{font-weight:700}
.acc-bar{height:6px;background:var(--b);border-radius:3px;margin-top:8px;overflow:hidden}
.acc-fill{height:100%;background:linear-gradient(90deg,var(--a),var(--a2));border-radius:3px;transition:width .6s ease}

/* Per-keypoint table */
.kp-table{width:100%;border-collapse:collapse;font-size:.75rem;margin-top:12px}
.kp-table th{background:var(--c);padding:8px 12px;text-align:left;
              font-size:.65rem;letter-spacing:.8px;text-transform:uppercase;
              color:var(--m);border-bottom:1px solid var(--b)}
.kp-table td{padding:7px 12px;border-bottom:1px solid rgba(255,255,255,.04)}
.kp-table tr:last-child td{border-bottom:none}
.win{color:var(--g);font-weight:700}

/* Image display */
.img-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px}
@media(max-width:600px){.img-grid,.model-grid{grid-template-columns:1fr}}
.img-card{background:var(--c);border:1px solid var(--b);border-radius:12px;overflow:hidden}
.img-card img{width:100%;display:block}
.img-label{padding:10px 14px;font-size:.75rem;font-weight:700;color:var(--m)}

/* Agreement */
.agree-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}
@media(max-width:600px){.agree-grid{grid-template-columns:repeat(2,1fr)}}
.agree-card{background:var(--c);border:1px solid var(--b);border-radius:10px;
            padding:14px;text-align:center}
.agree-val{font-size:1.4rem;font-weight:800;margin-bottom:4px}
.agree-lbl{font-size:.65rem;color:var(--m);text-transform:uppercase;letter-spacing:.6px}

.section-hdr{font-size:.65rem;font-weight:700;letter-spacing:1.5px;
             text-transform:uppercase;color:var(--m);
             border-bottom:1px solid var(--b);padding-bottom:6px;margin:18px 0 12px}
.not-trained{background:rgba(255,193,7,.1);border:1px solid var(--y);
             border-radius:8px;padding:12px 16px;font-size:.8rem;color:var(--y);margin-bottom:16px}
</style>
</head>
<body>
<header>
  <div class="logo"><span class="e">e</span><span class="w">Wardrobe</span><span class="ai">AI</span></div>
  <span class="badge">WARDROBE</span>
</header>

<div class="container">
  <!-- Upload -->
  <div class="upload-card" id="upload-card" onclick="document.getElementById('file-inp').click()">
    <div class="upload-icon">📸</div>
    <div class="upload-label"><strong>Tap to upload</strong> or drag &amp; drop a photo<br>
    JPG · PNG · WEBP</div>
    <input type="file" id="file-inp" accept="image/*" onchange="handleFile(event)"/>
  </div>
  <div style="text-align:center">
    <button class="camera-btn" onclick="document.getElementById('cam-inp').click()">
      📷 Use Camera
    </button>
    <input type="file" id="cam-inp" accept="image/*" capture="user"
           style="display:none" onchange="handleFile(event)"/>
  </div>

  <!-- Spinner -->
  <div id="spinner">
    <div class="spin"></div>
    <div id="spin-text" style="color:var(--m);font-size:.85rem">Running AI models…</div>
  </div>

  <!-- Results -->
  <div id="results" style="display:none">
    <div id="not-trained-msg" class="not-trained" style="display:none">
      ⚠️ Face CNN models not trained yet. Accuracy metrics unavailable.<br>
      Run: <strong>python -m src.mobile.trainer</strong> to train them.
    </div>

    <div class="tabs">
      <div class="tab active" onclick="showTab('face',this)">👁️ Face Detection</div>
      <div class="tab" onclick="showTab('body',this)">📏 Body Calibration</div>
      <div class="tab" onclick="showTab('combined',this)">🎯 Combined</div>
    </div>

    <!-- Face tab -->
    <div class="tab-panel active" id="tab-face">
      <div class="img-grid" id="face-img-grid"></div>
      <div class="section-hdr">Model Accuracy (Validation Split — training.csv)</div>
      <div class="model-grid" id="face-model-grid"></div>
      <div class="section-hdr">Per-Keypoint MAE (pixels)</div>
      <table class="kp-table" id="kp-table">
        <thead><tr><th>Keypoint</th><th id="kp-h1">Model 1</th><th id="kp-h2">Model 2</th><th>Winner</th></tr></thead>
        <tbody id="kp-body"></tbody>
      </table>
    </div>

    <!-- Body tab -->
    <div class="tab-panel" id="tab-body">
      <div class="img-grid" id="body-img-grid"></div>
      <div class="section-hdr">Detection Results</div>
      <div class="model-grid" id="body-model-grid"></div>
      <div class="section-hdr">Model Agreement</div>
      <div class="agree-grid" id="agree-grid"></div>
    </div>

    <!-- Combined tab -->
    <div class="tab-panel" id="tab-combined">
      <div id="combined-img"></div>
      <div class="section-hdr">Legend</div>
      <div style="font-size:.78rem;color:var(--m);line-height:2">
        🟣 DeepFaceCNN — face keypoints &nbsp;|&nbsp;
        🔴 LightFaceCNN — face keypoints &nbsp;|&nbsp;
        🩵 MediaPipe Pose — body landmarks &nbsp;|&nbsp;
        🟡 Face-Anchor — body proportions
      </div>
    </div>
  </div>
</div>

<script>
function showTab(name, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}

function handleFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append('image', file);
  fd.append('filename', file.name);
  runAnalysis(fd);
}

function runAnalysis(fd) {
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('results').style.display = 'none';

  const steps = ['Running DeepFaceCNN…','Running LightFaceCNN…',
                 'Running MediaPipe Pose…','Running Face-Anchor Estimator…',
                 'Computing accuracy…'];
  let si = 0;
  const iv = setInterval(() => {
    document.getElementById('spin-text').textContent = steps[si % steps.length];
    si++;
  }, 900);

  fetch('/analyse', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      clearInterval(iv);
      document.getElementById('spinner').style.display = 'none';
      renderResults(data);
    })
    .catch(err => {
      clearInterval(iv);
      document.getElementById('spinner').style.display = 'none';
      alert('Error: ' + err.message);
    });
}

function renderResults(data) {
  document.getElementById('results').style.display = 'block';
  document.getElementById('not-trained-msg').style.display =
    data.trained ? 'none' : 'block';

  // ── Face images ──
  const faceGrid = document.getElementById('face-img-grid');
  faceGrid.innerHTML = `
    <div class="img-card">
      <img src="data:image/jpeg;base64,${data.images.face}" alt="Face keypoints"/>
      <div class="img-label">🟣 DeepFaceCNN + 🔴 LightFaceCNN — Keypoints</div>
    </div>
    <div class="img-card">
      <img src="data:image/jpeg;base64,${data.images.body}" alt="Body landmarks"/>
      <div class="img-label">🩵 MediaPipe + 🟡 Face-Anchor — Body</div>
    </div>`;

  // ── Face model cards ──
  const best = data.face.length ? Math.min(...data.face.map(m => m.mae_px || 999)) : 0;
  const faceGrid2 = document.getElementById('face-model-grid');
  const badgeClass = ['badge-deep','badge-light'];
  faceGrid2.innerHTML = data.face.map((m, i) => {
    const isBest = m.mae_px && m.mae_px === best;
    const pctFill = m.mae_px ? Math.max(0, 100 - m.mae_px * 8) : 0;
    return `
      <div class="model-card ${isBest ? 'best' : ''}">
        <div class="mc-name">
          <span><span class="mc-badge ${badgeClass[i]}">${m.name}</span></span>
          ${isBest ? '<span class="best-tag">🏆 Best</span>' : ''}
        </div>
        <div class="kv"><span class="k">MAE (pixels)</span>
          <span class="v" style="color:${i===0?'#7C6FFF':'#FF6B9D'}">${m.mae_px || '—'} ${m.mae_px ? m.stars : ''}</span></div>
        <div class="kv"><span class="k">RMSE (pixels)</span>
          <span class="v">${m.rmse_px || '—'}</span></div>
        <div class="kv"><span class="k">Parameters</span>
          <span class="v">${m.params ? m.params.toLocaleString() : '—'}</span></div>
        <div class="kv"><span class="k">Description</span>
          <span class="v" style="font-size:.68rem">${m.description}</span></div>
        <div class="kv"><span class="k">Inference speed</span>
          <span class="v">${m.speed_ms} ms</span></div>
        ${m.mae_px ? `<div class="acc-bar"><div class="acc-fill" style="width:${pctFill}%"></div></div>` : ''}
      </div>`;
  }).join('');

  // Set column headers
  if (data.face.length >= 2) {
    document.getElementById('kp-h1').textContent = data.face[0].name;
    document.getElementById('kp-h2').textContent = data.face[1].name;
  }

  // ── Per-keypoint table ──
  const kpBody = document.getElementById('kp-body');
  if (data.face[0]?.per_kp && Object.keys(data.face[0].per_kp).length) {
    kpBody.innerHTML = Object.keys(data.face[0].per_kp).map(kp => {
      const v1 = data.face[0].per_kp[kp] || 0;
      const v2 = data.face[1]?.per_kp?.[kp] || 0;
      const win = v1 < v2 ? '◀ M1' : (v2 < v1 ? '▶ M2' : '=');
      const wc  = v1 < v2 ? 'win' : (v2 < v1 ? 'win' : '');
      return `<tr>
        <td>${kp.replace(/_/g,' ')}</td>
        <td class="${v1<v2?'win':''}">${v1.toFixed(3)}</td>
        <td class="${v2<v1?'win':''}">${v2.toFixed(3)}</td>
        <td class="${wc}">${win}</td>
      </tr>`;
    }).join('');
  } else {
    kpBody.innerHTML = `<tr><td colspan="4" style="color:var(--m);padding:12px">
      Train models first for per-keypoint breakdown.</td></tr>`;
  }

  // ── Body model cards ──
  const bodyBadge = ['badge-mp','badge-fa'];
  const bodyColour = ['#4ECDC4','#FFB432'];
  document.getElementById('body-model-grid').innerHTML = data.body.map((m, i) => `
    <div class="model-card">
      <div class="mc-name">
        <span class="mc-badge ${bodyBadge[i]}">${m.name}</span>
      </div>
      <div class="kv"><span class="k">Detected</span>
        <span class="v" style="color:${m.detected?'var(--g)':'var(--r)'}">${m.detected?'✓ Yes':'✗ No'}</span></div>
      <div class="kv"><span class="k">Confidence</span>
        <span class="v" style="color:${bodyColour[i]}">${(m.confidence*100).toFixed(1)}%</span></div>
      <div class="kv"><span class="k">Landmarks</span>
        <span class="v">${m.landmarks}</span></div>
      ${m.detected ? `
      <div class="kv"><span class="k">Shoulder width</span>
        <span class="v">${m.shoulder_cm} cm</span></div>
      <div class="kv"><span class="k">Height estimate</span>
        <span class="v">${m.height_cm} cm</span></div>
      <div class="kv"><span class="k">Body type</span>
        <span class="v">${m.body_type}</span></div>
      <div class="kv"><span class="k">Standard size</span>
        <span class="v" style="color:var(--a)">${m.size}</span></div>` : ''}
      <div class="kv"><span class="k">Speed</span>
        <span class="v">${m.speed_ms} ms</span></div>
      <div class="acc-bar"><div class="acc-fill" style="width:${m.confidence*100}%"></div></div>
    </div>`).join('');

  // ── Agreement ──
  const ag = data.agreement;
  document.getElementById('agree-grid').innerHTML = [
    ['Agreement', ag.agreement_pct + '%'],
    ['Shoulder diff', (ag.shoulder_diff_cm ?? '—') + ' cm'],
    ['Height diff', (ag.height_diff_cm ?? '—') + ' cm'],
    ['Size match', ag.size_match ? '✓ Yes' : '✗ No'],
  ].map(([l, v]) => `
    <div class="agree-card">
      <div class="agree-val">${v}</div>
      <div class="agree-lbl">${l}</div>
    </div>`).join('');

  // ── Combined image ──
  document.getElementById('combined-img').innerHTML = `
    <div class="img-card" style="max-width:600px;margin:0 auto">
      <img src="data:image/jpeg;base64,${data.images.combined}" alt="Combined"/>
    </div>`;

  // Scroll to results
  document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

// Drag-and-drop
const card = document.getElementById('upload-card');
card.addEventListener('dragover', e => { e.preventDefault(); card.style.borderColor='var(--a)'; });
card.addEventListener('dragleave', () => { card.style.borderColor=''; });
card.addEventListener('drop', e => {
  e.preventDefault(); card.style.borderColor='';
  if (e.dataTransfer.files[0]) {
    const fd = new FormData();
    fd.append('image', e.dataTransfer.files[0]);
    fd.append('filename', e.dataTransfer.files[0].name);
    runAnalysis(fd);
  }
});
</script>
</body>
</html>"""


# ── HTTP Handler ───────────────────────────────────────────────────────────

def parse_multipart(data: bytes, boundary: str):
    """Extract form fields from raw multipart body."""
    parts = {}
    sep   = ('--' + boundary).encode()
    for chunk in data.split(sep):
        if b'Content-Disposition' not in chunk:
            continue
        header, _, body = chunk.partition(b'\r\n\r\n')
        body = body.rstrip(b'\r\n--')
        h    = header.decode(errors='replace')
        name = ''
        for part in h.split(';'):
            part = part.strip()
            if part.startswith('name='):
                name = part.split('=',1)[1].strip('"')
        if name:
            parts[name] = body
    return parts


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass   # suppress default access logs

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(HTML.encode())

    def do_POST(self):
        if self.path != '/analyse':
            self.send_response(404); self.end_headers(); return

        ct      = self.headers.get('Content-Type', '')
        length  = int(self.headers.get('Content-Length', 0))
        body    = self.rfile.read(length)

        # Parse multipart
        boundary = ''
        for part in ct.split(';'):
            part = part.strip()
            if part.startswith('boundary='):
                boundary = part.split('=', 1)[1].strip()
        fields = parse_multipart(body, boundary)

        image_bytes = fields.get('image', b'')
        filename    = fields.get('filename', b'photo.jpg').decode(errors='replace')

        if not image_bytes:
            self._json({'error': 'No image received'}, 400); return

        try:
            result = analyse(image_bytes, filename)
        except Exception as e:
            import traceback; traceback.print_exc()
            result = {'error': str(e)}

        self._json(result)

    def _json(self, data: dict, code: int = 200):
        payload = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type',   'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


# ── Entry Point ────────────────────────────────────────────────────────────

PORT = 8501

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    url    = f'http://localhost:{PORT}'
    print(f"[eWardrobeAI] Server running at {url}")
    print(f"[eWardrobeAI] Open on mobile : http://<your-ip>:{PORT}")
    print(f"[eWardrobeAI] Press Ctrl+C to stop\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[eWardrobeAI] Stopped.")
