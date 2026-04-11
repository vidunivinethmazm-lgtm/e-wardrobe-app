/**
 * eWardrobeAI — Selfie Capture Module
 * Handles: live camera capture (getUserMedia) + file upload with validation
 */

'use strict';

let _stream     = null;
let _captureReady = false;

/* ── Camera ─────────────────────────────────────────────────────────────── */
async function startCamera() {
  const video   = document.getElementById('camera-video');
  const overlay = document.getElementById('camera-overlay');
  const ph      = document.getElementById('cam-placeholder');

  try {
    _stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    });
    video.srcObject = _stream;
    video.style.display    = 'block';
    overlay.style.display  = 'flex';
    ph.style.display       = 'none';
  } catch (err) {
    alert('Camera not available: ' + err.message + '\nPlease upload a photo instead.');
  }
}

function stopCamera() {
  const video   = document.getElementById('camera-video');
  const overlay = document.getElementById('camera-overlay');
  const ph      = document.getElementById('cam-placeholder');

  if (_stream) {
    _stream.getTracks().forEach(t => t.stop());
    _stream = null;
  }
  video.srcObject        = null;
  video.style.display    = 'none';
  overlay.style.display  = 'none';
  ph.style.display       = 'flex';
}

function capturePhoto() {
  const video  = document.getElementById('camera-video');
  const canvas = document.getElementById('selfie-canvas');
  const preview= document.getElementById('capture-preview');

  if (!_stream || !video.videoWidth) {
    alert('Camera not ready. Please wait a moment.');
    return;
  }

  // Draw video frame to canvas
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);

  const dataUrl = canvas.toDataURL('image/jpeg', 0.92);

  // Show preview on camera view
  preview.src          = dataUrl;
  preview.style.display= 'block';
  video.style.display  = 'none';
  document.getElementById('camera-overlay').style.display = 'none';

  // Convert to File
  canvas.toBlob(blob => {
    const file = new File([blob], 'selfie.jpg', { type: 'image/jpeg' });
    stopCamera();
    onSelfieReady(file, dataUrl);   // → app.js
  }, 'image/jpeg', 0.92);
}

/* ── File Upload ────────────────────────────────────────────────────────── */
const selfieInput = document.getElementById('selfie-file');
const uploadZone  = document.getElementById('upload-zone');

selfieInput.addEventListener('change', e => {
  if (e.target.files[0]) handleUploadedFile(e.target.files[0]);
});

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('over');
  if (e.dataTransfer.files[0]) handleUploadedFile(e.dataTransfer.files[0]);
});

function handleUploadedFile(file) {
  // Validate
  const ALLOWED = ['image/jpeg', 'image/png', 'image/webp'];
  if (!ALLOWED.includes(file.type)) {
    alert('Please upload a JPG, PNG, or WEBP image.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    alert('Image is too large. Please use an image under 10 MB.');
    return;
  }

  const reader = new FileReader();
  reader.onload = e => {
    // Basic face-presence check: ensure image loads correctly
    const img = new Image();
    img.onload = () => {
      if (img.width < 50 || img.height < 50) {
        alert('Image is too small. Please use a larger photo.');
        return;
      }
      onSelfieReady(file, e.target.result);   // → app.js
    };
    img.onerror = () => alert('Could not read image. Please try another file.');
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}
