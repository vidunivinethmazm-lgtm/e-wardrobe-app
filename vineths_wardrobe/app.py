"""
eWardrobeAI — FastAPI Backend Server
Swagger UI available at: http://localhost:8000/docs
ReDoc   UI available at: http://localhost:8000/redoc
"""

import io
import json
import base64
import logging
import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Annotated

from fastapi import (
    FastAPI, File, Form, UploadFile, HTTPException,
    WebSocket, WebSocketDisconnect, Body, Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles     import StaticFiles
from fastapi.responses       import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.openapi.docs    import get_redoc_html, get_swagger_ui_html
from pydantic                import BaseModel, Field

import cv2
import numpy as np

from src.virtual_tryon_pipeline import VirtualTryOnPipeline, TryOnRequest
from src.outfit_recommender     import CleaningStatus, Style

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)

# ── Application Lifespan ──────────────────────────────────────────────────────

pipeline: Optional[VirtualTryOnPipeline] = None
_sse_sessions: dict[str, asyncio.Queue] = {}
_model_registry = None   # lazy-loaded on first /api/accuracy call


def _print_accuracy_report():
    import warnings, threading
    warnings.filterwarnings('ignore')

    def bar(v, w=20): return '#' * int(v * w) + '-' * (w - int(v * w))

    def run():
        try:
            from src.model_registry import ModelRegistry
            reg = ModelRegistry()
            W = 70

            def sep(c='='): print(c * W, flush=True)
            def hdr(t):     sep(); print(f'  {t}'); sep()

            print('\n'); hdr('eWardrobeAI  --  STARTUP ACCURACY REPORT')

            # Stage 1
            r1 = reg.evaluate_stage1(4000)
            print('\n  STAGE 1 : Body Calibration  (4,000 samples)')
            sep('-')
            print(f"  {'Model':<35} {'Body Type %':>11} {'Bar':<22} {'Size %':>7} {'F1 %':>6}")
            sep('-')
            for m in r1['models']:
                bt   = m['bodyType']['accuracy']
                sz   = m['standardSize']['accuracy']
                f1   = m['bodyType']['f1_weighted']
                best = '  <-- BEST' if m['model'] == r1['bestBodyType'] else ''
                print(f"  {m['model']:<35} {bt*100:>10.1f}%  [{bar(bt)}]  {sz*100:>6.1f}%  {f1*100:>5.1f}%{best}")
            sep('-')

            # Stage 3
            r3 = reg.evaluate_stage3()
            print('\n  STAGE 3 : Outfit Recommendation  (6 scenarios, top-5)')
            sep('-')
            print(f"  {'Model':<35} {'Precision %':>11} {'Bar':<22} {'Cover %':>8} {'Score %':>8}")
            sep('-')
            for m in r3['models']:
                p    = m['precision_at_k']
                best = '  <-- BEST' if 'Heuristic' in m['model'] else ''
                print(f"  {m['model']:<35} {p*100:>10.1f}%  [{bar(p)}]  {m['coverage']*100:>7.1f}%  {m['avg_score']*100:>7.1f}%{best}")
            sep('-')

            # Stage 4
            r4 = reg.evaluate_stage4()
            print('\n  STAGE 4 : Avatar Scale Regression  (1,000 samples)')
            sep('-')
            print(f"  {'Model':<35} {'Fit %':>7} {'Bar':<22} {'RMSE %':>7} {'MAE %':>7}")
            sep('-')
            for m in r4['models']:
                fit  = max(0.0, 1.0 - m['rmse'])
                best = '  <-- BEST' if m['model'] == r4['bestRMSE'] else ''
                print(f"  {m['model']:<35} {fit*100:>6.1f}%  [{bar(fit)}]  {m['rmse']*100:>6.3f}%  {m['mae']*100:>6.3f}%{best}")
            sep('-')

            print('\n  STAGE 2 : Face CNN  -- train first: python scripts/train_all_models.py')

            print()
            hdr('BEST MODEL PER STAGE')
            print(f"  Stage 1  Body Type   :  {r1['bestBodyType']:<30} {r1['summary'][r1['bestBodyType']]['bodyTypeAccuracy']*100:.1f}%")
            print(f"  Stage 1  Size        :  {r1['bestSize']:<30} {r1['summary'][r1['bestSize']]['sizeAccuracy']*100:.1f}%")
            s3key = r3['bestPrecision'] + ' Recommender'
            print(f"  Stage 3  Recommender :  {s3key:<30} {r3['summary'][s3key]['precision_at_k']*100:.1f}%")
            print(f"  Stage 4  Scaler      :  {r4['bestRMSE']:<30} {(1 - r4['summary'][r4['bestRMSE']]['rmse'])*100:.1f}%")
            print(f"  Stage 4  Scaler      :  {r4['bestRMSE']}")
            sep()
            print()

        except Exception as e:
            print(f'\n  [Accuracy Report] Could not run: {e}\n')

    threading.Thread(target=run, daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("[App] Starting eWardrobeAI server …")
    pipeline = VirtualTryOnPipeline(use_mediapipe=True, use_cnn=True)
    _print_accuracy_report()
    yield
    logger.info("[App] Shutting down …")
    if pipeline:
        pipeline.teardown()


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "eWardrobeAI — Virtual Try-On API",
    description = """
## eWardrobeAI Smart Wardrobe System

AI-powered virtual try-on pipeline with 4 stages:

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | **Body Calibration** | Validates measurements, computes avatar scale |
| 2 | **Face Processing** | MediaPipe 468 landmarks + CNN 15 keypoints |
| 3 | **Outfit Recommendation** | NisfaMatchmaking + RaveehaOrganisationalDB |
| 4 | **3D Rendering** | Three.js avatar with Mixamo animations |

### Quick Test (no selfie needed)
Use **`POST /api/demo/tryon`** — runs the full pipeline with a synthetic face image.

### Garment IDs for status testing
`GAR-001` White Oxford Shirt · `GAR-002` Navy Polo · `GAR-005` Slim Chinos ·
`GAR-006` Dark Jeans · `GAR-007` Dress Trousers · `GAR-008` Wool Blazer ·
`GAR-009` Puffer Jacket · `GAR-010` Wrap Midi Dress · `GAR-012` Classic Suit
    """,
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = None,
    redoc_url   = None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), 'frontend')
if os.path.isdir(_FRONTEND_DIR):
    app.mount('/static', StaticFiles(directory=_FRONTEND_DIR), name='static')


@app.get('/docs', include_in_schema=False)
async def custom_swagger():
    return get_swagger_ui_html(
        openapi_url='/openapi.json',
        title=app.title + ' - Swagger UI',
        swagger_js_url='/static/static_docs/swagger-ui-bundle.js',
        swagger_css_url='/static/static_docs/swagger-ui.css',
    )


@app.get('/redoc', include_in_schema=False)
async def custom_redoc():
    return get_redoc_html(
        openapi_url='/openapi.json',
        title=app.title + ' - ReDoc',
        redoc_js_url='/static/static_docs/redoc.standalone.js',
    )


# ── Pydantic Request / Response Models ───────────────────────────────────────

class StatusUpdateBody(BaseModel):
    status: str = Field(
        ...,
        description="Garment cleaning status",
        examples=["Clean"],
        json_schema_extra={"enum": ["Clean", "Dirty", "In Laundry"]},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"status": "Dirty"}]
        }
    }


class TrainTriggerBody(BaseModel):
    epochs:     int = Field(100, ge=1,  le=500, description="Training epochs",     examples=[50])
    batch_size: int = Field(64,  ge=8,  le=256, description="Mini-batch size",     examples=[64])

    model_config = {
        "json_schema_extra": {
            "examples": [{"epochs": 50, "batch_size": 64}]
        }
    }


class DemoTryOnBody(BaseModel):
    shoulder_width_cm: float       = Field(42.0,  ge=30,  le=65,  description="Shoulder width (cm)",  examples=[42.0])
    chest_cm:          float       = Field(92.0,  ge=60,  le=160, description="Chest circumference",  examples=[92.0])
    waist_cm:          float       = Field(72.0,  ge=50,  le=150, description="Waist circumference",  examples=[72.0])
    height_cm:         float       = Field(168.0, ge=120, le=230, description="Standing height (cm)", examples=[168.0])
    hip_cm:            Optional[float] = Field(None, description="Hip circumference (auto if blank)", examples=[97.0])
    inseam_cm:         Optional[float] = Field(None, description="Inseam length (auto if blank)",     examples=[79.0])
    weight_kg:         Optional[float] = Field(None, description="Body weight kg (optional)",         examples=[68.0])
    styles:            str         = Field("smart_casual,casual", description="Comma-separated style preferences",
                                           examples=["smart_casual,casual"])
    occasion:          str         = Field("casual",
                                           description="Target occasion",
                                           examples=["casual"])
    animation_key:     str         = Field("idle",
                                           description="Mixamo animation: idle | walk | rotate | pose_t | pose_a | catwalk",
                                           examples=["walk"])
    top_k:             int         = Field(3, ge=1, le=10, description="Max outfit recommendations", examples=[3])

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "shoulder_width_cm": 42.0,
                "chest_cm":          92.0,
                "waist_cm":          72.0,
                "height_cm":        168.0,
                "hip_cm":            97.0,
                "inseam_cm":         79.0,
                "weight_kg":         68.0,
                "styles":           "smart_casual,casual",
                "occasion":         "office",
                "animation_key":    "walk",
                "top_k":             3,
            }]
        }
    }


# ── Helper: build pipeline result dict ───────────────────────────────────────

def _format_result(result) -> dict:
    return {
        'success':          True,
        'processingTimeMs': round(result.processing_time_ms, 2),
        'warnings':         result.warnings,
        'sizingProfile': {
            'standardSize':  result.sizing_profile.standard_size,
            'bodyType':      result.sizing_profile.body_type,
            'waistHipRatio': result.sizing_profile.waist_hip_ratio,
            'torsoLengthCm': result.sizing_profile.torso_length_cm,
        } if result.sizing_profile else None,
        'faceAnalysis': {
            'interEyeDistPx':   result.face_profile.inter_eye_dist,
            'landmark15Count':  len(result.face_profile.landmarks_15),
            'landmark468Count': len(result.face_profile.landmarks_468),
            'yawDeg':           result.face_profile.yaw_deg,
            'pitchDeg':         result.face_profile.pitch_deg,
            'hasFaceTexture':   result.face_profile.face_texture is not None,
        } if result.face_profile else None,
        'recommendations': [
            {
                'outfitId': rec.outfit_id,
                'name':     rec.name,
                'style':    rec.style.value,
                'score':    rec.score,
                'occasion': rec.occasion,
                'items': [
                    {
                        'garmentId': item.garment_id,
                        'name':      item.name,
                        'category':  item.category.value,
                        'assetPath': item.asset_path,
                        'status':    item.cleaning_status.value,
                    }
                    for item in rec.items
                ],
            }
            for rec in result.recommendations
        ],
        'renderPayload':    result.render_payload.to_dict() if result.render_payload else None,
        'alternateCount':   len(result.alternate_payloads),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────


# ── SSE: Real-Time Pipeline Stage Streaming ───────────────────────────────────

def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post(
    '/api/tryon/stream/start',
    tags=['Try-On'],
    summary="Start streaming try-on session",
    description="""
Creates a streaming session. Returns a `sessionId`.
Then open `GET /api/tryon/stream/{sessionId}` (SSE) to receive real-time stage updates.
Finally POST measurements to `POST /api/tryon/stream/{sessionId}/run`.
    """,
)
async def stream_start():
    sid = uuid.uuid4().hex
    _sse_sessions[sid] = asyncio.Queue()
    return {'sessionId': sid, 'streamUrl': f'/api/tryon/stream/{sid}'}


@app.get(
    '/api/tryon/stream/{session_id}',
    tags=['Try-On'],
    summary="SSE stream for real-time stage updates",
    description="Server-Sent Events stream. Connect before calling /run. Each stage emits a JSON event.",
)
async def stream_events(session_id: str, request: Request):
    if session_id not in _sse_sessions:
        raise HTTPException(404, f"Session '{session_id}' not found.")

    queue = _sse_sessions[session_id]

    async def event_generator():
        yield _sse_event('connected', {'sessionId': session_id, 'message': 'Stream ready'})
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield _sse_event(event['type'], event['data'])
                    if event['type'] == 'complete' or event['type'] == 'error':
                        break
                except asyncio.TimeoutError:
                    yield _sse_event('ping', {'ts': asyncio.get_event_loop().time()})
        finally:
            _sse_sessions.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )


@app.post(
    '/api/tryon/stream/{session_id}/run',
    tags=['Try-On'],
    summary="Run pipeline with SSE stage updates",
    description="Submit selfie + measurements. Streams stage events to the SSE connection.",
)
async def stream_run(
    session_id:        str,
    selfie:            UploadFile                         = File(...),
    shoulder_width_cm: Annotated[float, Form()]           = 42.0,
    chest_cm:          Annotated[float, Form()]           = 92.0,
    waist_cm:          Annotated[float, Form()]           = 72.0,
    height_cm:         Annotated[float, Form()]           = 168.0,
    hip_cm:            Annotated[Optional[float], Form()] = None,
    inseam_cm:         Annotated[Optional[float], Form()] = None,
    weight_kg:         Annotated[Optional[float], Form()] = None,
    styles:            Annotated[str, Form()]             = 'smart_casual,casual',
    occasion:          Annotated[str, Form()]             = 'casual',
    animation_key:     Annotated[str, Form()]             = 'idle',
    top_k:             Annotated[int, Form()]             = 5,
):
    if session_id not in _sse_sessions:
        raise HTTPException(404, f"Session '{session_id}' not found. Call /start first.")
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialised.")

    queue        = _sse_sessions[session_id]
    image_bytes  = await selfie.read()
    preferred_styles = [s.strip() for s in styles.split(',') if s.strip()]

    async def run_pipeline():
        try:
            # Stage 1
            await queue.put({'type':'stage', 'data': {'stage': 1, 'name': 'Body Calibration', 'status': 'running'}})
            from src.body_calibration import BodyCalibrator, BodyMeasurements
            cal = BodyCalibrator()
            val, scale, sizing = cal.calibrate(shoulder_width_cm, chest_cm, waist_cm, height_cm, hip_cm, inseam_cm, weight_kg)
            if not val.is_valid:
                await queue.put({'type':'error', 'data': {'message': '; '.join(val.errors)}})
                return
            await queue.put({'type':'stage', 'data': {
                'stage': 1, 'status': 'done',
                'result': {'size': sizing.standard_size, 'bodyType': sizing.body_type}
            }})
            await asyncio.sleep(0.05)

            # Stage 2
            await queue.put({'type':'stage', 'data': {'stage': 2, 'name': 'Face Processing', 'status': 'running'}})
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            face_profile = pipeline._face_proc.process(bgr) if bgr is not None else None
            from src.face_processor import FaceProfile
            if face_profile is None: face_profile = FaceProfile()
            await queue.put({'type':'stage', 'data': {
                'stage': 2, 'status': 'done',
                'result': {
                    'landmarks15':  len(face_profile.landmarks_15),
                    'landmarks468': len(face_profile.landmarks_468),
                    'interEyeDist': face_profile.inter_eye_dist,
                    'hasFaceTexture': face_profile.face_texture is not None,
                }
            }})
            await asyncio.sleep(0.05)

            # Stage 3
            await queue.put({'type':'stage', 'data': {'stage': 3, 'name': 'Outfit Matching', 'status': 'running'}})
            from src.outfit_recommender import Style
            style_map = {'casual':Style.CASUAL,'formal':Style.FORMAL,'smart_casual':Style.SMART,
                         'smart':Style.SMART,'sporty':Style.SPORTY,'evening':Style.EVENING}
            styles_enum = [style_map[s] for s in preferred_styles if s in style_map] or [Style.CASUAL]
            recommendations = pipeline._matchmaker.recommend(
                sizing.standard_size, sizing.body_type, styles_enum, occasion, top_k)
            await queue.put({'type':'stage', 'data': {
                'stage': 3, 'status': 'done',
                'result': {'outfitCount': len(recommendations), 'topScore': recommendations[0].score if recommendations else 0}
            }})
            await asyncio.sleep(0.05)

            # Stage 4
            await queue.put({'type':'stage', 'data': {'stage': 4, 'name': 'Avatar Generation', 'status': 'running'}})
            if face_profile.inter_eye_dist > 0:
                from src.body_calibration import BodyMeasurements
                scale = BodyCalibrator.compute_avatar_scale(
                    BodyMeasurements(shoulder_width_cm, chest_cm, waist_cm, height_cm, hip_cm, inseam_cm),
                    face_profile.inter_eye_dist)
            primary = pipeline._avatar_mgr.build_render_payload(scale, face_profile, recommendations[0], animation_key) if recommendations else None
            alts    = [pipeline._avatar_mgr.build_render_payload(scale, face_profile, r, 'idle') for r in recommendations[1:]]
            await queue.put({'type':'stage', 'data': {
                'stage': 4, 'status': 'done',
                'result': {'clothingMeshes': len(primary.clothing_assets) if primary else 0}
            }})

            # Complete — send full result
            await queue.put({'type':'complete', 'data': {
                'success': True,
                'sizingProfile': {'standardSize': sizing.standard_size, 'bodyType': sizing.body_type,
                                  'waistHipRatio': sizing.waist_hip_ratio, 'torsoLengthCm': sizing.torso_length_cm},
                'faceAnalysis': {'landmark15Count': len(face_profile.landmarks_15),
                                 'landmark468Count': len(face_profile.landmarks_468),
                                 'interEyeDistPx': face_profile.inter_eye_dist,
                                 'yawDeg': face_profile.yaw_deg, 'pitchDeg': face_profile.pitch_deg,
                                 'hasFaceTexture': face_profile.face_texture is not None},
                'recommendations': [
                    {'outfitId': r.outfit_id, 'name': r.name, 'style': r.style.value,
                     'score': r.score, 'occasion': r.occasion,
                     'items': [{'garmentId': i.garment_id, 'name': i.name,
                                'category': i.category.value, 'assetPath': i.asset_path,
                                'status': i.cleaning_status.value} for i in r.items]}
                    for r in recommendations
                ],
                'renderPayload':    primary.to_dict() if primary else None,
                'alternatePayloads': [p.to_dict() for p in alts],
            }})

        except Exception as e:
            logger.error(f"[SSE] Pipeline error: {e}", exc_info=True)
            await queue.put({'type':'error', 'data': {'message': str(e)}})

    asyncio.create_task(run_pipeline())
    return {'message': f'Pipeline started. Listen on /api/tryon/stream/{session_id}'}


@app.get('/accuracy', response_class=HTMLResponse, tags=['UI'],
         summary="Model Accuracy Dashboard",
         description="Interactive accuracy comparison dashboard for all 4 pipeline stages.")
async def serve_accuracy_dashboard():
    html_path = os.path.join(_FRONTEND_DIR, 'accuracy.html')
    if os.path.isfile(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h2>Accuracy dashboard not found.</h2>")


@app.get('/', response_class=HTMLResponse, tags=['UI'],
         summary="Frontend UI",
         description="Serves the Three.js virtual try-on web interface.")
async def serve_frontend():
    html_path = os.path.join(_FRONTEND_DIR, 'index.html')
    if os.path.isfile(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(f.read())
    return HTMLResponse(
        "<h2 style='font-family:sans-serif;padding:40px'>eWardrobeAI API is running.<br>"
        "<a href='/docs'>→ Open Swagger UI</a></h2>"
    )


@app.get('/api/health', tags=['System'],
         summary="Health check",
         description="Confirms server and pipeline are running.")
async def health_check():
    """Returns server status and pipeline readiness."""
    return {
        'status':          'ok',
        'pipelineReady':   pipeline is not None,
        'device':          str(__import__('torch').device('cuda' if __import__('torch').cuda.is_available() else 'cpu')),
        'cnnModelLoaded':  pipeline._face_proc._use_cnn if pipeline else False,
        'mediapipeActive': pipeline._face_proc._use_mp  if pipeline else False,
    }


# ── Demo Try-On (Swagger-friendly, no file upload) ────────────────────────────

@app.post(
    '/api/demo/tryon',
    tags=['Try-On'],
    summary="Demo try-on (no selfie required)",
    description="""
Runs the **full 4-stage pipeline** using a synthetic 96×96 grayscale face image.
Ideal for testing all pipeline stages directly from Swagger without uploading a selfie.

**Stages exercised:**
- ✅ Stage 1 — Body Calibration (your measurements)
- ✅ Stage 2 — Face Processing (synthetic face, CNN + fallback)
- ✅ Stage 3 — Outfit Recommendation (NisfaMatchmaking + RaveehaDB)
- ✅ Stage 4 — Avatar Render Payload assembly
    """,
)
async def demo_tryon(body: DemoTryOnBody):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialised.")

    # Generate a synthetic 96×96 grayscale face image (oval + eyes + nose)
    synthetic = _make_synthetic_face()
    _, buf = cv2.imencode('.jpg', synthetic)
    image_bytes = buf.tobytes()

    preferred_styles = [s.strip() for s in body.styles.split(',') if s.strip()]

    request = TryOnRequest(
        image_bytes       = image_bytes,
        shoulder_width_cm = body.shoulder_width_cm,
        chest_cm          = body.chest_cm,
        waist_cm          = body.waist_cm,
        height_cm         = body.height_cm,
        hip_cm            = body.hip_cm,
        inseam_cm         = body.inseam_cm,
        weight_kg         = body.weight_kg,
        preferred_styles  = preferred_styles,
        occasion          = body.occasion,
        animation_key     = body.animation_key,
        top_k             = body.top_k,
    )

    result = pipeline.process(request)

    if not result.success:
        raise HTTPException(422, detail={
            'error':    result.error_message,
            'warnings': result.warnings,
        })

    return _format_result(result)


def _make_synthetic_face() -> np.ndarray:
    """Generate a 96×96 grayscale image resembling a face for demo/test use."""
    img = np.zeros((96, 96), dtype=np.uint8)
    img[:] = 200                                          # skin-tone background
    cv2.ellipse(img, (48, 50), (38, 46), 0, 0, 360, 240, -1)   # face oval
    cv2.circle(img, (33, 38), 7, 160, -1)                # left eye socket
    cv2.circle(img, (63, 38), 7, 160, -1)                # right eye socket
    cv2.circle(img, (33, 38), 4, 60,  -1)                # left iris
    cv2.circle(img, (63, 38), 4, 60,  -1)                # right iris
    cv2.ellipse(img, (48, 62), (8, 5),  0, 0, 360, 130, -1)    # nose
    cv2.ellipse(img, (48, 76), (14, 6), 0, 0, 180, 80,  2)     # mouth
    return img


# ── Real Try-On (multipart, requires selfie) ──────────────────────────────────

@app.post(
    '/api/tryon',
    tags=['Try-On'],
    summary="Full try-on with selfie upload",
    description="""
Upload a real **selfie image** (JPEG/PNG) together with body measurements.
Returns the complete AvatarRenderPayload for the Three.js renderer.

Use **`/api/demo/tryon`** for quick Swagger testing without a file.
    """,
)
async def virtual_tryon(
    selfie:            UploadFile                       = File(...,   description="User selfie — JPEG or PNG"),
    shoulder_width_cm: Annotated[float, Form()]         = 42.0,
    chest_cm:          Annotated[float, Form()]         = 92.0,
    waist_cm:          Annotated[float, Form()]         = 72.0,
    height_cm:         Annotated[float, Form()]         = 168.0,
    hip_cm:            Annotated[Optional[float], Form()] = None,
    inseam_cm:         Annotated[Optional[float], Form()] = None,
    weight_kg:         Annotated[Optional[float], Form()] = None,
    styles:            Annotated[str,   Form()]         = 'smart_casual,casual',
    occasion:          Annotated[str,   Form()]         = 'casual',
    animation_key:     Annotated[str,   Form()]         = 'idle',
    top_k:             Annotated[int,   Form()]         = 3,
):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialised.")

    image_bytes      = await selfie.read()
    preferred_styles = [s.strip() for s in styles.split(',') if s.strip()]

    # ── Print model accuracy comparison to terminal ───────────────────────────
    try:
        from src.terminal_evaluator import run_on_upload
        import concurrent.futures
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=1),
            lambda: run_on_upload(
                image_bytes, shoulder_width_cm, chest_cm,
                waist_cm, height_cm, hip_cm, inseam_cm,
            )
        )
    except Exception as e:
        logger.warning(f"[Evaluator] Terminal evaluation error: {e}")

    request = TryOnRequest(
        image_bytes=image_bytes, shoulder_width_cm=shoulder_width_cm,
        chest_cm=chest_cm, waist_cm=waist_cm, height_cm=height_cm,
        hip_cm=hip_cm, inseam_cm=inseam_cm, weight_kg=weight_kg,
        preferred_styles=preferred_styles, occasion=occasion,
        animation_key=animation_key, top_k=top_k,
    )

    result = pipeline.process(request)
    if not result.success:
        raise HTTPException(422, detail={'error': result.error_message, 'warnings': result.warnings})

    return _format_result(result)


# ── Wardrobe Endpoints ────────────────────────────────────────────────────────

@app.get(
    '/api/wardrobe/summary',
    tags=['Wardrobe'],
    summary="Wardrobe cleaning status counts",
    description="Returns number of garments in each cleaning state: Clean, Dirty, In Laundry.",
)
async def wardrobe_summary():
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialised.")
    return pipeline.get_wardrobe_summary()


@app.get(
    '/api/wardrobe/items',
    tags=['Wardrobe'],
    summary="List all wardrobe items",
    description="Returns every garment with its ID, name, category, and current cleaning status.",
)
async def list_wardrobe_items():
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialised.")
    items = []
    for gid, rec in pipeline._wardrobe_db._records.items():
        items.append({
            'garmentId':      rec.garment_id,
            'name':           rec.name,
            'category':       rec.category.value,
            'style':          rec.style.value,
            'sizes':          rec.sizes,
            'colours':        rec.colours,
            'cleaningStatus': rec.cleaning_status.value,
            'availability':   rec.availability.value,
            'isWearable':     rec.is_wearable,
            'assetPath':      rec.asset_path,
            'tags':           rec.tags,
        })
    return {'total': len(items), 'items': items}


@app.patch(
    '/api/wardrobe/{garment_id}/status',
    tags=['Wardrobe'],
    summary="Update garment cleaning status",
    description="""
Change the cleaning status of a specific garment.

**Lifecycle:** `Clean` → (worn) → `Dirty` → (sent) → `In Laundry` → (returned) → `Clean`

**Example garment IDs:** GAR-001, GAR-002, GAR-005, GAR-006, GAR-007, GAR-008, GAR-009, GAR-010, GAR-012
    """,
)
async def update_garment_status(garment_id: str, body: StatusUpdateBody):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialised.")

    status_map = {
        'clean':      CleaningStatus.CLEAN,
        'dirty':      CleaningStatus.DIRTY,
        'in laundry': CleaningStatus.IN_LAUNDRY,
    }
    key = body.status.lower()
    if key not in status_map:
        raise HTTPException(400, f"Invalid status '{body.status}'. Use: Clean, Dirty, In Laundry")

    s = status_map[key]
    if s == CleaningStatus.DIRTY:      pipeline.mark_garment_worn(garment_id)
    elif s == CleaningStatus.IN_LAUNDRY: pipeline.mark_garment_laundering(garment_id)
    elif s == CleaningStatus.CLEAN:    pipeline.mark_garment_clean(garment_id)

    rec = pipeline._wardrobe_db.get(garment_id)
    if rec is None:
        raise HTTPException(404, f"Garment '{garment_id}' not found.")

    return {
        'garmentId':      garment_id,
        'name':           rec.name,
        'previousStatus': body.status,
        'newStatus':      rec.cleaning_status.value,
        'isWearable':     rec.is_wearable,
    }


# ── Avatar / Animation Endpoints ──────────────────────────────────────────────

@app.get(
    '/api/animations',
    tags=['Avatar'],
    summary="List Mixamo animation keys",
    description="Returns all available animation clip keys and their Mixamo track names.",
)
async def list_animations():
    from src.avatar_manager import MIXAMO_ANIMATIONS
    return {
        'animations': MIXAMO_ANIMATIONS,
        'usage': "Pass animation_key value to /api/demo/tryon or /api/tryon",
    }


@app.get(
    '/api/sizing/body-types',
    tags=['Body Calibration'],
    summary="List body type classifications",
    description="Returns all recognised body type labels used by NisfaMatchmaking for outfit filtering.",
)
async def list_body_types():
    return {
        'bodyTypes': [
            {'key': 'hourglass',           'description': 'Balanced shoulder/hip, narrow waist'},
            {'key': 'inverted_triangle',   'description': 'Broad shoulders relative to hips'},
            {'key': 'pear',                'description': 'Hips wider than shoulders'},
            {'key': 'rectangle',           'description': 'Similar chest, waist, and hip measurements'},
        ]
    }


@app.post(
    '/api/sizing/validate',
    tags=['Body Calibration'],
    summary="Validate body measurements only",
    description="""
Run **Stage 1 only** — validates measurements and returns the sizing profile
and avatar scale parameters **without** running face processing or outfit recommendation.
Useful for testing measurement validation in isolation.
    """,
)
async def validate_measurements(body: DemoTryOnBody):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialised.")

    from src.body_calibration import BodyCalibrator, BodyMeasurements

    cal = BodyCalibrator()
    val, scale, profile = cal.calibrate(
        shoulder_width_cm = body.shoulder_width_cm,
        chest_cm          = body.chest_cm,
        waist_cm          = body.waist_cm,
        height_cm         = body.height_cm,
        hip_cm            = body.hip_cm,
        inseam_cm         = body.inseam_cm,
        weight_kg         = body.weight_kg,
    )

    if not val.is_valid:
        raise HTTPException(422, detail={'errors': val.errors, 'warnings': val.warnings})

    return {
        'valid':    True,
        'warnings': val.warnings,
        'sizingProfile': {
            'standardSize':  profile.standard_size,
            'bodyType':      profile.body_type,
            'waistHipRatio': profile.waist_hip_ratio,
            'torsoLengthCm': profile.torso_length_cm,
            'shoulderRatio': profile.shoulder_ratio,
        },
        'avatarScaleParams': scale.to_dict(),
    }


# ── Research / Admin Endpoints ────────────────────────────────────────────────

@app.post(
    '/api/train',
    tags=['Research'],
    summary="Trigger CNN model training",
    description="""
Starts training the facial keypoint CNN on `training.csv` **in a background thread**.
The best model is saved to `models/face_keypoint_cnn.pth`.
Training plots are saved to `models/training_curves.png` and `models/keypoint_predictions.png`.

> ⚠️ This is a long-running operation (~25 min CPU / ~5 min GPU). Server remains responsive.
    """,
)
async def trigger_training(body: TrainTriggerBody):
    import concurrent.futures
    from src.face_keypoint_model import train

    loop     = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop.run_in_executor(executor, lambda: train(epochs=body.epochs, batch_size=body.batch_size))

    return {
        'message':     f'Training started — epochs={body.epochs}, batch_size={body.batch_size}.',
        'modelPath':   'models/face_keypoint_cnn.pth',
        'plotPaths':   ['models/training_curves.png', 'models/keypoint_predictions.png'],
        'note':        'Server remains responsive while training runs in background.',
    }


@app.get(
    '/api/wardrobe/history',
    tags=['Wardrobe'],
    summary="Full cleaning status change history",
    description="Returns recent status change events for all garments (SQLite-persisted).",
)
async def wardrobe_history(limit: int = 50):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialised.")
    return {'history': pipeline._wardrobe_db.get_status_history(limit=limit)}


@app.get(
    '/api/wardrobe/{garment_id}/history',
    tags=['Wardrobe'],
    summary="Garment status history",
    description="Returns status change history for a single garment. Example: GAR-001",
)
async def garment_history(garment_id: str):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialised.")
    return {
        'garmentId':  garment_id,
        'wearCount':  pipeline._wardrobe_db.get_wear_count(garment_id),
        'history':    pipeline._wardrobe_db.get_status_history(garment_id),
    }


@app.get(
    '/api/wardrobe/analytics/most-worn',
    tags=['Wardrobe'],
    summary="Most worn garments",
    description="Returns top-N garments ranked by number of times marked as Dirty (worn).",
)
async def most_worn(top_n: int = 5):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialised.")
    return {'mostWorn': pipeline._wardrobe_db.get_most_worn(top_n)}



# ── Accuracy / Model Comparison Endpoints ─────────────────────────────────────

def _get_registry():
    global _model_registry
    if _model_registry is None:
        from src.model_registry import ModelRegistry
        _model_registry = ModelRegistry()
    return _model_registry


@app.get(
    '/api/accuracy/catalogue',
    tags=['Model Accuracy'],
    summary="List all models per stage",
    description="Returns the full model catalogue for all 4 stages — names, types, metrics. No evaluation triggered.",
)
async def accuracy_catalogue():
    return _get_registry().get_model_catalogue()


@app.post(
    '/api/accuracy/stage/{stage_id}',
    tags=['Model Accuracy'],
    summary="Evaluate models for one stage",
    description="""
Trains (if needed) and evaluates all models for the specified stage.

| Stage | Models Compared | Key Metric |
|-------|----------------|------------|
| 1     | Rule-Based vs Random Forest vs Gradient Boosting | Body type accuracy |
| 2     | DeepFaceCNN vs LightFaceCNN | MAE in pixels |
| 3     | Heuristic vs TF-IDF Content-Based | Precision@K |
| 4     | Linear Scaler vs Ridge vs Lasso Regression | RMSE |

⚠️ Stage 2 requires `training.csv` and a trained DeepFaceCNN model.
    """,
)
async def evaluate_stage(stage_id: int):
    if stage_id not in (1, 2, 3, 4):
        raise HTTPException(400, "stage_id must be 1, 2, 3, or 4.")
    reg = _get_registry()
    try:
        if   stage_id == 1: report = reg.evaluate_stage1()
        elif stage_id == 2: report = reg.evaluate_stage2(retrain_light=False)
        elif stage_id == 3: report = reg.evaluate_stage3()
        elif stage_id == 4: report = reg.evaluate_stage4()
        return report
    except Exception as e:
        raise HTTPException(500, f"Evaluation failed: {e}")


@app.post(
    '/api/accuracy/all',
    tags=['Model Accuracy'],
    summary="Evaluate all 4 stages",
    description="Runs accuracy evaluation for all pipeline stages. Long-running (30s–5min depending on hardware).",
)
async def evaluate_all():
    import concurrent.futures
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    result   = await loop.run_in_executor(executor, _get_registry().evaluate_all)
    return result


@app.post(
    '/api/accuracy/stage/1/predict',
    tags=['Model Accuracy'],
    summary="Compare body type predictions from all Stage 1 models",
    description="Run all 3 Stage 1 models on given measurements and compare predictions.",
)
async def compare_stage1_predict(body: DemoTryOnBody):
    try:
        return _get_registry().predict_stage1(
            body.shoulder_width_cm, body.chest_cm,
            body.waist_cm, body.height_cm,
            body.hip_cm, body.inseam_cm,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post(
    '/api/accuracy/stage/4/predict',
    tags=['Model Accuracy'],
    summary="Compare avatar scale predictions from all Stage 4 models",
    description="Run all 3 Stage 4 models on given measurements and compare scale parameters.",
)
async def compare_stage4_predict(body: DemoTryOnBody):
    try:
        return _get_registry().predict_stage4(
            body.shoulder_width_cm, body.chest_cm,
            body.waist_cm, body.height_cm,
            body.hip_cm, body.inseam_cm,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(
    '/api/accuracy/stage/{stage_id}/cached',
    tags=['Model Accuracy'],
    summary="Get last cached accuracy report for a stage",
    description="Returns the most recent evaluation report without re-running evaluation.",
)
async def get_cached_accuracy(stage_id: int):
    report = _get_registry().get_cached_report(stage_id)
    if report is None:
        raise HTTPException(404, f"No cached report for stage {stage_id}. Run POST /api/accuracy/stage/{stage_id} first.")
    return report


@app.get(
    '/api/model/status',
    tags=['Research'],
    summary="Check if trained CNN model exists",
    description="Returns whether the trained model file is present on disk and its file size.",
)
async def model_status():
    from src.face_keypoint_model import MODEL_SAVE_PATH
    exists = os.path.isfile(MODEL_SAVE_PATH)
    size   = os.path.getsize(MODEL_SAVE_PATH) if exists else 0
    return {
        'modelPath':    MODEL_SAVE_PATH,
        'exists':       exists,
        'fileSizeMB':   round(size / 1_048_576, 2),
        'cnnLoaded':    pipeline._face_proc._use_cnn if pipeline else False,
    }


# ── WebSocket: Real-time Try-On ───────────────────────────────────────────────

@app.websocket('/ws/tryon')
async def websocket_tryon(websocket: WebSocket):
    """
    WebSocket for real-time landmark streaming (live camera).
    Send: {"type":"frame","imageB64":"<base64 JPEG>"}
    Receive: {"type":"render","landmarks15":{...},"landmarks468":[...],"headPose":{...}}
    """
    await websocket.accept()
    logger.info("[WS] Client connected.")
    try:
        while True:
            data    = await websocket.receive_json()
            img_b64 = data.get('imageB64', '')
            if data.get('type') != 'frame' or not img_b64:
                continue
            img_bytes = base64.b64decode(img_b64)
            arr       = np.frombuffer(img_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                await websocket.send_json({'type': 'error', 'msg': 'Bad frame'})
                continue
            if pipeline and pipeline._face_proc:
                profile = pipeline._face_proc.process(frame_bgr)
                await websocket.send_json({
                    'type':        'render',
                    'landmarks15': {k: {'x': v[0], 'y': v[1]} for k, v in profile.landmarks_15.items()},
                    'landmarks468': [{'x': lm.x_px, 'y': lm.y_px, 'z': lm.z} for lm in profile.landmarks_468],
                    'headPose':    {'yaw': profile.yaw_deg, 'pitch': profile.pitch_deg},
                    'interEyeDist': profile.inter_eye_dist,
                })
    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected.")
    except Exception as e:
        logger.error(f"[WS] Error: {e}")
        await websocket.close(code=1011)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=False, workers=1)
