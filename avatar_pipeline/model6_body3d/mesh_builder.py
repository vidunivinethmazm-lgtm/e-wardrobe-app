"""
Model 6 — 3D Body Reconstruction: procedural humanoid mesh.

Turns a `params.PARAM_NAMES` dict (head/neck/torso/limb sizes, each a
fraction of total height) plus a height (cm), a skin RGB color, and
(optionally) the user's own face crop + hair color (see `face_features.
extract_face_features`) into a low-poly humanoid triangle mesh: a torso
(lofted elliptical rings from neck to hip), two arms, two legs (each a
lofted tube with circular cross-sections, hands/feet as small spheres), a
UV-textured head, a "helmet" hair cap, and small eye/eyebrow/nose/mouth/ear
features.

This is the 3D analogue of `model4_avatar.synthetic_avatars.render_avatar`
(the 2D paper-doll renderer): a deliberately simple, fully programmatic
geometry that (a) is what `server.mock_pipeline` returns directly (no
TensorFlow needed), and (b) gives Model 6's CNN (`architecture.py`) a
well-defined target to regress towards when trained on real photos.

Coordinate system: meters, y-up, x-right, z-forward (+Z is "front", facing
the camera in `AvatarViewer3D`'s default pose). The mesh spans y in
[0, height_cm / 100] and is centered on the x=0/z=0 vertical axis.

`build_avatar_mesh` returns `{"parts": [...], "images": [...]}` — multiple
mesh primitives (each with its own material/color or texture), packaged
into a single `.glb` by `glb_export.mesh_to_glb_bytes`. See that module for
the exact "part"/material shape.
"""

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .face_features import DEFAULT_HAIR_RGB
from .face_measurements import (
    NEUTRAL_MEASUREMENTS,
    compute_face_measurements,
    measurements_to_head_params,
)
from .face_texture_builder import build_head_texture_warped

# MediaPipe FaceLandmarker can produce either 468 (standard) or 478 (with
# iris landmarks) points.  Both are valid for Delaunay-triangulation warping.
# The ~108-point Haar-cascade fallback does NOT match the MediaPipe topology
# and will produce a garbled warp, so we require >= 468.
_MIN_MEDIAPIPE_LANDMARK_COUNT = 468


def _has_proper_landmarks(landmarks_2d):
    """True when ``landmarks_2d`` has >= 468 MediaPipe face-mesh points
    (standard 468 or the newer 478 that includes iris landmarks)."""
    return (
        landmarks_2d is not None
        and hasattr(landmarks_2d, "shape")
        and landmarks_2d.shape[0] >= _MIN_MEDIAPIPE_LANDMARK_COUNT
    )


def _png_is_non_trivial(png_bytes, skin_rgb, min_bytes=600):
    """Heuristic: a PNG significantly larger than a flat fill of the same
    colour must contain visible face content (warped landmarks, blending
    artefacts, etc.).  A pure flat fill at 256×256 is ≈360 B; a warped
    face is typically 5-25 KB."""
    flat_size = len(_build_flat_png(skin_rgb))
    return len(png_bytes) > max(flat_size * 1.5, min_bytes)


def _build_flat_png(skin_rgb):
    """Return PNG bytes for a solid ``skin_rgb`` fill at ``_HEAD_TEXTURE_SIZE``."""
    import io
    from PIL import Image as _PIL
    img = _PIL.new("RGB", (_HEAD_TEXTURE_SIZE, _HEAD_TEXTURE_SIZE),
                    tuple(int(c) for c in np.clip(skin_rgb, 0, 255)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# Vertical layout, as a fraction of total height (y=0 at the feet).
ANCHORS = {
    "head_top": 1.00,
    "neck": 0.86,
    "shoulder": 0.82,
    "chest": 0.72,
    "waist": 0.60,
    "hip": 0.50,
    "elbow": 0.62,
    "wrist": 0.44,
    "hand_tip": 0.38,
    "knee": 0.28,
    "ankle": 0.06,
}

N_RING_SEGMENTS = 16
N_SPHERE_LAT = 8

# Facial-feature layout, all relative to `head_radius` (hr) and the head
# center `head_center = (0, H - hr, 0)`. +Z is "front".
_EYE_X = 0.40
_EYE_Y = 0.05
_EYE_Z = 0.85
_EYE_RADIUS = 0.13
_IRIS_Z = 0.93
_IRIS_RADIUS = 0.07
_EYEBROW_Y = 0.24
_EYEBROW_Z = 0.88
_EYEBROW_RADII = (0.16, 0.045, 0.05)
_NOSE_Y = -0.05
_NOSE_Z = 0.92
_NOSE_RADII = (0.13, 0.20, 0.22)
_MOUTH_Y = -0.35
_MOUTH_Z = 0.88
_MOUTH_RADII = (0.22, 0.07, 0.08)
_EAR_X = 0.95
_EAR_RADII = (0.10, 0.22, 0.16)

_EYE_WHITE_RGB = (250.0, 250.0, 248.0)
_EYE_IRIS_RGB = (75.0, 50.0, 35.0)
_MOUTH_TINT_RGB = np.array([150.0, 70.0, 70.0], dtype=np.float32)

_HAIR_COVERAGE = 0.4
_HAIR_PUFF = 1.08
_HEAD_TEXTURE_SIZE = 256


class _MeshBuilder:
    """Accumulates vertices/triangles for one mesh part."""

    def __init__(self):
        self._verts = []
        self._faces = []

    def add_ring(self, y, rx, rz, cx=0.0, cz=0.0, n=N_RING_SEGMENTS):
        """Adds a horizontal ellipse of `n` vertices at height `y`, centered
        at (cx, y, cz). Returns the list of new vertex indices, in angular
        order."""
        base = len(self._verts)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for angle in angles:
            self._verts.append((cx + rx * np.cos(angle), y, cz + rz * np.sin(angle)))
        return [base + i for i in range(n)]

    def add_point(self, x, y, z):
        self._verts.append((x, y, z))
        return len(self._verts) - 1

    def loft(self, ring_a, ring_b):
        """Connects two same-size rings with a band of triangles."""
        n = len(ring_a)
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = ring_a[i], ring_a[j], ring_b[j], ring_b[i]
            self._faces.append((a, b, c))
            self._faces.append((a, c, d))

    def fan_cap(self, ring, apex, flip=False):
        """Closes `ring` with a fan of triangles to a single `apex` vertex."""
        n = len(ring)
        for i in range(n):
            j = (i + 1) % n
            if flip:
                self._faces.append((apex, ring[j], ring[i]))
            else:
                self._faces.append((apex, ring[i], ring[j]))

    def add_ellipsoid(self, center, rx, ry, rz, n_lat=N_SPHERE_LAT, n_lon=N_RING_SEGMENTS):
        """Adds a UV-sphere scaled independently along each axis. The pole
        rings produce a few zero-area triangles, which is harmless (they
        simply don't render)."""
        base = len(self._verts)
        for i in range(n_lat + 1):
            theta = np.pi * i / n_lat
            y = np.cos(theta)
            r = np.sin(theta)
            for j in range(n_lon):
                phi = 2 * np.pi * j / n_lon
                x = r * np.cos(phi)
                z = r * np.sin(phi)
                self._verts.append((
                    center[0] + x * rx,
                    center[1] + y * ry,
                    center[2] + z * rz,
                ))
        for i in range(n_lat):
            for j in range(n_lon):
                a = base + i * n_lon + j
                b = base + i * n_lon + (j + 1) % n_lon
                c = base + (i + 1) * n_lon + (j + 1) % n_lon
                d = base + (i + 1) * n_lon + j
                self._faces.append((a, b, c))
                self._faces.append((a, c, d))

    def add_sphere(self, center, radius, n_lat=N_SPHERE_LAT, n_lon=N_RING_SEGMENTS):
        self.add_ellipsoid(center, radius, radius, radius, n_lat, n_lon)

    def build(self):
        vertices = np.asarray(self._verts, dtype=np.float32)
        faces = np.asarray(self._faces, dtype=np.uint32)
        normals = _compute_vertex_normals(vertices, faces)
        return vertices, faces, normals


def _compute_vertex_normals(vertices, faces):
    normals = np.zeros_like(vertices)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return (normals / lengths).astype(np.float32)


def _add_limb(mb, x, top_y, mid_y, bottom_y, top_r, bottom_r, tip_center_y, tip_radius, cap_top_flip):
    """Adds one arm or leg: a 3-ring loft (top -> mid -> bottom, tapering
    `top_r` -> `bottom_r`), a flat cap at the top ring, and a small sphere
    ("hand"/"foot") around `bottom`."""
    mid_r = (top_r + bottom_r) / 2.0
    top_ring = mb.add_ring(top_y, top_r, top_r, cx=x)
    mid_ring = mb.add_ring(mid_y, mid_r, mid_r, cx=x)
    bottom_ring = mb.add_ring(bottom_y, bottom_r, bottom_r, cx=x)

    mb.loft(top_ring, mid_ring)
    mb.loft(mid_ring, bottom_ring)

    top_apex = mb.add_point(x, top_y, 0.0)
    mb.fan_cap(top_ring, top_apex, flip=cap_top_flip)

    mb.add_sphere((x, tip_center_y, 0.0), tip_radius)


def _add_hair_cap(mb, center, head_radius, n_lat=N_SPHERE_LAT, n_lon=N_RING_SEGMENTS,
                  coverage=_HAIR_COVERAGE, puff=_HAIR_PUFF,
                  hair_style="short"):
    """Adds a 'helmet' hair shell covering the top `coverage` fraction of
    the head's latitude range, tapering from `puff * head_radius` at the
    crown to just outside `head_radius` at its rim, where a flat fan cap
    closes it off.

    ``hair_style`` adjusts the shape:
    - ``"short"`` / ``"buzz"`` — tight to the head (low puff, low coverage)
    - ``"medium"`` — default bowl cut
    - ``"long"`` / ``"straight"`` / ``"wavy"`` — larger puff, more coverage
    - ``"curly"`` — extra puff for volume
    - ``"ponytail"`` — standard coverage, extra puff at crown

    The style-aware shaping gives the procedural avatar a rough visual match
    to the selfie-detected hair style (Phase 1), even without a separate hair
    mesh GLB asset.
    """
    # Adjust coverage and puff based on detected hair style
    style_cfg = {
        "buzz":     {"coverage": 0.30, "puff": 1.02},
        "short":    {"coverage": 0.35, "puff": 1.06},
        "medium":   {"coverage": 0.40, "puff": 1.08},
        "straight": {"coverage": 0.45, "puff": 1.10},
        "long":     {"coverage": 0.50, "puff": 1.12},
        "wavy":     {"coverage": 0.48, "puff": 1.14},
        "curly":    {"coverage": 0.50, "puff": 1.18},
        "ponytail": {"coverage": 0.42, "puff": 1.15},
    }
    cfg = style_cfg.get(hair_style, style_cfg["medium"])
    coverage = cfg["coverage"]
    puff = cfg["puff"]

    k = max(1, int(round(coverage * n_lat)))
    rings = []
    for i in range(k + 1):
        theta = np.pi * i / n_lat
        y = np.cos(theta)
        r_xy = np.sin(theta)
        radius = head_radius * (puff + (1.005 - puff) * (i / k))
        ring = mb.add_ring(center[1] + y * radius, r_xy * radius, r_xy * radius,
                            cx=center[0], cz=center[2], n=n_lon)
        rings.append((ring, center[1] + y * radius))

    for (ring_a, _), (ring_b, _) in zip(rings, rings[1:]):
        mb.loft(ring_a, ring_b)

    rim_ring, rim_y = rings[-1]
    apex = mb.add_point(center[0], rim_y, center[2])
    mb.fan_cap(rim_ring, apex, flip=False)


def _build_head(center, radius, n_lat=N_SPHERE_LAT, n_lon=N_RING_SEGMENTS,
                rx=None, ry=None, rz=None):
    """Returns (vertices, faces, uvs) for a UV-mapped ellipsoid head.

    When rx/ry/rz are provided the head is shaped as an ellipsoid matching
    the user's detected face proportions (wider for a round face, narrower
    for an oval one).  ``radius`` is used as the fallback for any axis that
    is not specified.

    The seam column is duplicated (n_lon + 1 columns per ring) so the
    texture wraps cleanly: u=0/1 is the seam at the back of the head (-Z),
    u=0.5 faces +Z (the camera/'front'), and v=0 is the bottom (neck),
    v=1 is the top (crown — matching OpenGL/glTF convention)."""
    if rx is None:
        rx = radius
    if ry is None:
        ry = radius
    if rz is None:
        rz = radius

    verts, uvs = [], []
    phi_back = 1.5 * np.pi
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        y_unit = np.cos(theta)
        r_xy = np.sin(theta)
        v = 1.0 - i / n_lat   # glTF convention: v=0 = bottom, v=1 = top
        for c in range(n_lon + 1):
            phi = phi_back + 2 * np.pi * c / n_lon
            x_unit = r_xy * np.cos(phi)
            z_unit = r_xy * np.sin(phi)
            verts.append((center[0] + x_unit * rx,
                          center[1] + y_unit * ry,
                          center[2] + z_unit * rz))
            uvs.append((c / n_lon, v))

    faces = []
    stride = n_lon + 1
    for i in range(n_lat):
        for c in range(n_lon):
            a = i * stride + c
            b = i * stride + c + 1
            d_ = (i + 1) * stride + c
            e = (i + 1) * stride + c + 1
            faces.append((a, b, e))
            faces.append((a, e, d_))

    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(faces, dtype=np.uint32),
        np.asarray(uvs, dtype=np.float32),
    )


def _build_head_texture(skin_rgb, face_crop, texture_size=_HEAD_TEXTURE_SIZE,
                        selfie_rgb=None, landmarks_2d=None, blend_mode="feather",
                        face_width=None, face_height=None):
    """Builds the head mesh's texture: a `texture_size`x`texture_size` PNG.

    The texture is laid out with the face centred in the image (forehead
    toward the top, chin toward the bottom), matching the UV convention
    ``_build_head`` now uses: v=0 at the neck and v=1 at the crown, which
    aligns with OpenGL/glTF expectations (origin bottom-left in shader
    space, automatically corrected by the renderer).

    When ``selfie_rgb`` and ``landmarks_2d`` are both provided, uses
    Delaunay-triangulation-based warping (``face_texture_builder.
    build_head_texture_warped``) for a distortion-free, photorealistic
    face texture.  Otherwise falls back to the simple centre-paste method.

    When ``face_width``/``face_height`` are provided (pixel dimensions of
    the original face crop), the centre-paste respects the face aspect
    ratio for a more natural UV fit.

    See ``build_avatar_mesh`` for parameter docs.
    """
    # Delaunay-triangulation warping requires the full 468 MediaPipe
    # landmarks.  When those are available, try the warped path first
    # and fall back to centre-paste if it returns a flat texture (which
    # happens when the warped image has no visible face content).
    use_warp = (
        selfie_rgb is not None
        and landmarks_2d is not None
        and _has_proper_landmarks(landmarks_2d)
    )
    print(f"[_build_head_texture] use_warp={use_warp} "
          f"(selfie_rgb={'set' if selfie_rgb is not None else 'None'}, "
          f"landmarks_2d={'None' if landmarks_2d is None else getattr(landmarks_2d, 'shape', len(landmarks_2d))})")
    if use_warp:
        warped_png = build_head_texture_warped(
            selfie_rgb, landmarks_2d,
            tuple(int(c) for c in np.clip(skin_rgb, 0, 255)),
            texture_size=texture_size,
            blend_mode=blend_mode,
        )
        # If the warp produced more than just a flat fill, use it
        if _png_is_non_trivial(warped_png, skin_rgb):
            print(f"[_build_head_texture] Delaunay warp path used ({len(warped_png)} bytes)")
            return warped_png
        print("[_build_head_texture] warp produced a flat/trivial result, falling back to centre-paste")
        # Otherwise fall through to centre-paste
    else:
        print("[_build_head_texture] no proper landmarks - using legacy centre-paste fallback")

    # Legacy fallback: simple centre-paste with aspect-ratio awareness
    skin = tuple(int(c) for c in np.clip(skin_rgb, 0, 255))
    image = Image.new("RGB", (texture_size, texture_size), skin)
    if face_crop is not None:
        # ✅ Face Crop Aspect Ratio අනුව Target Size එක Adjust කරන්න
        base_w = int(texture_size * 0.55)
        base_h = int(texture_size * 0.65)
        if face_width is not None and face_height is not None and face_width > 0 and face_height > 0:
            user_aspect = face_width / face_height
            default_aspect = base_w / base_h
            if user_aspect > default_aspect:
                # පළල් මුහුණ — width ප්‍රකාරව scale කරන්න
                face_w = base_w
                face_h = int(base_w / user_aspect)
            else:
                # දිගටි මුහුණ — height ප්‍රකාරව scale කරන්න
                face_h = base_h
                face_w = int(base_h * user_aspect)
            face_w = max(8, face_w)
            face_h = max(8, face_h)
        else:
            face_w, face_h = base_w, base_h

        face_image = Image.fromarray(np.asarray(face_crop, dtype=np.uint8)).resize((face_w, face_h))
        offset = ((texture_size - face_w) // 2, int(texture_size * 0.28))
        # Elliptical mask: 22% top inset cuts hair above forehead; 6% sides/bottom keep ears and chin
        top_inset = max(2, int(face_h * 0.22))
        side_inset = max(2, int(face_w * 0.06))
        bot_inset = max(2, int(face_h * 0.06))
        mask = Image.new("L", (face_w, face_h), 0)
        ImageDraw.Draw(mask).ellipse(
            [side_inset, top_inset, face_w - side_inset, face_h - bot_inset], fill=255
        )
        mask = mask.filter(ImageFilter.GaussianBlur(max(3, int(min(face_w, face_h) * 0.06))))
        image.paste(face_image, offset, mask)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _part(mb, base_color_rgb, name, uvs=None, texture_index=None):
    vertices, faces, normals = mb.build()
    return {
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "uvs": uvs,
        "material": {
            "name": name,
            "base_color_rgb": tuple(float(c) for c in base_color_rgb),
            "texture_index": texture_index,
        },
    }


def build_avatar_mesh(params, height_cm, skin_rgb, face_crop=None, hair_rgb=None,
                      selfie_rgb=None, landmarks_2d=None, blend_mode="feather",
                      face_width=None, face_height=None, face_measurements=None,
                      hair_style="medium"):
    """params: dict with `params.PARAM_NAMES` keys (fractions of height).
    height_cm: user's height in cm — sets the overall mesh scale (meters).
    skin_rgb: (3,) array-like, 0-255 — the body/face/nose/ear color.
    face_crop: optional `texture_size`x`texture_size`x3 uint8 array (see
        `face_features.extract_face_features`) — the user's own face,
        textured onto the front of the head mesh.
    hair_rgb: optional (3,) array-like, 0-255 — the hair/eyebrow color
        (defaults to `face_features.DEFAULT_HAIR_RGB`).
    selfie_rgb: optional (H, W, 3) uint8 — the full selfie photo. When
        provided together with ``landmarks_2d``, enables Delaunay-triangulation
        warping for a distortion-free face texture (instead of simple paste).
    landmarks_2d: optional (N, 2) MediaPipe face-mesh landmarks (in selfie
        pixel coordinates).  Requires ``selfie_rgb``.
    blend_mode: ``"feather"`` (default) or ``"poisson"`` — blending method
        for the warped face onto the skin-tone canvas.
    face_width, face_height: optional pixel dimensions of the original face
        crop (before resize).  When provided, the centre-paste texture uses
        the correct aspect ratio for a more natural UV fit.
    face_measurements: optional dict from face_measurements.compute_face_measurements().
        When provided (or derivable from landmarks_2d), the head mesh is
        shaped as an ellipsoid matching the user's detected face proportions
        (wide/oval, eye spacing, nose width, mouth width) — mirroring how
        body measurements shape the body mesh.

    Returns {"parts": [...], "images": [...]}, all geometry in meters — see
    `glb_export.mesh_to_glb_bytes` for the exact "part"/material shape this
    is turned into.
    """
    if hair_rgb is None:
        hair_rgb = DEFAULT_HAIR_RGB
    skin_rgb = np.asarray(skin_rgb, dtype=np.float32)
    hair_rgb = np.asarray(hair_rgb, dtype=np.float32)

    H = height_cm / 100.0

    def r(name):
        return params[name] / 2.0 * H

    def y(name):
        return ANCHORS[name] * H

    # --- "skin" part: torso, arms, legs, nose, ears ---
    body = _MeshBuilder()

    neck_ring = body.add_ring(y("neck"), r("neck_radius"), r("neck_radius"))
    shoulder_ring = body.add_ring(y("shoulder"), r("shoulder_width"), r("chest_depth"))
    chest_ring = body.add_ring(y("chest"), r("chest_width"), r("chest_depth"))
    waist_ring = body.add_ring(y("waist"), r("waist_width"), r("waist_depth"))
    hip_ring = body.add_ring(y("hip"), r("hip_width"), r("hip_depth"))

    body.loft(neck_ring, shoulder_ring)
    body.loft(shoulder_ring, chest_ring)
    body.loft(chest_ring, waist_ring)
    body.loft(waist_ring, hip_ring)

    neck_apex = body.add_point(0.0, y("neck"), 0.0)
    body.fan_cap(neck_ring, neck_apex, flip=True)
    hip_apex = body.add_point(0.0, y("hip"), 0.0)
    body.fan_cap(hip_ring, hip_apex, flip=False)

    # --- Arms (centered just outside the shoulder ring's edge, overlapping
    # it by half the upper-arm radius so they read as attached limbs rather
    # than disappearing inside the torso silhouette) ---
    arm_x = r("shoulder_width") + r("upper_arm_radius") * 0.5
    for side in (-1.0, 1.0):
        _add_limb(
            body, side * arm_x,
            top_y=y("shoulder"), mid_y=y("elbow"), bottom_y=y("wrist"),
            top_r=r("upper_arm_radius"), bottom_r=r("forearm_radius"),
            tip_center_y=y("hand_tip"), tip_radius=r("forearm_radius") * 1.1,
            cap_top_flip=True,
        )

    # --- Legs ---
    leg_x = params["hip_width"] / 4.0 * H
    for side in (-1.0, 1.0):
        _add_limb(
            body, side * leg_x,
            top_y=y("hip"), mid_y=y("knee"), bottom_y=y("ankle"),
            top_r=r("thigh_radius"), bottom_r=r("calf_radius"),
            tip_center_y=y("ankle"), tip_radius=r("calf_radius") * 1.05,
            cap_top_flip=True,
        )

    # --- Head layout ---
    # Body measurements → body shape.  Face measurements → head shape.
    # Derive head geometry parameters from the user's selfie measurements
    # (the same way bust/waist/hips/height drive the body geometry).
    hr = r("head_radius")

    # If no pre-computed measurements were passed, try to derive them from
    # the MediaPipe landmarks that are already available for texture warping.
    _face_meas = face_measurements
    if _face_meas is None and landmarks_2d is not None:
        _face_meas = compute_face_measurements(landmarks_2d)
    if _face_meas is None:
        _face_meas = NEUTRAL_MEASUREMENTS

    hp = measurements_to_head_params(_face_meas, hr)

    head_rx  = hp["head_rx"]     # left-right semi-axis (user's face width)
    head_ry  = hp["head_ry"]     # vertical semi-axis  (body params, unchanged)
    head_rz  = hp["head_rz"]     # front-back semi-axis (≈ 80 % of width)
    EYE_X    = hp["eye_x"]       # fraction of head_rx
    EYE_Y    = hp["eye_y"]       # fraction of head_ry
    EYE_Z    = hp["eye_z"]       # fraction of head_rz
    EYE_R    = hp["eye_radius"]  # fraction of hr
    IRIS_R   = hp["iris_radius"]
    NOSE_Y   = hp["nose_y"]
    NOSE_Z   = hp["nose_z"]
    NOSE_RAD = hp["nose_radii"]
    MOUTH_Y  = hp["mouth_y"]
    MOUTH_Z  = hp["mouth_z"]
    MOUTH_RAD = hp["mouth_radii"]
    EAR_X    = hp["ear_x"]       # fraction of head_rx

    head_center = (0.0, H - head_ry, 0.0)

    # Nose + ears are skin-colored, so they join the "skin" part.
    body.add_ellipsoid(
        (head_center[0],
         head_center[1] + NOSE_Y * head_ry,
         head_center[2] + NOSE_Z * head_rz),
        *(c * hr for c in NOSE_RAD),
    )
    for side in (-1.0, 1.0):
        body.add_ellipsoid(
            (head_center[0] + side * EAR_X * head_rx, head_center[1], head_center[2]),
            *(c * hr for c in _EAR_RADII),
        )

    parts = [_part(body, skin_rgb, "skin")]
    images = []

    # --- "face" part: the head ellipsoid, UV-textured with skin color + the
    # user's own face crop (warped via Delaunay triangulation when
    # landmarks are available).  rx/ry/rz are shaped from the selfie
    # measurements so the head matches the user's actual face proportions. ---
    head_vertices, head_faces, head_uvs = _build_head(
        head_center, hr, rx=head_rx, ry=head_ry, rz=head_rz
    )
    head_normals = _compute_vertex_normals(head_vertices, head_faces)
    images.append(_build_head_texture(skin_rgb, face_crop,
                                      selfie_rgb=selfie_rgb,
                                      landmarks_2d=landmarks_2d,
                                      blend_mode=blend_mode,
                                      face_width=face_width,
                                      face_height=face_height))
    parts.append({
        "vertices": head_vertices,
        "faces": head_faces,
        "normals": head_normals,
        "uvs": head_uvs,
        "material": {"name": "face", "base_color_rgb": (255.0, 255.0, 255.0), "texture_index": 0},
    })

    # --- "hair" part: helmet cap + eyebrows ---
    # Eyebrow X scales with eye spacing (EYE_X fraction of head_rx).
    hair = _MeshBuilder()
    _add_hair_cap(hair, head_center, hr, hair_style=hair_style)
    for side in (-1.0, 1.0):
        hair.add_ellipsoid(
            (side * EYE_X * head_rx,
             head_center[1] + _EYEBROW_Y * head_ry,
             head_center[2] + _EYEBROW_Z * head_rz),
            *(c * hr for c in _EYEBROW_RADII),
        )
    parts.append(_part(hair, hair_rgb, "hair"))

    # --- "eye_white" / "eye_iris" parts ---
    # Eye position uses measurement-derived EYE_X/Y/Z (fractions of the
    # respective ellipsoid semi-axes) so eyes sit correctly for the user's
    # actual eye spacing and face proportions.
    eye_white = _MeshBuilder()
    eye_iris = _MeshBuilder()
    for side in (-1.0, 1.0):
        eye_white.add_sphere(
            (side * EYE_X * head_rx,
             head_center[1] + EYE_Y * head_ry,
             head_center[2] + EYE_Z * head_rz),
            EYE_R * hr,
        )
        eye_iris.add_sphere(
            (side * EYE_X * head_rx,
             head_center[1] + EYE_Y * head_ry,
             head_center[2] + _IRIS_Z * head_rz),
            IRIS_R * hr,
        )
    parts.append(_part(eye_white, _EYE_WHITE_RGB, "eye_white"))
    parts.append(_part(eye_iris, _EYE_IRIS_RGB, "eye_iris"))

    # --- "mouth" part: a skin-tinted-red ellipsoid ---
    mouth = _MeshBuilder()
    mouth.add_ellipsoid(
        (head_center[0],
         head_center[1] + MOUTH_Y * head_ry,
         head_center[2] + MOUTH_Z * head_rz),
        *(c * hr for c in MOUTH_RAD),
    )
    mouth_rgb = 0.5 * skin_rgb + 0.5 * _MOUTH_TINT_RGB
    parts.append(_part(mouth, mouth_rgb, "mouth"))

    return {"parts": parts, "images": images}
