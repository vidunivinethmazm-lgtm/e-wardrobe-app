"""Paints a user-supplied clothing photo directly onto the avatar body's own
texture atlas, instead of cutting a separate garment mesh (see
`garment_mesh.py`'s band-cut approach). For real, topologically-complex
bodies (e.g. a Renderpeople scan, as opposed to the simple synthetic
placeholder this pipeline was originally designed around), a separate
band-cut mesh doesn't track the body's actual surface well - it floats as an
offset blob. Painting onto the body's own cylindrical UV (see
`makehuman_mesh._load_base` / `build_personalized_glb`'s texture, and
`generate_makehuman_avatars`-style bodies use the same
u=angle-around-Y-axis, v=height-fraction convention) guarantees the
"garment" follows the real geometry exactly, since it IS the real geometry,
just re-textured in that region.

Trade-off: this is a texture decal, not a modeled garment - no sleeve
silhouette, no draping. It always looks "painted on" rather than "worn",
but it never floats, gaps, or misaligns the way a separately-cut mesh does
on a complex body.
"""
from __future__ import annotations

import io

from PIL import Image, ImageOps

from .makehuman_mesh import TEXTURE_SIZE

# Height-fraction (v) bands. upper_body/dress's upper bound is 0.80, not
# garment_mesh.py's 0.86 (tuned for the simple synthetic placeholder body):
# on the real Renderpeople mesh this asset now uses, the neck narrows
# starting around v=0.81 and the jaw/head widen again by v=0.85 - 0.86 lands
# on the neck/chin, which is what caused a t-shirt's paint to bleed onto the
# neck. 0.80 stays on the shoulder/collar side of that transition (measured
# directly from the base mesh's own per-height radius profile).
CATEGORY_V_BAND: dict[str, tuple[float, float]] = {
    "upper_body": (0.55, 0.80),
    "lower_body": (0.04, 0.53),
    "dress": (0.35, 0.80),
}

# Full cylindrical wrap (u = atan2(z, x)/(2*pi) + 0.5, computed around the
# BODY's central vertical axis - see garment_texture_paint's module
# docstring). A narrower "front-facing hemisphere" band was tried first, but
# it only reads correctly for geometry centred on that axis (the torso): an
# off-axis limb (a leg, offset left/right of centre) spans a much narrower,
# differently-positioned slice of the same 0-1 u range, so a fixed
# front-hemisphere band paints one leg while mostly missing the other. Full
# wrap covers both legs symmetrically (and any garment, front+back) at the
# cost of a visible seam down the back where u wraps 0->1, since a flat
# photo generally isn't seamlessly tileable - an acceptable trade for
# correctness over a purely front-only decal.
FULL_U_BAND: tuple[float, float] = (0.0, 1.0)

# An earlier version of this module tried to detect and exclude arm/hand
# vertices from the paint region (a per-height radius-outlier + connected-
# component heuristic), since a hanging arm can sit at the same height as
# the torso/legs and pick up their color. That approach was reverted: the
# arm is physically attached to the shoulder, so in any consistent UV
# projection they form ONE continuous region with no gap to threshold
# on - loosening the detector enough to catch the arm always caught the
# shoulder too (leaving a bare gap in the shirt, worse than a tinted hand),
# and tightening it back off caught nothing at all. There was no usable
# middle ground without real per-vertex body-part labels (this mesh has
# none - see PLACEHOLDER_README.txt). A tinted hand is an accepted,
# documented limitation instead of a hole in the garment.


def paint_garment_onto_texture(texture_png: bytes, garment_png: bytes, category: str, gender: str = "female") -> bytes:
    """Returns a new texture PNG with `garment_png` pasted into the UV
    region for `category`, alpha-blended over whatever was there (skin tone
    / previously baked face or garment paint). `gender` is accepted for
    interface stability (an earlier version needed it for a
    per-body-geometry mask) but currently unused."""
    v_band = CATEGORY_V_BAND.get(category)
    if v_band is None:
        raise ValueError(f"unknown category {category!r}")
    v0, v1 = v_band
    u0, u1 = FULL_U_BAND

    base = Image.open(io.BytesIO(texture_png)).convert("RGBA")
    if base.size != (TEXTURE_SIZE, TEXTURE_SIZE):
        base = base.resize((TEXTURE_SIZE, TEXTURE_SIZE))

    garment = Image.open(io.BytesIO(garment_png)).convert("RGBA")

    # Three.js's default Texture.flipY=true (used for the body/face texture
    # load path in AvatarViewer3D, unlike the separate-garment-mesh path
    # which explicitly sets flipY=false) means v=0 samples the BOTTOM row of
    # this source PNG and v=1 samples the TOP row - so higher v (closer to
    # the head) must land nearer y=0 in image space.
    y0 = int(round((1 - v1) * TEXTURE_SIZE))
    y1 = int(round((1 - v0) * TEXTURE_SIZE))
    x0 = int(round(u0 * TEXTURE_SIZE))
    x1 = int(round(u1 * TEXTURE_SIZE))

    band_w, band_h = max(1, x1 - x0), max(1, y1 - y0)
    garment_resized = garment.resize((band_w, band_h), Image.LANCZOS)
    # Cylindrical u increases in the opposite rotational sense from how a
    # viewer sees left/right when facing the body's front hemisphere -
    # confirmed empirically (a labelled test image came out mirrored) -
    # so flip horizontally to compensate.
    garment_resized = ImageOps.mirror(garment_resized)

    base.alpha_composite(garment_resized, (x0, y0))

    buf = io.BytesIO()
    base.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()
