"""Model 6 — garment meshes: builds a `.glb` for one catalog garment, shaped
to match a specific avatar's body.

There is no separate rigged garment asset or skeleton anywhere in this
pipeline (`makehuman_mesh.py` bakes a single static mesh + 6 blend-shape
morph targets, no bones/skin). So instead of skinning a garment to bones,
each garment is built as a *sub-mesh cut directly from the same MakeHuman
base body mesh* used by `makehuman_mesh.build_personalized_glb`: a vertical
band of the body (e.g. torso for a t-shirt, hips+legs for pants) is selected,
pushed outward slightly so it sits just above the skin, and given a flat
tint. Because it's literally a subset of the body's own vertices and morph
deltas, applying the *same* morph weights + height scale as the body keeps
it perfectly aligned as body shape changes -- the "shared skeleton" effect,
without any actual skeleton.

This is a placeholder garment style (fitted "second-skin" shapes, not real
cloth silhouettes) meant to make the wear-a-garment pipeline work end to end.
Swapping in real modelled garments later means replacing `_GARMENT_CATALOG`'s
band-based selection with an actual authored mesh that shares the base body's
6 morph target deltas -- the rest of this module (weight/scale application,
GLB writing) stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pygltflib import (
    GLTF2, Accessor, Asset, Buffer, BufferView,
    Material, Mesh, Node, PbrMetallicRoughness, Primitive, Scene,
)

from .makehuman_mesh import (
    MORPH_TARGET_NAMES, _ASSET_GENDER, ASSETS_DIR,
    _read_vec3_accessor, compute_morph_weights,
)

_POSITION_COMPONENT_TYPE = 5126  # FLOAT
_SCALAR = "SCALAR"
_VEC3 = "VEC3"
_VEC2 = "VEC2"
_TEX_COORD_TYPE = 5126  # FLOAT

# Matches face_customization._HEIGHT_CM_BY_GENDER -- garments must scale to
# the exact same height as the body they're worn on.
_HEIGHT_CM_BY_GENDER = {"male": 170, "female": 160, "neutral": 160}


@dataclass(frozen=True)
class GarmentDef:
    id: str
    name: str
    category: str  # "upper_body" | "lower_body" | "dress"
    color_hex: str
    y_band: tuple[float, float]  # fraction of body height, 0=feet, 1=head top
    offset: float  # metres, pushed outward from the body's central vertical axis


# Rough proportions for the MakeHuman base mesh (standing, feet at y=0).
# These are approximate band cuts, not a real garment silhouette -- see the
# module docstring. Tune `y_band` per item if a cut looks off on the actual
# base mesh.
_GARMENT_CATALOG: dict[str, GarmentDef] = {
    item.id: item
    for item in [
        GarmentDef("tshirt_white", "White T-Shirt", "upper_body", "#f5f5f5", (0.55, 0.86), 0.012),
        GarmentDef("tshirt_navy", "Navy T-Shirt", "upper_body", "#1f2a44", (0.55, 0.86), 0.012),
        GarmentDef("tshirt_black", "Black T-Shirt", "upper_body", "#1a1a1a", (0.55, 0.86), 0.012),
        GarmentDef("jeans_blue", "Blue Jeans", "lower_body", "#3b5c8a", (0.04, 0.53), 0.010),
        GarmentDef("pants_khaki", "Khaki Pants", "lower_body", "#a68a5c", (0.04, 0.53), 0.010),
        GarmentDef("shorts_black", "Black Shorts", "lower_body", "#222222", (0.04, 0.53), 0.010),
        GarmentDef("dress_red", "Red Dress", "dress", "#a4283c", (0.35, 0.86), 0.014),
        GarmentDef("dress_black", "Black Dress", "dress", "#111111", (0.35, 0.86), 0.014),
    ]
}


def list_garments() -> list[dict]:
    return [
        {"id": g.id, "name": g.name, "category": g.category, "color": g.color_hex}
        for g in _GARMENT_CATALOG.values()
    ]


def get_garment(garment_id: str) -> GarmentDef | None:
    return _GARMENT_CATALOG.get(garment_id)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


@lru_cache(maxsize=2)
def _load_base_full(gender: str):
    """Like `makehuman_mesh._load_base`, but also returns the triangle index
    buffer, needed here to cut a watertight sub-mesh out of the body."""
    glb_bytes = (ASSETS_DIR / f"{gender}.glb").read_bytes()
    gltf = GLTF2.load_from_bytes(glb_bytes)
    blob = gltf.binary_blob()

    prim = gltf.meshes[0].primitives[0]
    positions = _read_vec3_accessor(gltf, blob, prim.attributes.POSITION)

    idx_acc = gltf.accessors[prim.indices]
    idx_bv = gltf.bufferViews[idx_acc.bufferView]
    idx_offset = idx_bv.byteOffset + (idx_acc.byteOffset or 0)
    idx_dtype = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32}[idx_acc.componentType]
    indices = np.frombuffer(
        blob[idx_offset: idx_offset + idx_acc.count * np.dtype(idx_dtype).itemsize],
        dtype=idx_dtype,
    ).astype(np.uint32).reshape(-1, 3)

    target_names = gltf.meshes[0].extras["targetNames"]
    deltas = {
        name: _read_vec3_accessor(gltf, blob, target["POSITION"])
        for name, target in zip(target_names, prim.targets)
    }
    return positions, indices, deltas


def _cut_band(positions: np.ndarray, indices: np.ndarray, y_band: tuple[float, float]):
    """Selects the faces whose 3 vertices all fall within `y_band` (as a
    fraction of the body's total height), returns a compacted
    (sub_positions, sub_indices, vertex_map) where `vertex_map[i]` is the
    original body-vertex index sub-vertex `i` came from (so morph deltas and
    later per-avatar weights can be applied identically to the body)."""
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    span = y_max - y_min
    lo = y_min + y_band[0] * span
    hi = y_min + y_band[1] * span

    in_band = (positions[:, 1] >= lo) & (positions[:, 1] <= hi)
    face_mask = in_band[indices].all(axis=1)
    faces = indices[face_mask]
    if len(faces) == 0:
        raise ValueError(f"y_band {y_band} selected no faces from the base mesh")

    used_verts = np.unique(faces)
    remap = np.full(positions.shape[0], -1, dtype=np.int64)
    remap[used_verts] = np.arange(len(used_verts))

    sub_positions = positions[used_verts]
    sub_indices = remap[faces].astype(np.uint32)
    return sub_positions, sub_indices, used_verts


def _push_outward(positions: np.ndarray, amount: float) -> np.ndarray:
    """Offsets each vertex away from the mesh's central vertical (Y) axis by
    `amount` metres, so the garment sits just outside the body's skin instead
    of z-fighting with it."""
    centroid_xz = positions[:, [0, 2]].mean(axis=0)
    radial = positions[:, [0, 2]] - centroid_xz
    dist = np.linalg.norm(radial, axis=1, keepdims=True)
    dist_safe = np.where(dist < 1e-6, 1.0, dist)
    direction = radial / dist_safe
    out = positions.copy()
    out[:, [0, 2]] += direction * amount
    return out


def _compute_cylindrical_uvs(positions: np.ndarray) -> np.ndarray:
    """Generates cylindrical UV coordinates for a garment sub-mesh.

    The sub-mesh is a vertical band of the body — wrapping a cylindrical
    projection around the Y-axis gives natural, seamless UVs for a
    front/back garment texture:
      U = atan2(z, x) normalised to [0, 1] (0 = front-center, 0.5 = back)
      V = (y - y_min) / (y_max - y_min), 0 = bottom hem, 1 = collar
    """
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    u = (np.arctan2(z, x) / (2 * np.pi)) % 1.0
    y_min, y_max = y.min(), y.max()
    v = (y - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y)
    return np.stack([u, v], axis=1).astype(np.float32)


def _write_standalone_glb(
    positions: np.ndarray,
    indices: np.ndarray,
    rgb: tuple[int, int, int],
    uvs: np.ndarray | None = None,
) -> bytes:
    """Builds a minimal single-mesh GLB with optional UV attribute (TEXCOORD_0).

    Uses a flat `material.baseColorFactor` when `uvs` is None, so the
    fallback is unchanged for garments that don't carry a texture.
    When `uvs` is provided the material still uses the flat base-color tint
    (the actual texture is loaded out-of-band by AvatarViewer3D's
    `applyGarmentTexture`, matching the same pattern used for the face
    texture) but TEXCOORD_0 is embedded so three.js MeshBasicMaterial.map
    can sample it correctly.
    """
    blob = bytearray()

    def _append(data: bytes, alignment: int = 4) -> int:
        pad = (-len(data)) % alignment
        offset = len(blob)
        blob.extend(data + b"\x00" * pad)
        return offset

    pos_bytes = positions.astype(np.float32).tobytes()
    pos_offset = _append(pos_bytes)
    pos_bv = BufferView(buffer=0, byteOffset=pos_offset, byteLength=len(pos_bytes))

    idx_flat = indices.astype(np.uint32).reshape(-1)
    idx_bytes = idx_flat.tobytes()
    idx_offset = _append(idx_bytes)
    idx_bv = BufferView(buffer=0, byteOffset=idx_offset, byteLength=len(idx_bytes))

    buffer_views = [pos_bv, idx_bv]
    accessors_list = [
        Accessor(
            bufferView=0,
            componentType=_POSITION_COMPONENT_TYPE,
            count=len(positions),
            type=_VEC3,
            min=positions.min(axis=0).tolist(),
            max=positions.max(axis=0).tolist(),
        ),
        Accessor(
            bufferView=1,
            componentType=5125,
            count=len(idx_flat),
            type=_SCALAR,
        ),
    ]

    prim_attributes: dict = {"POSITION": 0}

    if uvs is not None:
        uv_bytes = uvs.astype(np.float32).tobytes()
        uv_offset = _append(uv_bytes)
        uv_bv = BufferView(buffer=0, byteOffset=uv_offset, byteLength=len(uv_bytes))
        buffer_views.append(uv_bv)
        accessors_list.append(
            Accessor(
                bufferView=len(buffer_views) - 1,
                componentType=_TEX_COORD_TYPE,
                count=len(uvs),
                type=_VEC2,
            )
        )
        prim_attributes["TEXCOORD_0"] = len(accessors_list) - 1

    base_color_factor = [c / 255.0 for c in rgb] + [1.0]

    gltf = GLTF2(
        asset=Asset(version="2.0"),
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[Node(mesh=0)],
        meshes=[Mesh(primitives=[Primitive(
            attributes=prim_attributes,
            indices=1,
            material=0,
        )])],
        materials=[Material(
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorFactor=base_color_factor,
                metallicFactor=0.0,
                roughnessFactor=0.85,
            ),
            doubleSided=True,
        )],
        buffers=[Buffer(byteLength=len(blob))],
        bufferViews=buffer_views,
        accessors=accessors_list,
    )
    gltf.set_binary_blob(bytes(blob))
    return b"".join(gltf.save_to_bytes())


def build_garment_glb(garment_id: str, body3d_params: dict,
                      face_shape: str, gender: str = "female") -> bytes | None:
    """Builds a garment GLB shaped to match one avatar. Applies the *same*
    morph weights + height scale `makehuman_mesh.build_personalized_glb` uses
    for the body (including the same fixed per-gender height, see
    `_HEIGHT_CM_BY_GENDER`), so the garment tracks the avatar's body shape
    and height exactly. UV coordinates (TEXCOORD_0, cylindrical projection)
    are embedded so `AvatarViewer3D` can apply a garment texture image.
    Returns None if `garment_id` isn't in the catalog.
    """
    garment = get_garment(garment_id)
    if garment is None:
        return None

    height_cm = _HEIGHT_CM_BY_GENDER.get(gender, _HEIGHT_CM_BY_GENDER["neutral"])
    asset_gender = _ASSET_GENDER.get(gender, "female")
    base_positions, base_indices, deltas = _load_base_full(asset_gender)
    sub_positions, sub_indices, vertex_map = _cut_band(base_positions, base_indices, garment.y_band)

    weights = compute_morph_weights(body3d_params, face_shape)

    # Deform the *full* body (not just the cut band) with the same weights
    # `makehuman_mesh.build_personalized_glb` uses, so the body-height-derived
    # scale factor below matches the body's exactly.
    full_positions = base_positions.copy()
    for name in MORPH_TARGET_NAMES:
        weight = weights.get(name, 0.0)
        if weight:
            full_positions += deltas[name] * weight

    positions = sub_positions.copy()
    for name in MORPH_TARGET_NAMES:
        weight = weights.get(name, 0.0)
        if weight:
            positions += deltas[name][vertex_map] * weight

    base_height = float(full_positions[:, 1].max() - full_positions[:, 1].min())
    if base_height > 0:
        scale = (height_cm / 100.0) / base_height
        positions *= scale

    positions = _push_outward(positions, garment.offset)

    # Generate cylindrical UVs so a front/back garment photo can be mapped.
    uvs = _compute_cylindrical_uvs(positions)

    return _write_standalone_glb(positions, sub_indices, _hex_to_rgb(garment.color_hex), uvs=uvs)
