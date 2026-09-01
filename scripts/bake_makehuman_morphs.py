"""Diff staged MakeHuman GLB exports, remap UVs, and bake morph targets into
the final base GLBs.

Run with plain Python (not Blender), after:

    blender --background --python scripts/generate_makehuman_avatars.py
    python scripts/bake_makehuman_morphs.py

Reads ``.codex/generated_makehuman/_stage/{name}_base.glb`` plus one
``_stage/{name}_<morph>.glb`` per MORPH_TARGET_NAME, diffs each morphed
export's POSITION against the base export's POSITION, remaps the base
export's TEXCOORD_0 so the head spans the full 0-1 UV range (spherical
projection, matching generate_test_avatars.py's _spherical_uv) while the rest
of the body is pinned to a single texel - so a face-photo texture
(AvatarViewer3D.applyTint) lands on the head instead of being smeared across
the body - and writes ``.codex/generated_makehuman/{name}.glb`` with the 6
morph deltas baked in as named morph targets (mesh.extras.targetNames,
mesh.weights, primitives[0].targets), reusing
generate_test_avatars.py's _add_morph_targets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from pygltflib import GLTF2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_test_avatars import MORPH_TARGET_NAMES, _add_morph_targets  # noqa: E402

OUT_DIR = ROOT / ".codex" / "generated_makehuman"
STAGE_DIR = OUT_DIR / "_stage"

# Top fraction of the mesh's bounding-box height (glTF Y-up) treated as
# "head" by _remap_uv_for_face_overlay.
HEAD_HEIGHT_FRACTION = 0.13

# UV coordinate assigned to non-head vertices: a single texel near the corner
# of a face photo, so AvatarViewer3D.applyTint's face-texture path reads as a
# flat tint across the body (matches the placeholder asset's UV layout).
BODY_UV = (0.95, 0.95)


def _read_positions(path: Path) -> tuple[bytes, np.ndarray]:
    glb_bytes = path.read_bytes()
    gltf = GLTF2.load_from_bytes(glb_bytes)
    blob = gltf.binary_blob()

    prim = gltf.meshes[0].primitives[0]
    acc = gltf.accessors[prim.attributes.POSITION]
    bv = gltf.bufferViews[acc.bufferView]
    offset = bv.byteOffset + (acc.byteOffset or 0)
    count = acc.count
    stride = bv.byteStride or 12

    if stride == 12:
        data = blob[offset: offset + count * 12]
        positions = np.frombuffer(data, dtype=np.float32).reshape(count, 3).copy()
    else:
        positions = np.zeros((count, 3), dtype=np.float32)
        for i in range(count):
            o = offset + i * stride
            positions[i] = np.frombuffer(blob[o:o + 12], dtype=np.float32)

    return glb_bytes, positions


def _remap_uv_for_face_overlay(positions: np.ndarray) -> np.ndarray:
    """(N, 3) glTF Y-up positions -> (N, 2) UVs: spherical projection (front
    of head -> u=0.5) for the top HEAD_HEIGHT_FRACTION of the bounding box,
    BODY_UV for everything else."""
    y = positions[:, 1]
    head_threshold = y.max() - HEAD_HEIGHT_FRACTION * (y.max() - y.min())
    head_mask = y > head_threshold

    center = positions[head_mask].mean(axis=0)
    rel = positions - center
    norm = rel / np.linalg.norm(rel, axis=1, keepdims=True).clip(min=1e-9)

    uv = np.tile(np.array(BODY_UV, dtype=np.float32), (len(positions), 1))
    uv[head_mask, 0] = 0.5 + np.arctan2(norm[head_mask, 0], norm[head_mask, 2]) / (2 * np.pi)
    uv[head_mask, 1] = 0.5 - np.arcsin(np.clip(norm[head_mask, 1], -1, 1)) / np.pi
    return uv


def _set_uv0(glb_bytes: bytes, uv: np.ndarray) -> bytes:
    gltf = GLTF2.load_from_bytes(glb_bytes)
    blob = bytearray(gltf.binary_blob() or b"")

    prim = gltf.meshes[0].primitives[0]
    acc = gltf.accessors[prim.attributes.TEXCOORD_0]
    bv = gltf.bufferViews[acc.bufferView]
    offset = bv.byteOffset + (acc.byteOffset or 0)

    data = uv.astype(np.float32).tobytes()
    if len(data) != acc.count * 8:
        raise ValueError(f"UV array size mismatch: {len(data)} != {acc.count * 8}")
    blob[offset:offset + len(data)] = data

    if acc.min is not None:
        acc.min = [float(uv[:, 0].min()), float(uv[:, 1].min())]
    if acc.max is not None:
        acc.max = [float(uv[:, 0].max()), float(uv[:, 1].max())]

    gltf.set_binary_blob(bytes(blob))
    return b"".join(gltf.save_to_bytes())


def bake(name: str) -> Path:
    base_bytes, base_positions = _read_positions(STAGE_DIR / f"{name}_base.glb")

    deltas = {}
    for morph_name in MORPH_TARGET_NAMES:
        _, morph_positions = _read_positions(STAGE_DIR / f"{name}_{morph_name}.glb")
        if morph_positions.shape != base_positions.shape:
            raise ValueError(
                f"{name}_{morph_name}.glb: vertex count {morph_positions.shape[0]} "
                f"!= base {base_positions.shape[0]}"
            )
        deltas[morph_name] = morph_positions - base_positions

    uv = _remap_uv_for_face_overlay(base_positions)
    base_bytes = _set_uv0(base_bytes, uv)

    glb_bytes = _add_morph_targets(base_bytes, deltas)
    out_path = OUT_DIR / f"{name}.glb"
    out_path.write_bytes(glb_bytes)
    return out_path


# Matches model1_body_shape's CLASS_NAMES / generate_makehuman_avatars.py's
# BODY_SHAPE_PRESETS keys exactly.
SHAPE_NAMES = ("Hourglass", "Pear", "Apple", "Rectangle", "InvertedTriangle")


def main() -> None:
    # Mirrors generate_makehuman_avatars.py's MAKEHUMAN_GENDERS so the two
    # scripts can be run over the same subset while iterating on presets.
    genders = os.environ.get("MAKEHUMAN_GENDERS", "male,female").split(",")
    for gender in genders:
        for shape in SHAPE_NAMES:
            out_path = bake(f"{gender}_{shape}")
            print(f"BAKED {out_path}")


if __name__ == "__main__":
    main()
