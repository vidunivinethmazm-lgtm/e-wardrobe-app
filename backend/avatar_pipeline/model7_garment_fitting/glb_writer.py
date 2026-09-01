"""Model 7 — shared single-mesh GLB writer + mesh-quality validation.

Extracted out of `garment_fit_runner.py` (where it started as a private
helper for the two `GarmentFitRunner` backends) so the EXPERIMENTAL
`multiview/avatar3d_providers.py` full-avatar path can reuse the exact same
GLB-writing/validation code instead of duplicating it — neither knows or
cares whether the mesh being written is a fitted garment or a full
reconstructed avatar, they're both just (vertices, faces, uvs, texture_png).
"""

from __future__ import annotations

import numpy as np
from pygltflib import (
    GLTF2, Accessor, Asset, Buffer, BufferView, Image as GLTFImage,
    Material, Mesh, Node, PbrMetallicRoughness, Primitive, Sampler, Scene,
    Texture, TextureInfo,
)

from .fitting_types import GarmentFittingError

MIN_VERTICES = 8
MIN_FACES = 4
MIN_BOUNDING_EXTENT = 1e-4


def validate_mesh_geometry(vertices: np.ndarray, faces: np.ndarray) -> None:
    """Raises `GarmentFittingError` if the mesh is too degenerate to be
    usable (empty, near-zero extent, or faces referencing out-of-range
    vertices) — the numeric half of `garment_mesh_generation.
    validate_garment_mesh`'s checks, factored out so callers that don't
    have a `GeneratedGarmentMesh` (e.g. a full reconstructed avatar mesh,
    which has no landmark contract) can still run the same quality gate.
    """
    if len(vertices) < MIN_VERTICES:
        raise GarmentFittingError(f"generated mesh has too few vertices ({len(vertices)}) to be usable")
    if len(faces) < MIN_FACES:
        raise GarmentFittingError(f"generated mesh has too few faces ({len(faces)}) to be usable")
    if faces.size and (faces.max() >= len(vertices) or faces.min() < 0):
        raise GarmentFittingError("generated mesh has faces referencing invalid vertex indices")

    extent = vertices.max(axis=0) - vertices.min(axis=0)
    if float(np.max(extent)) < MIN_BOUNDING_EXTENT:
        raise GarmentFittingError("generated mesh is degenerate (near-zero bounding extent)")
    if not np.all(np.isfinite(vertices)):
        raise GarmentFittingError("generated mesh contains non-finite vertex coordinates")


def write_mesh_glb(
    positions: np.ndarray, indices: np.ndarray, uvs: np.ndarray | None, texture_png: bytes | None,
) -> bytes:
    """Minimal single-mesh GLB writer preserving the mesh's own UVs/texture
    (never a flat placeholder tint when a real texture is available)."""
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
    accessors = [
        Accessor(bufferView=0, componentType=5126, count=len(positions), type="VEC3",
                 min=positions.min(axis=0).tolist(), max=positions.max(axis=0).tolist()),
        Accessor(bufferView=1, componentType=5125, count=len(idx_flat), type="SCALAR"),
    ]
    attributes = {"POSITION": 0}

    images, textures, samplers = [], [], []
    base_color_factor = [1.0, 1.0, 1.0, 1.0]
    base_color_texture = None

    if uvs is not None and len(uvs) == len(positions) and texture_png is not None:
        uv_bytes = uvs.astype(np.float32).tobytes()
        uv_offset = _append(uv_bytes)
        buffer_views.append(BufferView(buffer=0, byteOffset=uv_offset, byteLength=len(uv_bytes)))
        accessors.append(Accessor(
            bufferView=len(buffer_views) - 1, componentType=5126, count=len(uvs), type="VEC2",
            min=uvs.min(axis=0).tolist(), max=uvs.max(axis=0).tolist(),
        ))
        attributes["TEXCOORD_0"] = len(accessors) - 1

        img_offset = _append(texture_png)
        buffer_views.append(BufferView(buffer=0, byteOffset=img_offset, byteLength=len(texture_png)))
        images.append(GLTFImage(bufferView=len(buffer_views) - 1, mimeType="image/png"))
        samplers.append(Sampler(magFilter=9729, minFilter=9729))
        textures.append(Texture(source=0, sampler=0))
        base_color_texture = TextureInfo(index=0)
    else:
        base_color_factor = [0.85, 0.85, 0.85, 1.0]  # no texture available — neutral gray, not claimed as final

    gltf = GLTF2(
        asset=Asset(version="2.0"),
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[Node(mesh=0)],
        meshes=[Mesh(primitives=[Primitive(attributes=attributes, indices=1, material=0)])],
        materials=[Material(
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorFactor=base_color_factor, baseColorTexture=base_color_texture,
                metallicFactor=0.0, roughnessFactor=0.85,
            ),
            doubleSided=True,
        )],
        buffers=[Buffer(byteLength=len(blob))],
        bufferViews=buffer_views,
        accessors=accessors,
        images=images or None,
        textures=textures or None,
        samplers=samplers or None,
    )
    gltf.set_binary_blob(bytes(blob))
    return b"".join(gltf.save_to_bytes())
