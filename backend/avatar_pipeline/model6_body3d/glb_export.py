"""
Model 6 — 3D Body Reconstruction: minimal glTF 2.0 binary (.glb) writer.

`mesh_to_glb_bytes` packages a `mesh_builder.build_avatar_mesh(...)` result
— a list of mesh "parts" (each with positions, normals, optional UVs,
triangle indices, and a flat color or texture-indexed material) plus a list
of PNG texture images — into a self-contained `.glb` blob. No external
dependencies (no `trimesh` / `pygltflib`), since this needs to run in
`server.mock_pipeline` without TensorFlow or any extra packages.

A `.glb` file is: a 12-byte header, a JSON chunk (the glTF scene/mesh/
material/accessor/image descriptions), and a binary chunk (the raw vertex/
index/image buffers the JSON chunk's accessors and images point into). See
the glTF 2.0 spec, "Binary glTF Layout".

Each "part" becomes one mesh primitive + one material:
    {
        "vertices": (N, 3) float32, "faces": (M, 3) uint32,
        "normals": (N, 3) float32, "uvs": (N, 2) float32 or None,
        "material": {
            "name": str,
            "base_color_rgb": (r, g, b) in 0-255,
            "texture_index": int or None,  # index into mesh["images"]
        },
    }
Each entry in `mesh["images"]` is raw PNG bytes, embedded as a
`bufferView` + `image/png` glTF image and referenced by a texture.
"""

import json
import struct

import numpy as np

_GLB_MAGIC = 0x46546C67  # "glTF"
_GLB_VERSION = 2
_CHUNK_TYPE_JSON = 0x4E4F534A  # "JSON"
_CHUNK_TYPE_BIN = 0x004E4942  # "BIN\0"

_COMPONENT_TYPE_FLOAT = 5126
_COMPONENT_TYPE_UNSIGNED_INT = 5125
_TARGET_ARRAY_BUFFER = 34962
_TARGET_ELEMENT_ARRAY_BUFFER = 34963


def _pad(data, pad_byte):
    padding = (-len(data)) % 4
    return data + pad_byte * padding


def mesh_to_glb_bytes(mesh):
    """mesh: a `mesh_builder.build_avatar_mesh(...)` dict (`{"parts": [...],
    "images": [...]}`). Returns `bytes` of a complete `.glb` file: one mesh
    with one primitive per part, one material per part (flat color or
    `baseColorTexture`), and one image/texture per entry in `mesh["images"]`."""
    buffer_chunks = []
    buffer_views = []
    accessors = []

    def add_buffer_view(data, target=None):
        view = {"buffer": 0, "byteOffset": sum(len(c) for c in buffer_chunks), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        buffer_chunks.append(_pad(data, b"\x00"))
        return len(buffer_views) - 1

    def add_accessor(view_index, component_type, count, type_, **extra):
        accessor = {"bufferView": view_index, "componentType": component_type, "count": count, "type": type_}
        accessor.update(extra)
        accessors.append(accessor)
        return len(accessors) - 1

    images = []
    textures = []
    for png_bytes in mesh.get("images", []):
        view_index = add_buffer_view(png_bytes)
        images.append({"mimeType": "image/png", "bufferView": view_index})
        textures.append({"source": len(images) - 1})

    materials = []
    primitives = []
    for part in mesh["parts"]:
        vertices = np.asarray(part["vertices"], dtype="<f4")
        normals = np.asarray(part["normals"], dtype="<f4")
        indices = np.asarray(part["faces"], dtype="<u4").reshape(-1)

        pos_view = add_buffer_view(vertices.tobytes(), target=_TARGET_ARRAY_BUFFER)
        pos_acc = add_accessor(
            pos_view, _COMPONENT_TYPE_FLOAT, len(vertices), "VEC3",
            min=vertices.min(axis=0).tolist(), max=vertices.max(axis=0).tolist(),
        )

        norm_view = add_buffer_view(normals.tobytes(), target=_TARGET_ARRAY_BUFFER)
        norm_acc = add_accessor(norm_view, _COMPONENT_TYPE_FLOAT, len(normals), "VEC3")

        attributes = {"POSITION": pos_acc, "NORMAL": norm_acc}

        uvs = part.get("uvs")
        if uvs is not None:
            uvs = np.asarray(uvs, dtype="<f4")
            uv_view = add_buffer_view(uvs.tobytes(), target=_TARGET_ARRAY_BUFFER)
            attributes["TEXCOORD_0"] = add_accessor(uv_view, _COMPONENT_TYPE_FLOAT, len(uvs), "VEC2")

        idx_view = add_buffer_view(indices.tobytes(), target=_TARGET_ELEMENT_ARRAY_BUFFER)
        idx_acc = add_accessor(idx_view, _COMPONENT_TYPE_UNSIGNED_INT, len(indices), "SCALAR")

        material = part["material"]
        texture_index = material.get("texture_index")
        # When a texture is present, use a white baseColorFactor so the
        # texture shows through at full strength.  For flat-colour materials
        # use the specified rgb.
        if texture_index is not None:
            base_color_factor = [1.0, 1.0, 1.0, 1.0]
        else:
            color = np.clip(np.asarray(material["base_color_rgb"], dtype=np.float32) / 255.0, 0.0, 1.0)
            base_color_factor = [float(color[0]), float(color[1]), float(color[2]), 1.0]
        material_json = {
            "name": material.get("name", "material"),
            "pbrMetallicRoughness": {
                "baseColorFactor": base_color_factor,
                "metallicFactor": 0.0,
                "roughnessFactor": 0.85,
            },
            "doubleSided": True,
        }
        if texture_index is not None:
            material_json["pbrMetallicRoughness"]["baseColorTexture"] = {"index": texture_index}
        materials.append(material_json)

        primitives.append({
            "attributes": attributes,
            "indices": idx_acc,
            "material": len(materials) - 1,
            "mode": 4,  # TRIANGLES
        })

    buffer_bytes = b"".join(buffer_chunks)

    gltf = {
        "asset": {"version": "2.0", "generator": "eWardrobe avatar_pipeline model6_body3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "avatar"}],
        "meshes": [{"name": "avatar_mesh", "primitives": primitives}],
        "materials": materials,
        "buffers": [{"byteLength": len(buffer_bytes)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    if images:
        gltf["images"] = images
        gltf["textures"] = textures

    json_chunk = _pad(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_chunk = _pad(buffer_bytes, b"\x00")

    total_length = 12 + (8 + len(json_chunk)) + (8 + len(bin_chunk))

    out = struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total_length)
    out += struct.pack("<II", len(json_chunk), _CHUNK_TYPE_JSON) + json_chunk
    out += struct.pack("<II", len(bin_chunk), _CHUNK_TYPE_BIN) + bin_chunk
    return out
