"""
Tests for Model 6 (3D Body Reconstruction): face/hair extraction
(`face_features.extract_face_features`), the procedural mesh builder
(`mesh_builder.build_avatar_mesh`), the `.glb` writer (`glb_export.
mesh_to_glb_bytes`), and their use in `server.mock_pipeline.build_avatar`
and `pipeline_types.save_avatar_result` / `load_avatar_result`.

Runs without TensorFlow (mock pipeline only) — see avatar_pipeline/README.md
and the root README.md for mock vs. real mode.
"""

import io
import struct

import numpy as np
import pytest
from PIL import Image

from backend.avatar_pipeline.model6_body3d.face_features import DEFAULT_HAIR_RGB, extract_face_features
from backend.avatar_pipeline.model6_body3d.glb_export import mesh_to_glb_bytes
from backend.avatar_pipeline.model6_body3d.mesh_builder import build_avatar_mesh
from backend.avatar_pipeline.model6_body3d.params import (
    PARAM_NAMES,
    default_params_from_measurements,
)

_GLB_MAGIC = 0x46546C67
_GLB_VERSION = 2
_CHUNK_TYPE_JSON = 0x4E4F534A
_CHUNK_TYPE_BIN = 0x004E4942

HEIGHT_CM = 165.0


def _parse_glb(data):
    """Parses a `.glb` blob into (header, gltf_json_dict, bin_chunk_bytes)."""
    magic, version, total_length = struct.unpack_from("<III", data, 0)
    assert magic == _GLB_MAGIC
    assert version == _GLB_VERSION
    assert total_length == len(data)

    offset = 12
    json_len, json_type = struct.unpack_from("<II", data, offset)
    assert json_type == _CHUNK_TYPE_JSON
    offset += 8
    json_chunk = data[offset:offset + json_len]
    offset += json_len

    bin_len, bin_type = struct.unpack_from("<II", data, offset)
    assert bin_type == _CHUNK_TYPE_BIN
    offset += 8
    bin_chunk = data[offset:offset + bin_len]
    offset += bin_len

    assert offset == len(data)

    import json
    gltf = json.loads(json_chunk.decode("utf-8"))
    return gltf, bin_chunk


@pytest.fixture
def sample_params():
    return default_params_from_measurements("Hourglass", bust=92, waist=70, hips=98, height=HEIGHT_CM)


@pytest.fixture
def sample_mesh(sample_params):
    return build_avatar_mesh(sample_params, height_cm=HEIGHT_CM, skin_rgb=(215, 189, 150))


def test_extract_face_features_no_face_fallback():
    photo = np.full((96, 96, 3), 200, dtype=np.uint8)
    result = extract_face_features(photo)

    assert result["face_crop"] is None
    assert result["hair_rgb"] == DEFAULT_HAIR_RGB


def test_build_avatar_mesh_parts(sample_mesh, sample_params):
    parts = sample_mesh["parts"]
    part_names = [p["material"]["name"] for p in parts]
    assert part_names == ["skin", "face", "hair", "eye_white", "eye_iris", "mouth"]

    H = HEIGHT_CM / 100.0
    head_radius = sample_params["head_radius"] / 2.0 * H
    max_y = H + 0.1 * head_radius  # the hair cap pokes slightly above head_top

    all_vertices = []
    for part in parts:
        vertices, faces, normals = part["vertices"], part["faces"], part["normals"]

        assert vertices.dtype == np.float32
        assert normals.dtype == np.float32
        assert faces.dtype == np.uint32

        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert normals.shape == vertices.shape
        assert faces.ndim == 2 and faces.shape[1] == 3
        assert faces.max() < len(vertices)

        lengths = np.linalg.norm(normals, axis=1)
        assert np.all((np.isclose(lengths, 1.0, atol=1e-4)) | (lengths == 0))

        if part["material"]["name"] == "face":
            uvs = part["uvs"]
            assert uvs.dtype == np.float32
            assert uvs.shape == (vertices.shape[0], 2)
            assert uvs.min() >= 0.0 and uvs.max() <= 1.0
        else:
            assert part["uvs"] is None

        all_vertices.append(vertices)

    combined = np.concatenate(all_vertices, axis=0)
    assert combined[:, 1].min() >= -1e-3
    assert combined[:, 1].max() <= max_y + 1e-3


def test_build_avatar_mesh_head_texture(sample_mesh):
    images = sample_mesh["images"]
    assert len(images) == 1

    image = Image.open(io.BytesIO(images[0]))
    assert image.format == "PNG"
    assert image.size == (256, 256)


def test_mesh_to_glb_bytes_header_and_structure(sample_mesh):
    data = mesh_to_glb_bytes(sample_mesh)

    assert isinstance(data, bytes)
    assert data[:4] == b"glTF"
    # Both chunks (and the header) must be 4-byte aligned per the glTF spec.
    assert len(data) % 4 == 0

    gltf, bin_chunk = _parse_glb(data)

    assert gltf["asset"]["version"] == "2.0"
    assert gltf["scenes"][0]["nodes"] == [0]
    assert gltf["nodes"][0]["mesh"] == 0

    parts = sample_mesh["parts"]
    primitives = gltf["meshes"][0]["primitives"]
    assert len(primitives) == len(parts)
    assert len(gltf["materials"]) == len(parts)

    # One embedded image/texture: the head's face texture.
    assert len(gltf["images"]) == 1
    assert gltf["images"][0]["mimeType"] == "image/png"
    assert len(gltf["textures"]) == 1
    assert gltf["textures"][0]["source"] == 0

    for part, primitive, material in zip(parts, primitives, gltf["materials"]):
        assert "POSITION" in primitive["attributes"]
        assert "NORMAL" in primitive["attributes"]
        assert primitive["mode"] == 4  # TRIANGLES
        assert material["doubleSided"] is True

        if part["material"]["name"] == "face":
            assert "TEXCOORD_0" in primitive["attributes"]
            assert material["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
        else:
            assert "TEXCOORD_0" not in primitive["attributes"]
            expected_rgb = np.asarray(part["material"]["base_color_rgb"], dtype=np.float32) / 255.0
            assert material["pbrMetallicRoughness"]["baseColorFactor"][:3] == pytest.approx(
                expected_rgb.tolist(), abs=1e-6
            )

        # Round-trip this primitive's position/index data back out of the BIN chunk.
        pos_accessor = gltf["accessors"][primitive["attributes"]["POSITION"]]
        idx_accessor = gltf["accessors"][primitive["indices"]]

        pos_view = gltf["bufferViews"][pos_accessor["bufferView"]]
        idx_view = gltf["bufferViews"][idx_accessor["bufferView"]]

        positions = np.frombuffer(
            bin_chunk[pos_view["byteOffset"]:pos_view["byteOffset"] + pos_view["byteLength"]], dtype="<f4"
        ).reshape(-1, 3)
        np.testing.assert_allclose(positions, part["vertices"])

        indices = np.frombuffer(
            bin_chunk[idx_view["byteOffset"]:idx_view["byteOffset"] + idx_view["byteLength"]], dtype="<u4"
        )
        np.testing.assert_array_equal(indices, part["faces"].reshape(-1))

    # The embedded PNG bytes round-trip too.
    image_view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
    image_bytes = bin_chunk[image_view["byteOffset"]:image_view["byteOffset"] + image_view["byteLength"]]
    assert image_bytes == sample_mesh["images"][0]

    buffer = gltf["buffers"][0]
    assert buffer["byteLength"] == len(bin_chunk)


def test_mock_pipeline_build_avatar_includes_body3d():
    from backend.mock_pipeline import build_avatar

    photo = np.full((96, 96, 3), 200, dtype=np.uint8)
    result = build_avatar(photo, bust=92, waist=70, hips=98, height=165)

    assert isinstance(result.avatar_mesh_glb, bytes)
    assert result.avatar_mesh_glb[:4] == b"glTF"
    _parse_glb(result.avatar_mesh_glb)  # re-validate full structure

    assert set(result.body3d_params.keys()) == set(PARAM_NAMES)
    assert all(isinstance(v, float) for v in result.body3d_params.values())


def test_save_and_load_avatar_result_round_trip(tmp_path):
    from backend.mock_pipeline import build_avatar
    from backend.avatar_pipeline.pipeline_types import load_avatar_result, save_avatar_result

    photo = np.full((96, 96, 3), 200, dtype=np.uint8)
    result = build_avatar(photo, bust=92, waist=70, hips=98, height=165)

    save_avatar_result(result, tmp_path)
    loaded = load_avatar_result(tmp_path)

    assert loaded.avatar_mesh_glb == result.avatar_mesh_glb
    assert loaded.body3d_params == result.body3d_params
    assert loaded.body_shape == result.body_shape
