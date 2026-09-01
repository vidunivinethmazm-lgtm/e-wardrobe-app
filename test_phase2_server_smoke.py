"""
Server smoke test for Phase 2 (AVATAR_USE_REALISTIC=1).

Posts a real photo to POST /api/avatars via the Flask test client and checks
that no "Warning: Failed to ..." / "falling back to procedural" messages are
printed, and that the returned mesh GLB has the expected Phase 2 materials.
"""

import io
import os

os.environ["AVATAR_USE_REALISTIC"] = "1"
os.environ.setdefault("AVATAR_PIPELINE_MOCK", "1")

import contextlib

import numpy as np
from PIL import Image
from pygltflib import GLTF2

from avatar_pipeline.model6_body3d.face_features import extract_face_features
from server.app import app


def main():
    client = app.test_client()

    photo_path = os.path.join(
        os.path.dirname(__file__), "data", "model1_body_shape", "images", "00000.png"
    )
    with open(photo_path, "rb") as f:
        photo_bytes = f.read()

    photo_rgb = np.array(Image.open(io.BytesIO(photo_bytes)).convert("RGB"))
    has_face = extract_face_features(photo_rgb)["face_crop"] is not None
    print(f"Test photo has detectable face: {has_face}")

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        response = client.post(
            "/api/avatars",
            data={
                "photo": (io.BytesIO(photo_bytes), "photo.png"),
                "bust": "90",
                "waist": "70",
                "hips": "95",
                "height": "165",
            },
            content_type="multipart/form-data",
        )

    output = captured.getvalue()
    print(output)

    assert response.status_code == 200, f"Unexpected status {response.status_code}: {response.get_data(as_text=True)}"
    body = response.get_json()
    avatar_id = body["avatar_id"]
    print(f"Created avatar_id={avatar_id}, gender={body.get('gender')}")

    for bad in ("Warning: Failed to", "falling back to procedural"):
        assert bad not in output, f"Found '{bad}' in server output:\n{output}"
    print("OK: no warnings/fallback in server output")

    mesh_response = client.get(f"/api/avatars/{avatar_id}/mesh.glb")
    assert mesh_response.status_code == 200, f"mesh.glb returned {mesh_response.status_code}"
    glb = mesh_response.data
    print(f"mesh.glb: {len(glb)} bytes")

    gltf = GLTF2.load_from_bytes(glb)
    assert len(gltf.materials) >= 1, "Expected at least one material"
    body_mat = gltf.materials[0]
    print(f"baseColorFactor: {body_mat.pbrMetallicRoughness.baseColorFactor}")
    assert body_mat.pbrMetallicRoughness.baseColorTexture is not None, "Missing baseColorTexture"
    if has_face:
        assert body_mat.normalTexture is not None, "Missing normalTexture"
        print("OK: mesh.glb has baseColorTexture and normalTexture")
    else:
        print("OK: mesh.glb has baseColorTexture (no face detected, normalTexture skipped as expected)")

    print("\n=== Server smoke test passed ===")


if __name__ == "__main__":
    main()
