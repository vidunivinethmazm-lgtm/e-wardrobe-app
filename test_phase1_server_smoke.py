"""
Server smoke test for Phase 1 (default, AVATAR_USE_REALISTIC unset).

Confirms the procedural pipeline still works unchanged after the Phase 2
avatar_builder.py edits.
"""

import io
import os

os.environ.pop("AVATAR_USE_REALISTIC", None)
os.environ.setdefault("AVATAR_PIPELINE_MOCK", "1")

from server.app import app


def main():
    client = app.test_client()

    photo_path = os.path.join(
        os.path.dirname(__file__), "data", "model1_body_shape", "images", "00000.png"
    )
    with open(photo_path, "rb") as f:
        photo_bytes = f.read()

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

    assert response.status_code == 200, f"Unexpected status {response.status_code}: {response.get_data(as_text=True)}"
    body = response.get_json()
    avatar_id = body["avatar_id"]
    print(f"Created avatar_id={avatar_id}, gender={body.get('gender')}")

    mesh_response = client.get(f"/api/avatars/{avatar_id}/mesh.glb")
    assert mesh_response.status_code == 200, f"mesh.glb returned {mesh_response.status_code}"
    print(f"mesh.glb: {len(mesh_response.data)} bytes")

    print("\n=== Phase 1 smoke test passed ===")


if __name__ == "__main__":
    main()
