"""
End-to-end test for the Phase 2 avatar builder fixes.

Builds a realistic avatar with synthetic inputs (distinct skin/hair colors and
a synthetic face crop), writes it to out_avatar.glb, then reloads it with both
trimesh and pygltflib to assert that:

- the body material's baseColorFactor reflects the requested skin_rgb
  (not a placeholder, and not driven near-black by a texture/factor
  interaction bug)
- the head material has both a baseColorTexture (face texture) and a
  normalTexture (procedural skin normal map)
- the mesh has TEXCOORD_0
- there are 2 geometries (body + hair) and the hair geometry's material
  reflects the requested hair_rgb
"""

import io
import os

import numpy as np
import trimesh
from pygltflib import GLTF2

from backend.avatar_pipeline.model6_body3d.avatar_builder import create_avatar_builder


def main():
    builder = create_avatar_builder(use_realistic=True)
    assert builder is not None, "Failed to create AvatarBuilder (assets missing?)"

    skin_rgb = np.array([180.0, 120.0, 90.0])
    hair_rgb = (20, 20, 200)

    # Synthetic 128x128 RGB face crop (distinct from skin_rgb so the
    # composited face texture is visually identifiable).
    face_crop = np.zeros((128, 128, 3), dtype=np.uint8)
    face_crop[:, :, 0] = 255  # solid red

    body3d_params = {
        "shoulder_width": 0.30,
        "hip_width": 0.20,
    }

    glb = builder.build_realistic_avatar(
        gender="female",
        facial_analysis={"hair_style": "wavy"},
        body3d_params=body3d_params,
        height=165,
        skin_rgb=skin_rgb,
        hair_rgb=hair_rgb,
        face_crop=face_crop,
    )

    assert glb is not None, "build_realistic_avatar returned None"

    out_path = os.path.join(os.path.dirname(__file__), "out_avatar.glb")
    with open(out_path, "wb") as f:
        f.write(glb)
    print(f"Wrote {out_path} ({len(glb)} bytes)")

    # --- pygltflib checks ---
    gltf = GLTF2.load_from_bytes(glb)

    assert len(gltf.materials) >= 1, "Expected at least one material"
    body_mat = gltf.materials[0]
    bcf = body_mat.pbrMetallicRoughness.baseColorFactor
    expected_norm = [c / 255.0 for c in skin_rgb]
    for actual, expected in zip(bcf[:3], expected_norm):
        assert abs(actual - expected) < 1e-3, f"baseColorFactor {bcf} != skin_rgb {expected_norm}"
    print(f"OK: body baseColorFactor {bcf} matches requested skin_rgb")

    assert body_mat.pbrMetallicRoughness.baseColorTexture is not None, "Missing baseColorTexture"
    assert body_mat.normalTexture is not None, "Missing normalTexture"
    print("OK: head material has baseColorTexture and normalTexture")

    # --- trimesh checks ---
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    assert isinstance(scene, trimesh.Scene), f"Expected Scene, got {type(scene)}"

    geometries = list(scene.geometry.values())
    print(f"Geometries: {len(geometries)}")
    assert len(geometries) == 2, f"Expected 2 geometries (body + hair), got {len(geometries)}"

    body_geom, hair_geom = geometries
    assert hasattr(body_geom.visual, "uv") and body_geom.visual.uv is not None, "Body mesh missing UVs"
    print(f"OK: body mesh has UVs, shape {body_geom.visual.uv.shape}")

    hair_material = hair_geom.visual.material
    hair_color = getattr(hair_material, "baseColorFactor", None)
    print(f"Hair material baseColorFactor: {hair_color}")
    # trimesh's PBRMaterial.baseColorFactor accessor returns uint8 (0-255),
    # regardless of the normalized floats passed in at construction time.
    for actual, expected in zip(hair_color[:3], hair_rgb):
        assert abs(int(actual) - expected) <= 1, f"hair baseColorFactor {hair_color} != hair_rgb {hair_rgb}"
    print("OK: hair material reflects requested hair_rgb")

    print("\n=== All e2e checks passed ===")


if __name__ == "__main__":
    main()
