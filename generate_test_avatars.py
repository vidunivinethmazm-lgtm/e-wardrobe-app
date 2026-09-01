"""
Generate placeholder humanoid GLB models for testing Phase 2.

Creates simple geometric humanoid models — with UV-mapped skin materials,
textures, and 6 morph targets (blendshapes) so the runtime pipeline
(avatar_builder.py, and the mobile on-device viewer) has something to
texture, recolor, and scale to a user's body measurements — that can be used
for testing before creating detailed models in Blender.

Coordinate system: centimeters, y-up, x-right, z-forward (+Z is "front"),
matching avatar_pipeline/model6_body3d/mesh_builder.py's convention. y=0 is
the feet, y=height is the top of the head.

Morph targets (see MORPH_TARGET_NAMES): each base model's mesh.primitives[0]
carries 6 named morph targets (mesh.extras.targetNames) whose weights are set
at runtime from the user's body measurements (mobile/src/services/bodyScaling.ts)
or body3d_params (avatar_pipeline/model6_body3d/params.py).

Usage:
    python generate_test_avatars.py

Output:
    assets/avatars/male/base.glb
    assets/avatars/female/base.glb
    assets/hair/{male,female}/*.glb
    mobile/assets/avatars/{male,female}.glb (copies, for the on-device viewer)
    mobile/assets/hair/{short,long,curly,buzz,ponytail}.glb (copies)
"""

import shutil
import zlib
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from pygltflib import GLTF2, Accessor, BufferView, FLOAT, VEC3


SKIN_TONE = (210, 180, 140)
HAIR_TONE = (139, 69, 19)

# Rotates a Z-axis-aligned primitive (trimesh's default for cylinder/capsule/
# revolve) so its long axis points along +Y ("up" in our convention).
_Z_TO_Y = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])

# Names of the 6 morph targets baked into every base humanoid model, in the
# order they're written to mesh.primitives[0].targets / mesh.extras.targetNames.
# Mirrored in mobile/src/types.ts's MorphTargetName.
MORPH_TARGET_NAMES = [
    "shoulderWidth", "hipWidth", "armLength", "legLength", "bodyType", "headWidth",
]

# Morph delta gain constants (first pass; tune visually in
# https://gltf-viewer.donmccurdy.com/ by loading a generated base.glb and
# dragging its morph-target sliders).
SHOULDER_TORSO_GAIN = 0.5
HIP_TORSO_GAIN = 0.5
BODY_TYPE_GAIN = 0.35
ARM_LENGTH_GAIN = 0.4
LEG_LENGTH_GAIN = 0.3
HEAD_WIDTH_GAIN = 1.0
SHOULDER_SHIFT_GAIN = 0.5
HIP_SHIFT_GAIN = 0.5


def _make_skin_material() -> trimesh.visual.material.PBRMaterial:
    """Build a shared PBR material with a flat skin-tone placeholder texture.

    The runtime avatar_builder replaces this texture with the user's face
    crop and recolors baseColorFactor to the detected skin tone, so the exact
    placeholder color here only matters for assets that are previewed
    directly (e.g. in a glTF viewer) before that happens.
    """
    texture = Image.new("RGB", (256, 256), SKIN_TONE)
    return trimesh.visual.material.PBRMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        baseColorTexture=texture,
        roughnessFactor=0.6,
        metallicFactor=0.0,
    )


def _spherical_uv(local_vertices: np.ndarray) -> np.ndarray:
    """Equirectangular UV from vertex directions about the local origin.

    +Z (front) maps to u=0.5, the top (+Y) maps to v=0, so a face texture
    placed at the center of the texture image lands on the front of the head.
    """
    norm = local_vertices / np.linalg.norm(local_vertices, axis=1, keepdims=True)
    u = 0.5 + np.arctan2(norm[:, 0], norm[:, 2]) / (2 * np.pi)
    v = 0.5 - np.arcsin(np.clip(norm[:, 1], -1, 1)) / np.pi
    return np.stack([u, v], axis=1)


def _flat_uv(n: int) -> np.ndarray:
    """Constant UV pointing at the texture's reserved flat-skin-tone corner."""
    return np.tile([0.95, 0.95], (n, 1))


def _to_y_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.apply_transform(_Z_TO_Y)
    return mesh


def _set_visual(mesh: trimesh.Trimesh, uv: np.ndarray, material) -> trimesh.Trimesh:
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return mesh


def _zero_deltas(n: int) -> dict:
    """A dict of {morph target name: (n, 3) zero displacement array}."""
    return {name: np.zeros((n, 3), dtype=np.float32) for name in MORPH_TARGET_NAMES}


def create_humanoid_model(gender: str, height: float = 170):
    """Create a humanoid mesh with a shared, UV-mapped skin material and 6
    morph-target displacement sets.

    Args:
        gender: 'male' or 'female'
        height: Height in cm

    Returns:
        (mesh, deltas) where `mesh` is a trimesh.Trimesh with TextureVisuals
        (UVs + a shared PBRMaterial), and `deltas` is a dict mapping each name
        in MORPH_TARGET_NAMES to an (N, 3) float32 array of per-vertex
        displacements (same vertex order/count as `mesh.vertices`) for that
        morph target at weight = 1.
    """
    scale = height / 170  # Normalize to 170cm reference rig
    material = _make_skin_material()
    parts = []
    deltas_list = []

    # --- Head: slightly non-spherical for a more head-like silhouette. ---
    head = trimesh.creation.icosphere(subdivisions=3, radius=8 * scale)
    head.apply_scale([0.92, 1.05, 0.95])
    head_uv = _spherical_uv(head.vertices)
    head_local = head.vertices.copy()  # Y-up, centered on head origin
    head.apply_translation([0, 155 * scale, 0])
    _set_visual(head, head_uv, material)

    head_deltas = _zero_deltas(len(head.vertices))
    head_deltas["headWidth"][:, 0] = head_local[:, 0] * HEAD_WIDTH_GAIN
    parts.append(head)
    deltas_list.append(head_deltas)

    # --- Neck: short connector closing the gap between the torso top (145)
    # and the head's base (~146.6), so the head doesn't float above the body.
    neck = trimesh.creation.cylinder(radius=4.5 * scale, height=5 * scale, sections=24)
    _to_y_up(neck)
    neck.apply_translation([0, 145.5 * scale, 0])
    _set_visual(neck, _flat_uv(len(neck.vertices)), material)
    parts.append(neck)
    deltas_list.append(_zero_deltas(len(neck.vertices)))

    # --- Torso: tapered profile (wider chest, narrower waist) via revolve. ---
    torso_profile = np.array([
        [9.0 * scale, 0.0],
        [10.0 * scale, 25.0 * scale],
        [10.5 * scale, 40.0 * scale],
        [8.0 * scale, 55.0 * scale],
    ])
    torso = trimesh.creation.revolve(torso_profile, sections=24, cap=True)
    _to_y_up(torso)
    torso_local = torso.vertices.copy()  # Y = height (0..55*scale), X/Z = radial
    torso.apply_translation([0, 90 * scale, 0])
    _set_visual(torso, _flat_uv(len(torso.vertices)), material)

    # Split at the waist: above -> shoulder/chest region, below -> hip region.
    waist_y_local = 0.53 * (55.0 * scale)  # matches avatar_builder.py's _WAIST_Y_FRACTION
    upper_mask = torso_local[:, 1] > waist_y_local
    lower_mask = ~upper_mask

    torso_deltas = _zero_deltas(len(torso.vertices))
    torso_deltas["shoulderWidth"][upper_mask, 0] = torso_local[upper_mask, 0] * SHOULDER_TORSO_GAIN
    torso_deltas["shoulderWidth"][upper_mask, 2] = torso_local[upper_mask, 2] * SHOULDER_TORSO_GAIN
    torso_deltas["hipWidth"][lower_mask, 0] = torso_local[lower_mask, 0] * HIP_TORSO_GAIN
    torso_deltas["hipWidth"][lower_mask, 2] = torso_local[lower_mask, 2] * HIP_TORSO_GAIN
    torso_deltas["bodyType"][:, 0] = torso_local[:, 0] * BODY_TYPE_GAIN
    torso_deltas["bodyType"][:, 2] = torso_local[:, 2] * BODY_TYPE_GAIN
    parts.append(torso)
    deltas_list.append(torso_deltas)

    # --- Pelvis: gender-differentiated width (narrower for male, wider for female). ---
    pelvis_radius = (11.0 if gender == "female" else 9.0) * scale
    pelvis = trimesh.creation.cylinder(radius=pelvis_radius, height=15 * scale, sections=24)
    _to_y_up(pelvis)
    pelvis_local = pelvis.vertices.copy()  # X/Z = radial, Y = height (centered at 0)
    pelvis.apply_translation([0, 85 * scale, 0])
    _set_visual(pelvis, _flat_uv(len(pelvis.vertices)), material)

    pelvis_deltas = _zero_deltas(len(pelvis.vertices))
    pelvis_deltas["hipWidth"][:, 0] = pelvis_local[:, 0] * HIP_TORSO_GAIN
    pelvis_deltas["hipWidth"][:, 2] = pelvis_local[:, 2] * HIP_TORSO_GAIN
    pelvis_deltas["bodyType"][:, 0] = pelvis_local[:, 0] * BODY_TYPE_GAIN
    pelvis_deltas["bodyType"][:, 2] = pelvis_local[:, 2] * BODY_TYPE_GAIN
    parts.append(pelvis)
    deltas_list.append(pelvis_deltas)

    # --- Arms + hand stubs. ---
    shoulder_x = (13.5 if gender == "male" else 12.5) * scale  # overlaps the torso's chest radius (10.0)
    arm_half_span = 30 * scale  # capsule(height=52*scale) -> Y in [-arm_half_span, arm_half_span]
    for side in (-1, 1):
        arm = trimesh.creation.capsule(height=52 * scale, radius=4 * scale, count=[10, 10])
        _to_y_up(arm)

        # 0 at the top (shoulder, fixed) .. 1 at the bottom (wrist, extends down).
        ramp = np.clip((arm_half_span - arm.vertices[:, 1]) / (2 * arm_half_span), 0, 1)
        # Taper so the arm is slimmer at the wrist than at the shoulder.
        taper = 1.0 - 0.25 * ramp
        arm.vertices[:, 0] *= taper
        arm.vertices[:, 2] *= taper
        arm_local = arm.vertices.copy()  # X/Z = radial, Y = long axis (top = shoulder)
        arm.apply_translation([side * shoulder_x, 115 * scale, 0])
        _set_visual(arm, _flat_uv(len(arm.vertices)), material)

        arm_deltas = _zero_deltas(len(arm.vertices))
        arm_deltas["armLength"][:, 1] = -ramp * (52 * scale) * ARM_LENGTH_GAIN
        arm_deltas["shoulderWidth"][:, 0] = side * shoulder_x * SHOULDER_SHIFT_GAIN
        arm_deltas["bodyType"][:, 0] = arm_local[:, 0] * BODY_TYPE_GAIN
        arm_deltas["bodyType"][:, 2] = arm_local[:, 2] * BODY_TYPE_GAIN
        parts.append(arm)
        deltas_list.append(arm_deltas)

        hand = trimesh.creation.icosphere(subdivisions=2, radius=3.5 * scale)
        hand.apply_scale([1.0, 1.3, 0.85])  # elongated along the arm, hand-like rather than a ball
        hand_local = hand.vertices.copy()
        hand.apply_translation([side * shoulder_x, 83 * scale, 0])  # overlaps the wrist (arm bottom = 85)
        _set_visual(hand, _flat_uv(len(hand.vertices)), material)

        hand_deltas = _zero_deltas(len(hand.vertices))
        hand_deltas["armLength"][:, 1] = -(52 * scale) * ARM_LENGTH_GAIN  # follows the wrist fully
        hand_deltas["shoulderWidth"][:, 0] = side * shoulder_x * SHOULDER_SHIFT_GAIN
        hand_deltas["bodyType"][:, 0] = hand_local[:, 0] * BODY_TYPE_GAIN
        hand_deltas["bodyType"][:, 2] = hand_local[:, 2] * BODY_TYPE_GAIN
        parts.append(hand)
        deltas_list.append(hand_deltas)

    # --- Legs + foot stubs. ---
    hip_x = (8.0 if gender == "male" else 9.0) * scale
    leg_half_span = 37 * scale  # capsule(height=74*scale) -> Y in [-leg_half_span, leg_half_span]
    for side in (-1, 1):
        leg = trimesh.creation.capsule(height=74 * scale, radius=5 * scale, count=[10, 10])
        _to_y_up(leg)

        # 0 at the top (hip, fixed) .. 1 at the bottom (ankle, extends down).
        ramp = np.clip((leg_half_span - leg.vertices[:, 1]) / (2 * leg_half_span), 0, 1)
        # Taper so the leg is slimmer at the ankle than at the hip.
        taper = 1.0 - 0.3 * ramp
        leg.vertices[:, 0] *= taper
        leg.vertices[:, 2] *= taper
        leg_local = leg.vertices.copy()  # X/Z = radial, Y = long axis (top = hip)
        leg.apply_translation([side * hip_x, 43 * scale, 0])  # top (80) overlaps the pelvis bottom (77.5)
        _set_visual(leg, _flat_uv(len(leg.vertices)), material)

        leg_deltas = _zero_deltas(len(leg.vertices))
        leg_deltas["legLength"][:, 1] = -ramp * (74 * scale) * LEG_LENGTH_GAIN
        leg_deltas["hipWidth"][:, 0] = side * hip_x * HIP_SHIFT_GAIN
        leg_deltas["bodyType"][:, 0] = leg_local[:, 0] * BODY_TYPE_GAIN
        leg_deltas["bodyType"][:, 2] = leg_local[:, 2] * BODY_TYPE_GAIN
        parts.append(leg)
        deltas_list.append(leg_deltas)

        foot = trimesh.creation.icosphere(subdivisions=2, radius=4.5 * scale)
        foot.apply_scale([1.0, 0.6, 1.6])
        foot_local = foot.vertices.copy()
        foot.apply_translation([side * hip_x, 7 * scale, 4 * scale])  # overlaps the ankle (leg bottom = 6)
        _set_visual(foot, _flat_uv(len(foot.vertices)), material)

        foot_deltas = _zero_deltas(len(foot.vertices))
        foot_deltas["legLength"][:, 1] = -(74 * scale) * LEG_LENGTH_GAIN  # follows the ankle fully
        foot_deltas["hipWidth"][:, 0] = side * hip_x * HIP_SHIFT_GAIN
        foot_deltas["bodyType"][:, 0] = foot_local[:, 0] * BODY_TYPE_GAIN
        foot_deltas["bodyType"][:, 2] = foot_local[:, 2] * BODY_TYPE_GAIN
        parts.append(foot)
        deltas_list.append(foot_deltas)

    mesh = trimesh.util.concatenate(parts)
    deltas = {
        name: np.concatenate([d[name] for d in deltas_list], axis=0).astype(np.float32)
        for name in MORPH_TARGET_NAMES
    }
    return mesh, deltas


def create_hairstyle(style: str, gender: str) -> trimesh.Trimesh:
    """Create a hairstyle mesh with its own tinted PBR material.

    Args:
        style: Hairstyle name (short, long, curly, etc.)
        gender: 'male' or 'female'

    Returns:
        trimesh.Trimesh object for the hairstyle, positioned to sit on the
        scalp of create_humanoid_model()'s head for the same gender.
    """
    # Match create_humanoid_model's per-gender scale so hair lines up with
    # the scalp (head center is at y = 155 * scale for that gender).
    height = 170 if gender == "male" else 160
    scale = height / 170
    scalp_y = 155 * scale

    if gender == "male":
        if style == "short":
            hair = trimesh.creation.icosphere(subdivisions=2, radius=9 * scale)
            hair.apply_scale([1.0, 0.7, 1.0])
        elif style == "buzz":
            hair = trimesh.creation.icosphere(subdivisions=1, radius=8.5 * scale)
        elif style == "medium":
            hair = trimesh.creation.icosphere(subdivisions=2, radius=9.5 * scale)
            hair.apply_scale([1.0, 0.9, 1.0])
        elif style == "textured":
            hair = trimesh.creation.icosphere(subdivisions=3, radius=10 * scale)
        else:  # slicked_back
            hair = trimesh.creation.icosphere(subdivisions=2, radius=9 * scale)
            hair.apply_scale([1.1, 0.8, 1.1])
    else:  # female
        if style == "pixie_cut":
            hair = trimesh.creation.icosphere(subdivisions=1, radius=8.5 * scale)
        elif style == "short_straight":
            hair = trimesh.creation.icosphere(subdivisions=2, radius=9.2 * scale)
            hair.apply_scale([1.0, 1.2, 1.0])
        elif style == "medium_straight":
            hair = trimesh.creation.capsule(height=16 * scale, radius=9.5 * scale, count=[8, 8])
            _to_y_up(hair)
            hair.apply_translation([0, -8 * scale, 0])
        elif style == "long_straight":
            hair = trimesh.creation.capsule(height=36 * scale, radius=9.5 * scale, count=[8, 8])
            _to_y_up(hair)
            hair.apply_translation([0, -18 * scale, 0])
        elif style == "long_wavy":
            hair = trimesh.creation.capsule(height=36 * scale, radius=10.5 * scale, count=[8, 8])
            _to_y_up(hair)
            hair.apply_translation([0, -18 * scale, 0])
        elif style == "long_curly":
            hair = trimesh.creation.icosphere(subdivisions=3, radius=11 * scale)
            hair.apply_scale([1.0, 2.5, 1.0])
        elif style == "ponytail":
            head_hair = trimesh.creation.icosphere(subdivisions=2, radius=8.5 * scale)
            tail = trimesh.creation.capsule(height=24 * scale, radius=3 * scale, count=[8, 8])
            _to_y_up(tail)
            tail.apply_translation([0, -20 * scale, -8 * scale])
            hair = trimesh.util.concatenate([head_hair, tail])
        else:  # bun
            hair = trimesh.creation.icosphere(subdivisions=2, radius=6 * scale)
            hair.apply_translation([0, 12 * scale, 0])

    # Subtle, reproducible "bumpy" surface displacement along vertex normals.
    seed = zlib.crc32(f"{gender}_{style}".encode())
    rng = np.random.default_rng(seed)
    displacement = rng.normal(0, 0.3 * scale, (len(hair.vertices), 1))
    hair.vertices = hair.vertices + hair.vertex_normals * displacement

    # Position on the scalp.
    hair.apply_translation([0, scalp_y, 0])

    # Own material, tinted with the placeholder hair color (recolored at
    # runtime from the user's detected hair_rgb).
    hair_color_norm = [c / 255.0 for c in HAIR_TONE]
    hair.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=hair_color_norm + [1.0],
            roughnessFactor=0.75,
            metallicFactor=0.0,
        )
    )

    return hair


def _add_morph_targets(glb_bytes: bytes, deltas: dict) -> bytes:
    """Append `deltas` (see create_humanoid_model) to `glb_bytes` as named
    morph targets on its first mesh primitive."""
    gltf = GLTF2.load_from_bytes(glb_bytes)
    blob = bytearray(gltf.binary_blob() or b"")

    prim = gltf.meshes[0].primitives[0]
    n = gltf.accessors[prim.attributes.POSITION].count

    targets = []
    target_names = []
    for name in MORPH_TARGET_NAMES:
        arr = deltas[name].astype(np.float32)
        if arr.shape != (n, 3):
            raise ValueError(f"morph target '{name}': expected shape ({n}, 3), got {arr.shape}")

        data = arr.tobytes()
        pad = (-len(data)) % 4
        offset = len(blob)
        blob.extend(data + b"\x00" * pad)

        bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(data)))

        acc_idx = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=bv_idx, componentType=FLOAT, count=n, type=VEC3,
            min=[float(arr[:, i].min()) for i in range(3)],
            max=[float(arr[:, i].max()) for i in range(3)],
        ))

        targets.append({"POSITION": acc_idx})
        target_names.append(name)

    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))

    prim.targets = targets
    gltf.meshes[0].weights = [0.0] * len(targets)
    gltf.meshes[0].extras = {"targetNames": target_names}

    return b"".join(gltf.save_to_bytes())


def save_glb(mesh: trimesh.Trimesh, path: Path):
    """Save mesh as GLB file.

    Args:
        mesh: trimesh.Trimesh to save
        path: Path to save GLB file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type="glb")
    print(f"Created: {path}")


def save_glb_with_morphs(mesh: trimesh.Trimesh, deltas: dict, path: Path):
    """Save `mesh` as a GLB file with `deltas` baked in as named morph
    targets (see create_humanoid_model / _add_morph_targets)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    glb_bytes = mesh.export(file_type="glb")
    glb_bytes = _add_morph_targets(glb_bytes, deltas)
    path.write_bytes(glb_bytes)
    print(f"Created: {path} (with {len(MORPH_TARGET_NAMES)} morph targets)")


def main():
    """Generate all test avatars."""

    print("=== Generating Test Avatar Models ===\n")

    base_dir = Path("assets")

    print("Base Models:")
    male_model, male_deltas = create_humanoid_model("male", height=170)
    save_glb_with_morphs(male_model, male_deltas, base_dir / "avatars" / "male" / "base.glb")

    female_model, female_deltas = create_humanoid_model("female", height=160)
    save_glb_with_morphs(female_model, female_deltas, base_dir / "avatars" / "female" / "base.glb")

    print("\nMale Hairstyles:")
    male_styles = ["short", "buzz", "medium", "textured", "slicked_back"]
    for style in male_styles:
        hair = create_hairstyle(style, "male")
        save_glb(hair, base_dir / "hair" / "male" / f"{style}.glb")

    print("\nFemale Hairstyles:")
    female_styles = [
        "pixie_cut", "short_straight", "medium_straight", "long_straight",
        "long_wavy", "long_curly", "ponytail", "bun"
    ]
    for style in female_styles:
        hair = create_hairstyle(style, "female")
        save_glb(hair, base_dir / "hair" / "female" / f"{style}.glb")

    print("\nMobile assets:")
    mobile_avatars = Path("mobile_app/assets/avatars")
    mobile_avatars.mkdir(parents=True, exist_ok=True)
    shutil.copy(base_dir / "avatars" / "male" / "base.glb", mobile_avatars / "male.glb")
    print(f"Created: {mobile_avatars / 'male.glb'}")
    shutil.copy(base_dir / "avatars" / "female" / "base.glb", mobile_avatars / "female.glb")
    print(f"Created: {mobile_avatars / 'female.glb'}")

    mobile_hair = Path("mobile_app/assets/hair")
    mobile_hair.mkdir(parents=True, exist_ok=True)
    # mobile/src/types.ts's HairAssetKey -> source GLB generated above.
    hair_map = {
        "short": base_dir / "hair" / "male" / "short.glb",
        "long": base_dir / "hair" / "female" / "long_straight.glb",
        "curly": base_dir / "hair" / "female" / "long_curly.glb",
        "buzz": base_dir / "hair" / "male" / "buzz.glb",
        "ponytail": base_dir / "hair" / "female" / "ponytail.glb",
    }
    for key, src in hair_map.items():
        dest = mobile_hair / f"{key}.glb"
        shutil.copy(src, dest)
        print(f"Created: {dest}")

    print("\n=== Done! ===")
    print("\nGenerated models:")
    print("  - assets/avatars/male/base.glb (6 morph targets)")
    print("  - assets/avatars/female/base.glb (6 morph targets)")
    print("  - assets/hair/male/*.glb (5 styles)")
    print("  - assets/hair/female/*.glb (8 styles)")
    print("  - mobile/assets/avatars/{male,female}.glb")
    print("  - mobile/assets/hair/{short,long,curly,buzz,ponytail}.glb")
    print("\nYou can now test Phase 2:")
    print("  $env:AVATAR_USE_REALISTIC=1")
    print("  python -m server.app")
    print("\nNext: Import and refine these in Blender for better quality!")


if __name__ == "__main__":
    main()
