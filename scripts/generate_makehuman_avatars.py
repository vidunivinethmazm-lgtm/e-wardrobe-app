"""Generate MakeHuman/MPFB humanoid base avatars as staged GLB files.

Run through Blender, not normal Python:

    blender --background --python scripts/generate_makehuman_avatars.py

For each gender this writes a "base" GLB (all 6 morph-target shape keys at
0.0) plus one "morphed" GLB per shape key (that key at 1.0, others at 0.0) to
``.codex/generated_makehuman/_stage``. Run ``scripts/bake_makehuman_morphs.py``
(plain Python, not Blender) afterwards to diff the morphed exports against the
base export, remap UVs for the face-overlay feature, and bake the result into
``.codex/generated_makehuman/{male,female}.glb`` with the 6 named morph
targets the mobile app expects (``mobile/src/types.ts``'s ``MorphTargetName``).

The script does not overwrite the app's committed placeholder assets; copy
the final files into ``mobile/assets/avatars`` only after validating them.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path

import bpy

from bl_ext.user_default.mpfb.services.humanservice import HumanService
from bl_ext.user_default.mpfb.services.targetservice import TargetService

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / ".codex" / "generated_makehuman"
STAGE_DIR = OUT_DIR / "_stage"

# Maps each MorphTargetName (mobile/src/types.ts) to the MakeHuman target(s)
# (mpfb/data/targets/**/*.target.gz, matched by TargetService.target_full_path)
# that are combined into a single shape key. Order matches MORPH_TARGET_NAMES
# in generate_test_avatars.py.
MORPH_TARGET_SOURCES: dict[str, list[str]] = {
    "shoulderWidth": ["measure-shoulder-dist-incr"],
    "hipWidth": ["hip-scale-horiz-incr"],
    "armLength": ["measure-upperarm-length-incr", "measure-lowerarm-length-incr"],
    "legLength": ["measure-upperleg-height-incr", "measure-lowerleg-height-incr"],
    "bodyType": ["measure-waist-circ-incr"],
    "headWidth": ["head-scale-horiz-incr"],
}

MORPH_TARGET_NAMES = list(MORPH_TARGET_SOURCES)


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.62
    return material


def _read_target_string(full_path: str) -> str:
    if str(full_path).endswith(".gz"):
        with gzip.open(full_path, "rb") as handle:
            return handle.read().decode("utf-8")
    with open(full_path, "r") as handle:
        return handle.read()


def _add_morph_shape_keys(basemesh: bpy.types.Object) -> None:
    """Create one shape key per MORPH_TARGET_NAMES entry from the mapped
    MakeHuman target file(s), combining multiple targets (e.g. upper + lower
    arm length) additively into a single shape key. Each key's value is reset
    to 0.0 so the base mesh stays undeformed."""
    for morph_name, target_names in MORPH_TARGET_SOURCES.items():
        shape_key = None
        for i, target_name in enumerate(target_names):
            full_path = TargetService.target_full_path(target_name)
            if not full_path:
                raise RuntimeError(f"MakeHuman target not found: {target_name!r}")
            target_string = _read_target_string(full_path)
            shape_key = TargetService.target_string_to_shape_key(
                target_string, morph_name, basemesh, reuse_existing=(i > 0)
            )
        shape_key.value = 0.0


def _apply_shape_targets(basemesh: bpy.types.Object, shape_targets: dict[str, float]) -> None:
    """Apply named MakeHuman targets at a fixed, non-zero weight - baked into
    the exported "base" mesh (unlike _add_morph_shape_keys's zero-valued,
    runtime-adjustable keys). Used for the bust/waist/hip targets that define
    each body-shape category (Hourglass/Pear/Apple/Rectangle/InvertedTriangle).
    Shape key names are prefixed "shape_" so they can't collide with
    MORPH_TARGET_NAMES."""
    for target_name, weight in shape_targets.items():
        full_path = TargetService.target_full_path(target_name)
        if not full_path:
            raise RuntimeError(f"MakeHuman target not found: {target_name!r}")
        target_string = _read_target_string(full_path)
        shape_key = TargetService.target_string_to_shape_key(
            target_string, f"shape_{target_name}", basemesh, reuse_existing=False
        )
        shape_key.value = weight


def _export_glb(basemesh: bpy.types.Object, filepath: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    basemesh.select_set(True)
    bpy.context.view_layer.objects.active = basemesh

    filepath.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(filepath),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_materials="EXPORT",
        export_yup=True,
        export_morph=False,
    )


def create_avatar(
    name: str,
    gender: float,
    height: float,
    weight: float,
    muscle: float,
    cupsize: float = 0.5,
    shape_targets: dict[str, float] | None = None,
) -> dict[str, Path]:
    reset_scene()

    macros = TargetService.get_default_macro_info_dict()
    macros.update(
        {
            "gender": gender,
            "height": height,
            "weight": weight,
            "muscle": muscle,
            "cupsize": cupsize,
            "age": 0.42,
            "proportions": 0.52,
        }
    )

    basemesh = HumanService.create_human(
        mask_helpers=True,
        detailed_helpers=False,
        extra_vertex_groups=False,
        feet_on_ground=True,
        scale=0.1,
        macro_detail_dict=macros,
    )
    basemesh.name = name
    basemesh.data.name = f"{name}_mesh"

    skin = make_material(f"{name}_skin", (0.78, 0.58, 0.42, 1.0))
    basemesh.data.materials.clear()
    basemesh.data.materials.append(skin)

    if shape_targets:
        _apply_shape_targets(basemesh, shape_targets)

    _add_morph_shape_keys(basemesh)

    paths: dict[str, Path] = {"base": STAGE_DIR / f"{name}_base.glb"}
    _export_glb(basemesh, paths["base"])

    key_blocks = basemesh.data.shape_keys.key_blocks
    for morph_name in MORPH_TARGET_NAMES:
        key_blocks[morph_name].value = 1.0
        morph_path = STAGE_DIR / f"{name}_{morph_name}.glb"
        _export_glb(basemesh, morph_path)
        paths[morph_name] = morph_path
        key_blocks[morph_name].value = 0.0

    return paths



# Per-gender baseline macros (unchanged from the original 2-variant script).
GENDER_PRESETS: dict[str, dict[str, float]] = {
    "male": {"gender": 1.0, "height": 0.58, "muscle": 0.56},
    "female": {"gender": 0.0, "height": 0.48, "muscle": 0.42},
}

# Body-SHAPE presets (Hourglass/Pear/Apple/Rectangle/InvertedTriangle -
# matching model1_body_shape's CLASS_NAMES exactly), as opposed to the earlier
# slim/average/curvy SIZE presets. Shape is a bust:waist:hip ratio,
# independent of overall size, so `weight`/`muscle` stay at the gender
# baseline (average) for all 5 - only the targets below vary. Each target is
# an independent measure-*-incr/decr shape key (mpfb/data/targets/torso/,
# hip/, stomach/), applied at a fixed non-zero weight via _apply_shape_targets
# (see that function's docstring for why this differs from the runtime
# morphs). `cupsize` is the breast-volume macro (female-only effect,
# confirmed via target filename inspection) - included per shape since it
# reinforces the bust-circumference targets on female meshes specifically.
#
# Second pass, after measuring each target's actual max displacement ceiling
# directly from its .target.gz data (all of these turned out to be the same
# fixed-per-vertex-displacement format - there's no real "multiplicative
# scale" target type, despite that being assumed during planning). Every
# primary differentiator below is now at its measured 1.0 ceiling (2-5cm on a
# ~1.5-1.7m figure - a real but modest effect, this category of target was
# never designed for dramatic change in isolation, unlike the macrodetails
# weight blend used for the slim/average/curvy size variants). Rectangle's
# only available lever (stomach-tone-incr) caps at just 0.85cm even at 1.0 -
# flagged as the highest-risk-for-weak-distinctiveness shape, now backed by
# a measured number instead of a guess.
BODY_SHAPE_PRESETS: dict[str, dict] = {
    "Hourglass": {
        "cupsize": 0.7,
        "targets": {
            "measure-waist-circ-decr": 1.0,
            "measure-bust-circ-incr": 1.0,
            "measure-hips-circ-incr": 1.0,
            "hip-scale-horiz-incr": 1.0,
        },
    },
    "Pear": {
        "cupsize": 0.35,
        "targets": {
            "measure-waist-circ-decr": 0.4,
            "measure-bust-circ-decr": 0.6,
            "measure-hips-circ-incr": 1.0,
            "hip-scale-horiz-incr": 1.0,
        },
    },
    "Apple": {
        "cupsize": 0.5,
        "targets": {
            "measure-waist-circ-incr": 1.0,
            "measure-bust-circ-incr": 0.5,
            "measure-hips-circ-incr": 0.5,
            "stomach-tone-decr": 1.0,
        },
    },
    "Rectangle": {
        "cupsize": 0.5,
        "targets": {
            "stomach-tone-incr": 1.0,
        },
    },
    "InvertedTriangle": {
        "cupsize": 0.5,
        "targets": {
            "measure-bust-circ-incr": 1.0,
            "measure-hips-circ-decr": 1.0,
            "hip-scale-horiz-decr": 1.0,
            "torso-vshape-incr": 1.0,
        },
    },
}


def main() -> None:
    # For faster iteration while tuning presets, restrict which genders run
    # via MAKEHUMAN_GENDERS (comma-separated, e.g. "female"). Defaults to both.
    gender_names = os.environ.get("MAKEHUMAN_GENDERS", "male,female").split(",")
    for gender_name in gender_names:
        gender_macros = GENDER_PRESETS[gender_name]
        for shape_name, shape_preset in BODY_SHAPE_PRESETS.items():
            name = f"{gender_name}_{shape_name}"
            paths = create_avatar(
                name,
                gender=gender_macros["gender"],
                height=gender_macros["height"],
                weight=0.5,
                muscle=gender_macros["muscle"],
                cupsize=shape_preset.get("cupsize", 0.5),
                shape_targets=shape_preset.get("targets"),
            )
            for key in ("base", *MORPH_TARGET_NAMES):
                print(f"MAKEHUMAN_STAGE {name} {key} {paths[key]}")


if __name__ == "__main__":
    main()
