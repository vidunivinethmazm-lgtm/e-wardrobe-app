"""Model 7 — Blender-side fitting of a *generated* garment mesh (from
Unique3D/`garment_mesh_generation.py`, never a category template) onto the
existing personalized MakeHuman avatar.

Run through Blender, not normal Python (invoked by
`avatar_pipeline.model7_garment_fitting.garment_fit_runner.
BlenderGarmentFitRunner`, same convention as `scripts/blender_fit_face.py`):

    blender --background --python scripts/blender_fit_garment_mesh.py -- <config.json>

`config.json` (written by `garment_fit_runner.py`):
    {
        "avatar_glb": "<path to the existing personalized avatar .glb>",
        "garment_glb": "<path to the GENERATED garment mesh — its own
                          topology/UVs/texture, not a template>",
        "region_centers": {"shoulder": [x,y,z], "chest": [...], "waist": [...], "hip": [...]},
        "region_scales": {"shoulder_scale": ..., "chest_scale": ..., ...},
        "output_glb": "<path to write the fitted garment .glb to>"
    }

Pipeline run here on the *imported generated garment mesh itself* (its
vertex/face count, UVs, and material/texture are preserved throughout —
this script only ever displaces vertices and adds modifiers, it never
imports or substitutes a different mesh):

    1. Region-proximity vertex groups — one group per `region_centers` key,
       weighted by inverse distance from that region's avatar-space center
       (mirrors `garment_region_fitting.region_deform_mesh`'s Python
       fallback, but as real Blender vertex groups so the modifiers below
       can target them).
    2. Region-wise local scaling — each vertex is displaced radially from
       the mesh's own vertical axis by its blended `region_scales` weight,
       i.e. the same math `region_deform_mesh` does, run once more here so
       the Blender path doesn't depend on the mock path having pre-scaled
       the mesh.
    3. Surface Deform — binds the (now region-scaled) garment to the avatar
       body mesh.
    4. Shrinkwrap, small positive offset — sits the garment just outside
       the avatar's skin instead of clipping into it.
    5. Collision (avatar) + Cloth (garment) simulation — natural drape
       instead of a rigid scaled shell.
    6. Corrective Smooth — relaxes local distortion from steps 2-5 while
       preserving the garment's own silhouette/pattern.
    7. Apply modifiers, export the result as GLB — same UVs/material/image
       the imported garment mesh already had.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
import mathutils


def _parse_args() -> dict:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("usage: blender --background --python blender_fit_garment_mesh.py -- <config.json>")
    return json.loads(Path(argv[argv.index("--") + 1]).read_text())


def _reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _import_glb(path: str, name: str) -> bpy.types.Object:
    bpy.ops.import_scene.gltf(filepath=path)
    imported = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not imported:
        raise RuntimeError(f"no mesh found in imported glTF: {path}")
    obj = imported[0]
    obj.name = name
    return obj


def _build_region_vertex_groups(garment_obj, region_centers: dict) -> dict:
    """Step 1: one vertex group per region, inverse-distance weighted —
    every vertex gets SOME weight in every group (normalized so weights
    across all groups sum to 1 per vertex), giving the smooth cross-region
    blend `region_deform_mesh` documents, expressed as real vertex groups."""
    mesh = garment_obj.data
    centers = {name: mathutils.Vector(point) for name, point in region_centers.items()}
    groups = {name: garment_obj.vertex_groups.new(name=f"region_{name}") for name in centers}

    for vertex in mesh.vertices:
        dists = {name: (vertex.co - center).length + 1e-3 for name, center in centers.items()}
        inv = {name: 1.0 / d for name, d in dists.items()}
        total = sum(inv.values())
        for name, group in groups.items():
            weight = inv[name] / total
            group.add([vertex.index], weight, "REPLACE")

    return groups


def _apply_region_scaling(garment_obj, region_centers: dict, region_scales: dict) -> None:
    """Step 2: displaces each vertex radially from the mesh's own vertical
    (Y) axis by its region-blended scale, and vertically by `length_scale`
    around the mesh's own vertical center — same math as
    `garment_region_fitting.region_deform_mesh`."""
    mesh = garment_obj.data
    centers = {name: mathutils.Vector(point) for name, point in region_centers.items()}
    scale_key = {"shoulder": "shoulder_scale", "chest": "chest_scale", "waist": "waist_scale", "hip": "hip_scale"}

    y_values = [v.co.y for v in mesh.vertices]
    y_center = sum(y_values) / len(y_values) if y_values else 0.0
    length_scale = region_scales.get("length_scale", 1.0)

    for vertex in mesh.vertices:
        dists = {name: (vertex.co - center).length + 1e-3 for name, center in centers.items()}
        inv = {name: 1.0 / d for name, d in dists.items()}
        total = sum(inv.values())
        blended_scale = sum((inv[name] / total) * region_scales.get(scale_key[name], 1.0) for name in centers)

        vertex.co.x *= blended_scale
        vertex.co.z *= blended_scale
        vertex.co.y = y_center + (vertex.co.y - y_center) * length_scale

    mesh.update()


def _apply_surface_deform(garment_obj, avatar_obj) -> None:
    modifier = garment_obj.modifiers.new(name="GarmentSurfaceDeform", type="SURFACE_DEFORM")
    modifier.target = avatar_obj
    bpy.context.view_layer.objects.active = garment_obj
    bpy.ops.object.surfacedeform_bind(modifier=modifier.name)


def _apply_shrinkwrap(garment_obj, avatar_obj, offset: float = 0.008) -> None:
    modifier = garment_obj.modifiers.new(name="GarmentShrinkwrap", type="SHRINKWRAP")
    modifier.target = avatar_obj
    modifier.wrap_method = "NEAREST_SURFACEPOINT"
    modifier.offset = offset


def _apply_cloth_simulation(garment_obj, avatar_obj, frames: int = 20) -> None:
    avatar_obj.modifiers.new(name="Collision", type="COLLISION")

    cloth_modifier = garment_obj.modifiers.new(name="Cloth", type="CLOTH")
    settings = cloth_modifier.settings
    settings.quality = 5
    settings.mass = 0.3
    settings.tension_stiffness = 15
    settings.compression_stiffness = 15
    settings.shear_stiffness = 5
    settings.bending_stiffness = 0.5

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frames
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)


def _apply_corrective_smooth(garment_obj) -> None:
    modifier = garment_obj.modifiers.new(name="GarmentCorrectiveSmooth", type="CORRECTIVE_SMOOTH")
    modifier.factor = 0.4
    modifier.iterations = 3


def _apply_all_modifiers(obj) -> None:
    bpy.context.view_layer.objects.active = obj
    for modifier in list(obj.modifiers):
        try:
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        except RuntimeError:
            bpy.ops.object.convert(target="MESH")
            break


def _export_glb(obj, output_path: str) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(filepath=output_path, export_format="GLB", use_selection=True)


def main() -> None:
    config = _parse_args()

    _reset_scene()
    avatar_obj = _import_glb(config["avatar_glb"], "avatar")
    garment_obj = _import_glb(config["garment_glb"], "garment")  # the GENERATED mesh, never a template

    _build_region_vertex_groups(garment_obj, config["region_centers"])
    _apply_region_scaling(garment_obj, config["region_centers"], config["region_scales"])
    _apply_surface_deform(garment_obj, avatar_obj)
    _apply_shrinkwrap(garment_obj, avatar_obj)
    _apply_cloth_simulation(garment_obj, avatar_obj)
    _apply_corrective_smooth(garment_obj)
    _apply_all_modifiers(garment_obj)

    Path(config["output_glb"]).parent.mkdir(parents=True, exist_ok=True)
    _export_glb(garment_obj, config["output_glb"])


if __name__ == "__main__":
    main()
