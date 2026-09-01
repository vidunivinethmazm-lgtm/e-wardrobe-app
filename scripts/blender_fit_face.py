"""Model 6 — Blender-side 3D face fitting: blends an already-scaled-and-
aligned generated face mesh (see `avatar_pipeline.model6_body3d.
face_mesh_fitting.align_face_mesh`) into the existing MakeHuman avatar's
head region only.

Run through Blender, not normal Python (invoked by
`avatar_pipeline.model6_body3d.face_fit_runner.BlenderFaceFitRunner`, same
convention as `scripts/generate_makehuman_avatars.py` /
`scripts/blender_fit_garment.py`):

    blender --background --python scripts/blender_fit_face.py -- <config.json>

`config.json` (written by `face_fit_runner.py`):
    {
        "avatar_glb": "<path to the existing personalized avatar .glb>",
        "generated_face_glb": "<path to the fitted (scaled+aligned) face mesh>",
        "eye_center": [x, y, z],   # avatar-space, from the fitting stage
        "chin": [x, y, z],
        "jaw_left": [x, y, z],
        "jaw_right": [x, y, z],
        "output_glb": "<path to write the merged avatar .glb to>"
    }

The face mesh arrives *already* scaled/rotated/translated into the avatar's
coordinate system (Python-side, via `face_mesh_fitting.align_face_mesh` —
this script does no additional scale/rotation math, only region-limited
mesh integration):

    1. Region-wise blend target — a "face region" vertex group on the
       avatar's head mesh, selected by proximity to `eye_center`/`chin`
       (a sphere roughly `2x` the eye-to-chin distance), NOT the whole
       head/body: this is what guarantees the body and neck stay untouched.
    2. Surface Deform / Shrinkwrap (small positive offset) of the avatar's
       face-region vertices toward the generated face mesh, restricted to
       that vertex group so nothing outside the face region moves.
    3. Boundary blending — vertex-group weights fall off smoothly from 1.0
       at the face center to 0.0 at the region boundary (forehead/cheeks/
       jaw/neck), so the seam between original and fitted geometry isn't a
       hard edge.
    4. Corrective Smooth, restricted to the same vertex group.
    5. Face-UV texture bake: the generated face's texture (if any) is baked
       into the avatar's existing face-UV texture, inside the same region.
    6. Apply modifiers, export the merged avatar as GLB.

This script does the real, avatar-preserving 3D integration; the Python-
only `MockFaceFitRunner` (see `face_fit_runner.py`) is a safe passthrough
used when Blender isn't available (dev machines, CI) — it does not attempt
this blend at all rather than approximate it outside Blender's modifier
stack.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


def _parse_args() -> dict:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("usage: blender --background --python blender_fit_face.py -- <config.json>")
    config_path = Path(argv[argv.index("--") + 1])
    return json.loads(config_path.read_text())


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


def _face_region_radius(eye_center, chin) -> float:
    import mathutils
    eye_v = mathutils.Vector(eye_center)
    chin_v = mathutils.Vector(chin)
    return 2.0 * (eye_v - chin_v).length


def _build_face_region_vertex_group(avatar_obj: bpy.types.Object, eye_center, chin) -> str:
    """Step 1 + 3: creates (or overwrites) a 'face_region' vertex group on
    the avatar's head mesh with weights falling off smoothly from 1.0 at
    `eye_center` to 0.0 at `radius` away — this is the *only* thing later
    steps are allowed to touch, guaranteeing the body/neck stay untouched."""
    import mathutils

    group_name = "face_region"
    if group_name in avatar_obj.vertex_groups:
        avatar_obj.vertex_groups.remove(avatar_obj.vertex_groups[group_name])
    group = avatar_obj.vertex_groups.new(name=group_name)

    center = mathutils.Vector(eye_center)
    radius = _face_region_radius(eye_center, chin)

    mesh = avatar_obj.data
    for vertex in mesh.vertices:
        distance = (vertex.co - center).length
        if distance >= radius:
            continue
        # Smoothstep falloff: 1.0 at the center, 0.0 at the boundary.
        t = 1.0 - (distance / radius)
        weight = t * t * (3.0 - 2.0 * t)
        group.add([vertex.index], weight, "REPLACE")

    return group_name


def _apply_surface_deform_to_region(avatar_obj, face_obj, group_name: str) -> None:
    """Step 2: Surface Deform binds the avatar's face-region vertices to the
    generated (already scaled/aligned) face mesh, restricted to
    `group_name` via the modifier's vertex group mask so nothing outside
    the face region is affected."""
    modifier = avatar_obj.modifiers.new(name="FaceSurfaceDeform", type="SURFACE_DEFORM")
    modifier.target = face_obj
    modifier.vertex_group = group_name
    bpy.context.view_layer.objects.active = avatar_obj
    bpy.ops.object.surfacedeform_bind(modifier=modifier.name)


def _apply_shrinkwrap_to_region(avatar_obj, face_obj, group_name: str, offset: float = 0.004) -> None:
    modifier = avatar_obj.modifiers.new(name="FaceShrinkwrap", type="SHRINKWRAP")
    modifier.target = face_obj
    modifier.wrap_method = "NEAREST_SURFACEPOINT"
    modifier.offset = offset
    modifier.vertex_group = group_name


def _apply_corrective_smooth_to_region(avatar_obj, group_name: str) -> None:
    modifier = avatar_obj.modifiers.new(name="FaceCorrectiveSmooth", type="CORRECTIVE_SMOOTH")
    modifier.factor = 0.5
    modifier.iterations = 3
    modifier.vertex_group = group_name


def _bake_face_texture(avatar_obj, face_obj) -> None:
    """Step 5: bakes the generated face mesh's material/texture into the
    avatar's existing face-UV image, restricted (by the UV layout itself —
    the avatar's face UVs only cover the head) to the face region. Skips
    silently if either object has no material/image to bake from/to — the
    geometry blend above still applies without a texture bake."""
    if not face_obj.data.materials or not avatar_obj.data.materials:
        return
    try:
        bpy.context.view_layer.objects.active = avatar_obj
        face_obj.select_set(True)
        avatar_obj.select_set(True)
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.bake_type = "DIFFUSE"
        bpy.ops.object.bake(type="DIFFUSE")
    except RuntimeError:
        # Bake requires a configured image node per material; not fatal to
        # the geometry blend if the avatar's face material isn't set up for
        # it in a given asset — the mesh integration above already ran.
        pass


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
    face_obj = _import_glb(config["generated_face_glb"], "generated_face")

    group_name = _build_face_region_vertex_group(avatar_obj, config["eye_center"], config["chin"])
    _apply_surface_deform_to_region(avatar_obj, face_obj, group_name)
    _apply_shrinkwrap_to_region(avatar_obj, face_obj, group_name)
    _apply_corrective_smooth_to_region(avatar_obj, group_name)
    _bake_face_texture(avatar_obj, face_obj)
    _apply_all_modifiers(avatar_obj)

    Path(config["output_glb"]).parent.mkdir(parents=True, exist_ok=True)
    _export_glb(avatar_obj, config["output_glb"])


if __name__ == "__main__":
    main()
