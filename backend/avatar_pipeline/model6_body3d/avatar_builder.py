"""
Realistic avatar builder combining base models with hairstyles and textures.

Phase 2: Photorealistic avatars by merging:
- Base humanoid model (male/female)
- Matching hairstyle
- Face texture
- Skin/hair materials

This module handles combining multiple GLB files into a single realistic avatar.
"""

import io
import platform
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# On Python 3.13+/Windows, platform.system() can call into platform._wmi_query(),
# which queries WMI and hangs indefinitely if the WMI service is unavailable
# (e.g. in sandboxed environments). trimesh.interfaces.blender calls
# platform.system() at import time, so disable the WMI path first - this makes
# platform fall back to its `ver` subprocess check instead.
platform._wmi = None

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from pygltflib import (
        GLTF2,
        BufferView,
        Image as GltfImage,
        NormalMaterialTexture,
        PbrMetallicRoughness,
        Texture,
        TextureInfo,
    )
    HAS_PYGLTFLIB = True
except ImportError:
    HAS_PYGLTFLIB = False

from .asset_loader import AvatarAssetManager, select_hairstyle_for_facial_analysis


# body3d_params values (e.g. shoulder_width, hip_width) are fractions of
# height -- see avatar_pipeline/model6_body3d/params.py PARAM_RANGES, both
# roughly (0.16-0.34) with a midpoint of ~0.25. That midpoint is the "neutral"
# baseline a deformation scale of 1.0 should correspond to.
_NEUTRAL_WIDTH_FRACTION = 0.25

# Fraction of the mesh's Y-extent (min_y..max_y) that separates "upper body"
# (torso/arms/head, scaled by shoulder_width) from "lower body" (pelvis/legs,
# scaled by hip_width). Calibrated against generate_test_avatars.py's
# placeholder geometry.
_WAIST_Y_FRACTION = 0.53


def _embed_image(gltf: "GLTF2", blob: bytearray, image: Image.Image) -> int:
    """Append a PNG image to the GLB binary blob, register it as a texture.

    Returns the new texture's index into gltf.textures.
    """
    png_io = io.BytesIO()
    image.save(png_io, format="PNG")
    png_bytes = png_io.getvalue()

    # glTF buffer views must be 4-byte aligned.
    pad = (-len(png_bytes)) % 4
    offset = len(blob)
    blob.extend(png_bytes + b"\x00" * pad)

    bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(png_bytes)))

    img_idx = len(gltf.images)
    gltf.images.append(GltfImage(mimeType="image/png", bufferView=bv_idx))

    tex_idx = len(gltf.textures)
    gltf.textures.append(Texture(source=img_idx))

    return tex_idx


def _make_skin_normal_map(size: int = 256, strength: float = 6.0, seed: int = 42) -> Image.Image:
    """Generate a subtle, reproducible procedural normal map for skin detail."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (size, size, 2)).astype(np.float32)

    smoothed = np.empty_like(noise)
    for i in range(2):
        chan = noise[..., i]
        chan_8bit = ((chan - chan.min()) / (np.ptp(chan) + 1e-8) * 255).astype(np.uint8)
        blurred = Image.fromarray(chan_8bit, mode="L").filter(ImageFilter.GaussianBlur(2))
        smoothed[..., i] = (np.asarray(blurred, dtype=np.float32) / 255.0 - 0.5) * 2.0

    nx = smoothed[..., 0] * strength
    ny = smoothed[..., 1] * strength
    nz = np.full((size, size), 250.0, dtype=np.float32)

    normal = np.stack([nx, ny, nz], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

    rgb = ((normal * 0.5 + 0.5) * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


class AvatarBuilder:
    """Builds realistic avatars from base models, hairstyles, and textures."""

    def __init__(self, asset_manager: AvatarAssetManager):
        """Initialize avatar builder with asset manager.

        Args:
            asset_manager: Initialized AvatarAssetManager instance
        """
        self.assets = asset_manager

    def build_realistic_avatar(
        self,
        gender: str,
        facial_analysis: Dict[str, str],
        body3d_params: Dict[str, float],
        height: float,
        skin_rgb: np.ndarray,
        hair_rgb: tuple,
        face_crop: Optional[np.ndarray] = None,
        selfie_rgb: Optional[np.ndarray] = None,
        landmarks_2d: Optional[np.ndarray] = None,
        blend_mode: str = "feather",
        face_width: Optional[int] = None,
        face_height: Optional[int] = None,
    ) -> Optional[bytes]:
        """Build a realistic avatar combining base model and hairstyle.

        Uses ``makehuman_mesh.build_personalized_glb`` for morph deformation +
        face texture application (UV coordinates are preserved correctly), then
        adds the hairstyle on top.

        Args:
            gender: 'male' or 'female'
            facial_analysis: Result from extract_facial_analysis()
            body3d_params: Body proportions from default_params_from_measurements()
            height: Height in cm
            skin_rgb: Skin color RGB array
            hair_rgb: Hair color RGB tuple
            face_crop: Face texture as a square RGB array (optional)
            selfie_rgb: Full selfie photo for Delaunay warping (optional)
            landmarks_2d: MediaPipe face-mesh landmarks (optional)
            blend_mode: 'feather' or 'poisson' blending
            face_width: Original face crop width in pixels (optional)
            face_height: Original face crop height in pixels (optional)

        Returns:
            GLB file bytes (combined avatar), or None if assets not available
        """
        # Select and load matching hairstyle
        hairstyle_name = select_hairstyle_for_facial_analysis(facial_analysis, gender)
        hairstyle_glb = self.assets.get_hairstyle_model(gender, hairstyle_name)

        try:
            # Build the deformed + textured base mesh via makehuman_mesh, which
            # preserves the MakeHuman UV layout correctly (trimesh-based deformation
            # strips UV coordinates, causing the head to render black).
            from .makehuman_mesh import build_personalized_glb
            face_shape = facial_analysis.get("face_shape", "oval")

            base_glb = build_personalized_glb(
                body3d_params, height, skin_rgb, face_shape,
                face_crop=face_crop,
                gender=gender,
                selfie_rgb=selfie_rgb,
                landmarks_2d=landmarks_2d,
                blend_mode=blend_mode,
                face_width=face_width,
                face_height=face_height,
            )

            # Combine with hairstyle if available
            if hairstyle_glb is not None:
                return self._combine_with_hairstyle(base_glb, hairstyle_glb, hair_rgb)
            return base_glb

        except Exception as e:
            print(f"Error building realistic avatar: {e}")
            # Fallback to raw base model
            return self.assets.get_base_avatar_model(gender)

    def _deform_base_model(
        self,
        base_glb: bytes,
        body3d_params: Dict[str, float],
        height: float,
        gender: str,
    ) -> bytes:
        """Deform base model to match body parameters.

        Uses trimesh to:
        1. Load the GLB model
        2. Scale to target height
        3. Deform vertices based on body proportions
        4. Re-export as GLB

        Args:
            base_glb: GLB file bytes
            body3d_params: Body proportions dict (shoulder_width, hip_width,
                etc, as fractions of height -- see params.py PARAM_RANGES)
            height: Height in cm
            gender: 'male' or 'female'

        Returns:
            Deformed GLB bytes
        """
        if not HAS_TRIMESH:
            print("Warning: trimesh not installed, returning base model undeformed")
            return base_glb

        try:
            # Load GLB - handle both Scene and Trimesh objects
            loaded = trimesh.load(io.BytesIO(base_glb), file_type='glb')

            # If loaded as a Scene, extract the first geometry
            if isinstance(loaded, trimesh.Scene):
                geometries = list(loaded.geometry.values())
                if geometries:
                    mesh = geometries[0]
                else:
                    print("Warning: Scene has no geometries, returning base model")
                    return base_glb
            else:
                mesh = loaded

            # Get base height (matches generate_test_avatars.py's reference rig)
            base_height = 170 if gender == 'male' else 160
            height_scale = height / base_height

            # Normalize width params (fractions of height) against the
            # ~0.25 "neutral" baseline.
            shoulder_width = body3d_params.get('shoulder_width', _NEUTRAL_WIDTH_FRACTION)
            hip_width = body3d_params.get('hip_width', _NEUTRAL_WIDTH_FRACTION)
            shoulder_scale = shoulder_width / _NEUTRAL_WIDTH_FRACTION
            hip_scale = hip_width / _NEUTRAL_WIDTH_FRACTION

            # Apply height scaling uniformly
            mesh.apply_scale([height_scale, height_scale, height_scale])

            vertices = mesh.vertices
            if len(vertices) > 0:
                min_y = vertices[:, 1].min()
                max_y = vertices[:, 1].max()
                waist_y = min_y + (max_y - min_y) * _WAIST_Y_FRACTION

                # Above the waist (torso/arms/head): scale by shoulder_width.
                upper_mask = vertices[:, 1] > waist_y
                vertices[upper_mask, 0] *= shoulder_scale
                vertices[upper_mask, 2] *= shoulder_scale

                # At/below the waist (pelvis/legs): scale by hip_width.
                lower_mask = ~upper_mask
                vertices[lower_mask, 0] *= hip_scale
                vertices[lower_mask, 2] *= hip_scale

                mesh.vertices = vertices

            # Export to GLB bytes
            result = io.BytesIO()
            mesh.export(result, file_type='glb')
            result.seek(0)
            return result.read()

        except Exception as e:
            print(f"Warning: Failed to deform model: {e}")
            return base_glb

    def _apply_materials(
        self,
        glb: bytes,
        skin_rgb: np.ndarray,
    ) -> bytes:
        """Apply the detected skin color/material to the avatar's body material.

        Args:
            glb: GLB file bytes
            skin_rgb: Skin color RGB array (0-255)

        Returns:
            Modified GLB bytes with the skin material applied
        """
        if not HAS_PYGLTFLIB:
            print("Warning: pygltflib not installed, returning model with unchanged materials")
            return glb

        try:
            gltf = GLTF2.load_from_bytes(glb)

            skin_rgb_norm = [float(c) / 255.0 for c in np.clip(skin_rgb[:3], 0, 255)]
            skin_material = self.assets.get_skin_material(skin_rgb)

            if gltf.materials:
                skin_mat = gltf.materials[0]
                if skin_mat.pbrMetallicRoughness is None:
                    skin_mat.pbrMetallicRoughness = PbrMetallicRoughness()

                skin_mat.pbrMetallicRoughness.baseColorFactor = skin_rgb_norm + [1.0]
                skin_mat.pbrMetallicRoughness.roughnessFactor = skin_material['roughness']
                skin_mat.pbrMetallicRoughness.metallicFactor = skin_material['metallic']

            return b"".join(gltf.save_to_bytes())

        except Exception as e:
            print(f"Warning: Failed to apply materials: {e}")
            return glb

    def _apply_face_texture(
        self,
        glb: bytes,
        face_crop: np.ndarray,
        skin_rgb: np.ndarray,
        selfie_rgb: np.ndarray | None = None,
        landmarks_2d: np.ndarray | list | None = None,
        blend_mode: str = "feather",
        face_width: int | None = None,
        face_height: int | None = None,
    ) -> bytes:
        """Apply a face texture (plus a procedural normal map) to the head.

        When ``selfie_rgb`` and ``landmarks_2d`` are provided, uses Delaunay-
        triangulation warping (``face_texture_builder.build_head_texture_warped``)
        for photorealistic results.  Otherwise composites the face crop onto
        a skin-tone canvas with a soft elliptical falloff.

        When ``face_width``/``face_height`` are provided, the centre-paste
        respects the selfie face aspect ratio for a more natural UV fit.

        Args:
            glb: GLB file bytes
            face_crop: RGB face texture array (H, W, 3)
            skin_rgb: Skin color RGB array (0-255), used for the canvas
                background so the texture blends with the flat-shaded body
            selfie_rgb: Full selfie photo for Delaunay warping (optional)
            landmarks_2d: MediaPipe face-mesh landmarks (optional)
            blend_mode: 'feather' or 'poisson' blending
            face_width, face_height: optional pixel dimensions of the original
                face crop for aspect-ratio-aware texture sizing

        Returns:
            GLB with face texture and normal map applied
        """
        if not HAS_PYGLTFLIB:
            print("Warning: pygltflib not installed, skipping face texture")
            return glb

        try:
            # ── Delaunay-triangulation warping path (MakeHuman UV) ──────
            if selfie_rgb is not None and landmarks_2d is not None:
                from .face_texture_builder import build_head_texture_warped, _FACE_UV_ANCHORS_MAKEHUMAN
                texture_png = build_head_texture_warped(
                    selfie_rgb, landmarks_2d,
                    tuple(int(c) for c in np.clip(skin_rgb[:3], 0, 255)),
                    texture_size=256,
                    blend_mode=blend_mode,
                    uv_anchors=_FACE_UV_ANCHORS_MAKEHUMAN,
                )
                face_texture = Image.open(io.BytesIO(texture_png))
            else:
                # ── Legacy centre-paste + feather mask path ──────────────
                if face_crop is None:
                    # No selfie warp and no face_crop — return unmodified GLB
                    return glb
                if face_crop.dtype != np.uint8:
                    face_crop_uint8 = np.clip(face_crop, 0, 255).astype(np.uint8)
                else:
                    face_crop_uint8 = face_crop

                if face_crop_uint8.ndim == 2:
                    face_img = Image.fromarray(face_crop_uint8, mode='L').convert('RGB')
                else:
                    face_img = Image.fromarray(face_crop_uint8, mode='RGB')

                # Composite the face onto a skin-tone canvas with a soft
                # elliptical mask so it fades into the surrounding skin color.
                canvas_size = 256
                skin_color = tuple(int(c) for c in np.clip(skin_rgb[:3], 0, 255))
                canvas = Image.new('RGB', (canvas_size, canvas_size), skin_color)

                # ✅ Face Crop Aspect Ratio අනුව Target Size එක Adjust කරන්න
                base_face_size = int(canvas_size * 0.7)
                if face_width is not None and face_height is not None and face_width > 0 and face_height > 0:
                    user_aspect = face_width / face_height
                    if user_aspect > 1.0:
                        face_w = base_face_size
                        face_h = int(base_face_size / user_aspect)
                    else:
                        face_h = base_face_size
                        face_w = int(base_face_size * user_aspect)
                    face_w = max(8, face_w)
                    face_h = max(8, face_h)
                else:
                    face_w = face_h = base_face_size

                face_resized = face_img.resize((face_w, face_h))
                offset = ((canvas_size - face_w) // 2, (canvas_size - face_h) // 2)
                face_layer = canvas.copy()
                face_layer.paste(face_resized, offset)

                mask = Image.new('L', (canvas_size, canvas_size), 0)
                margin = int(canvas_size * 0.1)
                ImageDraw.Draw(mask).ellipse(
                    [margin, margin, canvas_size - margin, canvas_size - margin], fill=255
                )
                mask = mask.filter(ImageFilter.GaussianBlur(canvas_size * 0.06))

                face_texture = Image.composite(face_layer, canvas, mask)

            gltf = GLTF2.load_from_bytes(glb)
            blob = bytearray(gltf.binary_blob() or b"")

            gltf.images = gltf.images or []
            gltf.bufferViews = gltf.bufferViews or []
            gltf.textures = gltf.textures or []

            face_tex_idx = _embed_image(gltf, blob, face_texture)
            normal_tex_idx = _embed_image(gltf, blob, _make_skin_normal_map())

            if gltf.buffers:
                gltf.buffers[0].byteLength = len(blob)
            gltf.set_binary_blob(bytes(blob))

            if gltf.materials:
                head_mat = gltf.materials[0]
                if head_mat.pbrMetallicRoughness is None:
                    head_mat.pbrMetallicRoughness = PbrMetallicRoughness()

                head_mat.pbrMetallicRoughness.baseColorTexture = TextureInfo(index=face_tex_idx)
                # baseColorFactor must be white [1,1,1,1] when a texture is present;
                # glTF PBR multiplies the sampled texture by this factor, so any non-white
                # value tints/darkens the face texture and makes it appear invisible.
                head_mat.pbrMetallicRoughness.baseColorFactor = [1.0, 1.0, 1.0, 1.0]
                head_mat.normalTexture = NormalMaterialTexture(index=normal_tex_idx, scale=0.3)

            return b"".join(gltf.save_to_bytes())

        except Exception as e:
            print(f"Warning: Failed to apply face texture: {e}")
            return glb

    def _combine_with_hairstyle(
        self,
        avatar_glb: bytes,
        hairstyle_glb: bytes,
        hair_rgb: tuple,
    ) -> bytes:
        """Combine avatar with hairstyle GLB, recolored to the detected hair color.

        Args:
            avatar_glb: Avatar GLB bytes
            hairstyle_glb: Hairstyle GLB bytes
            hair_rgb: Hair color for material (0-255)

        Returns:
            Combined GLB bytes
        """
        if not HAS_TRIMESH:
            print("WARNING [avatar_builder]: trimesh not installed — cannot combine hairstyle.")
            print("  Install it with: pip install trimesh")
            print("  Returning avatar without hairstyle.")
            return avatar_glb

        try:
            avatar_scene = trimesh.load(io.BytesIO(avatar_glb), file_type='glb')
            hair_scene = trimesh.load(io.BytesIO(hairstyle_glb), file_type='glb')

            if not isinstance(avatar_scene, trimesh.Scene):
                avatar_scene = trimesh.Scene([avatar_scene])

            if isinstance(hair_scene, trimesh.Scene):
                hair_geometries = list(hair_scene.geometry.values())
            else:
                hair_geometries = [hair_scene]

            hair_material_props = self.assets.get_hair_material(hair_rgb)
            hair_material = trimesh.visual.material.PBRMaterial(
                baseColorFactor=hair_material_props['baseColor'],
                roughnessFactor=hair_material_props['roughness'],
                metallicFactor=hair_material_props['metallic'],
            )

            for geometry in hair_geometries:
                try:
                    geometry.visual = trimesh.visual.TextureVisuals(material=hair_material)
                except Exception:
                    hair_rgba = np.array([*hair_rgb[:3], 255], dtype=np.uint8)
                    geometry.visual.vertex_colors = np.tile(hair_rgba, (len(geometry.vertices), 1))
                avatar_scene.add_geometry(geometry)

            result = io.BytesIO()
            avatar_scene.export(result, file_type='glb')
            result.seek(0)
            return result.read()

        except Exception as e:
            print(f"Warning: Failed to combine with hairstyle: {e}")
            return avatar_glb


def create_avatar_builder(use_realistic: bool = False) -> Optional[AvatarBuilder]:
    """Factory function to create avatar builder.

    Args:
        use_realistic: Whether to enable realistic avatars (requires assets)

    Returns:
        AvatarBuilder instance if realistic, None otherwise
    """
    if not use_realistic:
        return None

    # Find assets directory
    import os
    current_dir = os.path.dirname(__file__)
    # Navigate up to project root: model6_body3d -> model6_body3d -> avatar_pipeline -> New avatar
    project_root = os.path.dirname(os.path.dirname(current_dir))
    assets_dir = os.path.join(project_root, "assets")

    if not os.path.exists(assets_dir):
        print(f"Warning: Assets directory not found: {assets_dir}")
        print("Realistic avatars require:")
        print("  - assets/avatars/male/base.glb")
        print("  - assets/avatars/female/base.glb")
        print("  - assets/hair/male/*.glb")
        print("  - assets/hair/female/*.glb")
        return None

    asset_manager = AvatarAssetManager(assets_dir)
    return AvatarBuilder(asset_manager)
