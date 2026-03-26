"""
eWardrobeAI — Stage 3 & 4: Avatar Manager & 3D Asset Loader

Responsibilities
----------------
1. Maintain the mapping from GarmentRecord.asset_path to physical .glb/.fbx files
2. Build the complete AvatarRenderPayload: a JSON-serialisable object consumed
   by the Three.js / React Three Fiber frontend renderer
3. Integrate AvatarScaleParams (from BodyCalibrator) + FaceProfile (from FaceProcessor)
   + OutfitRecommendation (from NisfaMatchmaking) into a unified scene description
4. Define Mixamo animation metadata for walking, rotating, and pose animations

3D Asset Conventions
--------------------
All avatar .glb files exported from Blender with:
  - Y-up coordinate system
  - Armature origin at world origin (0, 0, 0)
  - Bone hierarchy compatible with Mixamo rig naming (Hips → Spine → … → Head)
  - UV maps present for both body mesh and head mesh (for face texture application)

Clothing .glb files:
  - Exported with blend-shape morph targets for size variation
  - Rigged to the same Mixamo-compatible skeleton
  - Named following convention: {garment_id}_{colour}_{size}.glb
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import logging

from src.body_calibration import AvatarScaleParams
from src.face_processor import FaceProfile
from src.outfit_recommender import OutfitRecommendation, GarmentRecord

logger = logging.getLogger(__name__)

# ── Asset Registry ────────────────────────────────────────────────────────────
ASSET_BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')

# Base avatar .glb file — Blender-rigged, Mixamo-compatible
BASE_AVATAR_GLB = os.path.join(ASSET_BASE_DIR, 'avatars', 'base_avatar.glb')

# Mixamo animation clip names (must match baked animation track names in .glb)
MIXAMO_ANIMATIONS = {
    'idle':     'Mixamo_Idle',
    'walk':     'Mixamo_Walking',
    'rotate':   'Mixamo_TurnLeft',
    'pose_t':   'Mixamo_TPose',
    'pose_a':   'Mixamo_APose',
    'catwalk':  'Mixamo_CatwalkWalk',
}

# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class AnimationConfig:
    """Defines which Mixamo animation to play and its playback settings."""
    clip_name:     str
    loop:          bool  = True
    time_scale:    float = 1.0
    fade_duration: float = 0.5      # cross-fade seconds


@dataclass
class ClothingAsset:
    """
    Single clothing mesh to be layered onto the avatar.
    Maps directly to a Three.js GLTFLoader import.
    """
    garment_id:  str
    asset_path:  str                 # relative path to .glb / .fbx
    asset_exists: bool = False
    morph_target: Optional[str] = None  # blend-shape key for size morphing
    colour_hex:   str = '#FFFFFF'
    layer_order:  int = 0            # rendering depth: 0=base, 1=mid, 2=outer


@dataclass
class AvatarRenderPayload:
    """
    Complete scene description JSON sent to the Three.js renderer.

    The frontend renderer (tryon_renderer.js) deserialises this object and:
      1. Loads the base avatar GLB
      2. Applies bone-level scale transforms from scale_params
      3. Applies the face texture to the head mesh UV map
      4. Loads each clothing asset and attaches it to the skeleton
      5. Plays the default animation
    """
    # Avatar
    avatar_glb_path:  str
    scale_params:     dict                   # AvatarScaleParams.to_dict()

    # Face
    face_texture_b64: Optional[str]          # base64-encoded PNG
    head_yaw_deg:     float = 0.0
    head_pitch_deg:   float = 0.0

    # Outfit
    clothing_assets:  list[ClothingAsset] = field(default_factory=list)
    outfit_name:      str = ''
    outfit_id:        str = ''

    # Animation
    animation:        AnimationConfig = field(
        default_factory=lambda: AnimationConfig(MIXAMO_ANIMATIONS['idle'])
    )

    # Scene
    background_colour: str = '#F5F5F5'
    camera_fov:        float = 45.0
    camera_position:   dict = field(
        default_factory=lambda: {'x': 0, 'y': 1.6, 'z': 3.5}
    )

    def to_dict(self) -> dict:
        return {
            'avatarGlbPath':   self.avatar_glb_path,
            'scaleParams':     self.scale_params,
            'faceTextureB64':  self.face_texture_b64,
            'headYawDeg':      self.head_yaw_deg,
            'headPitchDeg':    self.head_pitch_deg,
            'clothingAssets': [
                {
                    'garmentId':   ca.garment_id,
                    'assetPath':   ca.asset_path,
                    'assetExists': ca.asset_exists,
                    'morphTarget': ca.morph_target,
                    'colourHex':   ca.colour_hex,
                    'layerOrder':  ca.layer_order,
                }
                for ca in self.clothing_assets
            ],
            'outfitName':        self.outfit_name,
            'outfitId':          self.outfit_id,
            'animation': {
                'clipName':     self.animation.clip_name,
                'loop':         self.animation.loop,
                'timeScale':    self.animation.time_scale,
                'fadeDuration': self.animation.fade_duration,
            },
            'scene': {
                'backgroundColour': self.background_colour,
                'cameraFov':        self.camera_fov,
                'cameraPosition':   self.camera_position,
            },
        }


# ── AvatarManager ─────────────────────────────────────────────────────────────

class AvatarManager:
    """
    Builds AvatarRenderPayload objects consumed by the Three.js renderer.

    Usage
    -----
    >>> mgr = AvatarManager()
    >>> payload = mgr.build_render_payload(
    ...     scale_params=scale,
    ...     face_profile=profile,
    ...     outfit=recommendation,
    ...     animation_key='walk',
    ... )
    >>> renderer_json = payload.to_dict()
    """

    def __init__(self, asset_base_dir: str = ASSET_BASE_DIR):
        self._asset_dir = asset_base_dir
        self._verify_asset_dir()

    def _verify_asset_dir(self):
        if not os.path.isdir(self._asset_dir):
            logger.warning(
                f"[AvatarManager] Asset directory not found: {self._asset_dir}. "
                "3D assets will be marked as missing but rendering will proceed "
                "in demo mode using placeholder geometry."
            )

    # ── Main Builder ──────────────────────────────────────────────────────────

    def build_render_payload(
        self,
        scale_params: AvatarScaleParams,
        face_profile: FaceProfile,
        outfit: OutfitRecommendation,
        animation_key: str = 'idle',
    ) -> AvatarRenderPayload:
        """
        Assemble all inputs into a single AvatarRenderPayload.

        Steps
        -----
        1. Resolve avatar GLB path
        2. Encode face texture as base64 PNG
        3. Build ClothingAsset list from outfit items
        4. Select animation config from animation_key
        5. Compose and return AvatarRenderPayload
        """
        # 1. Avatar base model
        avatar_path = self._resolve_path(BASE_AVATAR_GLB)

        # 2. Face texture → base64 PNG
        face_b64 = None
        if face_profile.face_texture is not None:
            face_b64 = self._encode_texture(face_profile.face_texture)

        # 3. Clothing assets
        clothing = self._build_clothing_assets(outfit.items)

        # 4. Animation
        clip_name = MIXAMO_ANIMATIONS.get(
            animation_key, MIXAMO_ANIMATIONS['idle']
        )
        anim_config = AnimationConfig(
            clip_name=clip_name,
            loop=True,
            time_scale=1.0 if animation_key != 'catwalk' else 0.85,
        )

        payload = AvatarRenderPayload(
            avatar_glb_path  = avatar_path,
            scale_params     = scale_params.to_dict(),
            face_texture_b64 = face_b64,
            head_yaw_deg     = face_profile.yaw_deg,
            head_pitch_deg   = face_profile.pitch_deg,
            clothing_assets  = clothing,
            outfit_name      = outfit.name,
            outfit_id        = outfit.outfit_id,
            animation        = anim_config,
        )

        logger.info(
            f"[AvatarManager] Render payload built: outfit='{outfit.name}' "
            f"anim='{clip_name}' garments={len(clothing)}"
        )
        return payload

    # ── Clothing Asset Construction ───────────────────────────────────────────

    def _build_clothing_assets(
        self, items: list[GarmentRecord]
    ) -> list[ClothingAsset]:
        """
        Map each GarmentRecord to a ClothingAsset with resolved file paths.
        Items whose .glb files do not exist on disk are flagged (assetExists=False)
        so the renderer can substitute a placeholder mesh.
        """
        LAYER_MAP = {
            'top':       1,
            'bottom':    1,
            'dress':     1,
            'suit':      1,
            'outerwear': 2,
            'footwear':  0,
            'accessory': 3,
        }

        COLOUR_MAP = {
            'white':        '#FFFFFF',
            'light_blue':   '#ADD8E6',
            'navy':         '#001F5B',
            'black':        '#111111',
            'charcoal':     '#36454F',
            'khaki':        '#C3B091',
            'olive':        '#808000',
            'emerald':      '#50C878',
            'terracotta':   '#E2725B',
            'dark_indigo':  '#1B1464',
            'mid_grey':     '#808080',
            'cobalt_blue':  '#0047AB',
            'burgundy':     '#800020',
            'deep_red':     '#8B0000',
            'champagne':    '#F7E7CE',
        }

        assets: list[ClothingAsset] = []
        for item in items:
            full_path = os.path.join(self._asset_dir, '..', item.asset_path)
            exists    = os.path.isfile(full_path)
            colour_hex = COLOUR_MAP.get(
                item.colours[0] if item.colours else 'black', '#333333'
            )
            layer = LAYER_MAP.get(item.category.value, 1)

            assets.append(ClothingAsset(
                garment_id   = item.garment_id,
                asset_path   = item.asset_path,
                asset_exists = exists,
                colour_hex   = colour_hex,
                layer_order  = layer,
            ))

        # Sort by layer order for correct draw order
        assets.sort(key=lambda a: a.layer_order)
        return assets

    # ── Texture Encoding ──────────────────────────────────────────────────────

    @staticmethod
    def _encode_texture(texture_rgb: np.ndarray) -> str:
        """
        Encode a 512×512 RGB numpy array as a base64-encoded PNG string.
        The frontend loads this as a data URI: 'data:image/png;base64,…'
        """
        try:
            import cv2
            bgr = cv2.cvtColor(texture_rgb, cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode('.png', bgr)
            if success:
                return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.warning(f"[AvatarManager] Texture encoding failed: {e}")
        return ''

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _resolve_path(self, abs_path: str) -> str:
        """Return a web-accessible relative path for use in the frontend."""
        try:
            rel = os.path.relpath(abs_path, start=os.path.join(
                os.path.dirname(__file__), '..'
            ))
            return rel.replace('\\', '/')
        except ValueError:
            return abs_path

    def list_available_animations(self) -> list[str]:
        return list(MIXAMO_ANIMATIONS.keys())

    def get_animation_config(self, key: str) -> AnimationConfig:
        clip = MIXAMO_ANIMATIONS.get(key, MIXAMO_ANIMATIONS['idle'])
        return AnimationConfig(
            clip_name=clip,
            loop=key not in ('pose_t', 'pose_a'),
        )
