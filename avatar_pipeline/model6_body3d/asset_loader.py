"""
Asset loader for realistic avatar generation (Phase 2).

Manages loading and caching of GLB models, hairstyles, and materials.
Enables photorealistic avatars by combining base humanoid models with
hairstyles and realistic textures.

Usage:
    asset_manager = AvatarAssetManager("assets")
    base_glb = asset_manager.get_base_avatar_model("female")
    hair_glb = asset_manager.get_hairstyle_model("female", "long_wavy")
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np


class AvatarAssetManager:
    """Manages loading and caching of avatar assets (models, hairstyles, materials)."""
    
    def __init__(self, assets_dir: str):
        """Initialize asset manager with path to assets directory.
        
        Args:
            assets_dir: Path to /assets directory containing:
                - avatars/male/base.glb
                - avatars/female/base.glb
                - hair/male/*.glb
                - hair/female/*.glb
        """
        self.assets_dir = Path(assets_dir)
        self.avatars_dir = self.assets_dir / "avatars"
        self.hair_dir = self.assets_dir / "hair"
        self.materials_dir = self.assets_dir / "materials"
        
        # Cache loaded models in memory
        self._model_cache = {}
        self._material_cache = {}
        
        # Load hairstyle mapping
        self.hairstyle_map = self._load_hairstyle_map()
    
    def _load_hairstyle_map(self) -> Dict[str, Dict[str, str]]:
        """Load mapping of hairstyles to GLB file paths.
        
        Returns:
            Dict mapping gender -> hairstyle -> file path:
            {
              'male': {'short': '/path/to/short.glb', ...},
              'female': {'long_wavy': '/path/to/long_wavy.glb', ...}
            }
        """
        hair_map = {}
        
        if not self.hair_dir.exists():
            print(f"Warning: Hair directory not found: {self.hair_dir}")
            return hair_map
        
        for gender_dir in self.hair_dir.glob('*'):
            if gender_dir.is_dir():
                gender = gender_dir.name
                hair_map[gender] = {}
                
                for glb_file in gender_dir.glob('*.glb'):
                    style_name = glb_file.stem  # filename without .glb extension
                    hair_map[gender][style_name] = str(glb_file)
        
        return hair_map
    
    def get_base_avatar_model(self, gender: str) -> Optional[bytes]:
        """Load base avatar model GLB for gender.
        
        Args:
            gender: 'male', 'female', or 'neutral' (defaults to male if neutral)
            
        Returns:
            GLB file bytes, or None if not found
            
        Raises:
            ValueError: If gender is invalid (not male/female/neutral)
        """
        if gender not in ['male', 'female', 'neutral']:
            raise ValueError(f"Invalid gender: {gender}. Must be 'male', 'female', or 'neutral'")
        
        # Map neutral to male by default
        if gender == 'neutral':
            gender = 'male'
        
        cache_key = f"base_{gender}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        model_path = self.avatars_dir / gender / "base.glb"
        if not model_path.exists():
            print(f"Warning: Base avatar model not found: {model_path}")
            print(f"  Expected location: {model_path}")
            print(f"  Create male/female base GLB models in {self.avatars_dir}/")
            return None
        
        with open(model_path, 'rb') as f:
            glb_data = f.read()
            self._model_cache[cache_key] = glb_data
        
        return self._model_cache[cache_key]
    
    def get_hairstyle_model(self, gender: str, hair_style: str) -> Optional[bytes]:
        """Load hairstyle GLB model.
        
        Args:
            gender: 'male' or 'female'
            hair_style: style name from facial_analysis
            
        Returns:
            GLB file bytes or None if not found
        """
        if gender not in self.hairstyle_map:
            return None
        
        # If exact style not found, use first available for gender
        if hair_style not in self.hairstyle_map[gender]:
            available_styles = list(self.hairstyle_map[gender].keys())
            if not available_styles:
                return None
            hair_style = available_styles[0]
            print(f"Hair style '{hair_style}' not found, using '{hair_style}'")
        
        cache_key = f"hair_{gender}_{hair_style}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        model_path = self.hairstyle_map[gender][hair_style]
        if not Path(model_path).exists():
            print(f"Warning: Hairstyle model not found: {model_path}")
            return None
        
        with open(model_path, 'rb') as f:
            glb_data = f.read()
            self._model_cache[cache_key] = glb_data
        
        return self._model_cache[cache_key]
    
    def get_available_hairstyles(self, gender: str) -> list:
        """Get list of available hairstyles for gender.
        
        Args:
            gender: 'male' or 'female'
            
        Returns:
            List of hairstyle names
        """
        return list(self.hairstyle_map.get(gender, {}).keys())
    
    def get_skin_material(self, skin_rgb: np.ndarray) -> Dict[str, Any]:
        """Get skin material properties for RGB color.
        
        Args:
            skin_rgb: (3,) array or (r, g, b) values, 0-255
            
        Returns:
            Material properties dict for glTF material
        """
        # Convert to 0-1 range
        r, g, b = [c / 255.0 for c in skin_rgb[:3]]
        
        return {
            'baseColor': [r, g, b, 1.0],
            'roughness': 0.6,        # Skin is somewhat matte
            'metallic': 0.0,
            'subsurface': 0.05,      # Subsurface scattering for realism
            'name': 'Skin_Material'
        }
    
    def get_hair_material(self, hair_rgb: tuple) -> Dict[str, Any]:
        """Get hair material properties for RGB color.
        
        Args:
            hair_rgb: (r, g, b) tuple, 0-255
            
        Returns:
            Material properties dict for glTF material
        """
        r, g, b = [c / 255.0 for c in hair_rgb]
        
        return {
            'baseColor': [r, g, b, 1.0],
            'roughness': 0.7,        # Hair is rougher than skin
            'metallic': 0.0,
            'anisotropic': 0.3,      # Hair has anisotropic properties
            'name': 'Hair_Material'
        }
    
    def clear_cache(self):
        """Clear model cache to free memory."""
        self._model_cache.clear()


def select_hairstyle_for_facial_analysis(facial_analysis: Dict[str, str], gender: str) -> str:
    """Select best hairstyle from facial_analysis for gender.
    
    Maps detected hair style to available GLB assets.
    
    Args:
        facial_analysis: Result dict from extract_facial_analysis()
        gender: 'male', 'female', or 'neutral' (defaults to male if neutral)
        
    Returns:
        Hairstyle name to load (e.g., 'long_wavy', 'short')
    """
    detected_style = facial_analysis.get('hair_style', 'short')
    
    # Map neutral gender to male
    if gender == 'neutral':
        gender = 'male'
    
    # Map detected styles to common asset names
    if gender == 'male':
        style_map = {
            'short': 'short',
            'buzz': 'buzz',
            'curly': 'textured',
            'wavy': 'textured',
            'long': 'textured',
            'medium': 'medium',
        }
    else:  # female
        style_map = {
            'short': 'pixie_cut',
            'medium': 'medium_straight',
            'long': 'long_straight',
            'curly': 'long_curly',
            'wavy': 'long_wavy',
            'straight': 'long_straight',
            'ponytail': 'ponytail',
            'buzz': 'pixie_cut',
        }
    
    return style_map.get(detected_style, 'short')
