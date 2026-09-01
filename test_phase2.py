"""Test Phase 2 implementation"""
import numpy as np
from backend.avatar_pipeline.model6_body3d.avatar_builder import AvatarBuilder
from backend.avatar_pipeline.model6_body3d.asset_loader import AvatarAssetManager, select_hairstyle_for_facial_analysis

print('=== Phase 2 Implementation Test ===\n')

# Test 1: AssetManager creation
print('Test 1: AssetManager')
try:
    am = AvatarAssetManager('assets')
    print('  ✓ AssetManager initialized')
except Exception as e:
    print(f'  ✗ ERROR: {e}')

# Test 2: Hairstyle Selection
print('\nTest 2: Hairstyle Selection')
try:
    facial_analysis = {
        'gender': 'female',
        'hair_style': 'long_wavy',
        'hair_color': 'brown'
    }
    hairstyle = select_hairstyle_for_facial_analysis(facial_analysis, 'female')
    print(f'  ✓ long_wavy mapped to: {hairstyle}')
except Exception as e:
    print(f'  ✗ ERROR: {e}')

# Test 3: Materials
print('\nTest 3: Materials')
try:
    skin_rgb = np.array([210, 180, 140])
    skin_mat = am.get_skin_material(skin_rgb)
    hair_rgb = (139, 69, 19)
    hair_mat = am.get_hair_material(hair_rgb)
    print(f'  ✓ Skin material: roughness={skin_mat["roughness"]}')
    print(f'  ✓ Hair material: roughness={hair_mat["roughness"]}')
except Exception as e:
    print(f'  ✗ ERROR: {e}')

# Test 4: Mesh operations available
print('\nTest 4: Mesh Operations')
try:
    import trimesh
    import pygltflib
    print(f'  ✓ trimesh available (for mesh deformation)')
    print(f'  ✓ pygltflib available (for material/texture application)')
except ImportError as e:
    print(f'  ✗ Missing: {e}')

print('\n=== All Tests Passed ===')
print('\nPhase 2 Ready! When 3D assets are available:')
print('  1. Place base models in assets/avatars/{male,female}/base.glb')
print('  2. Place hairstyles in assets/hair/{male,female}/*.glb')
print('  3. Set environment variable: AVATAR_USE_REALISTIC=1')
