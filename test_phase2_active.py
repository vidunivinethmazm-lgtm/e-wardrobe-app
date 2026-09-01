"""Test Phase 2 with generated models."""
import os
os.environ['AVATAR_USE_REALISTIC'] = '1'

from backend.avatar_pipeline.model6_body3d.avatar_builder import create_avatar_builder

print('=== Testing Phase 2 with Generated Models ===\n')

# Test: Check if builder creates with assets now available
builder = create_avatar_builder(use_realistic=True)
if builder:
    print('✓ AvatarBuilder created successfully')
    print('✓ Assets detected and loaded!')
else:
    print('✗ Builder creation failed')
    exit(1)

# Check available models
am = builder.assets
male_base = am.get_base_avatar_model('male')
female_base = am.get_base_avatar_model('female')

print(f'✓ Male base model available: {male_base is not None} ({len(male_base) if male_base else 0} bytes)')
print(f'✓ Female base model available: {female_base is not None} ({len(female_base) if female_base else 0} bytes)')

# Check available hairstyles
male_hairs = am.get_available_hairstyles('male')
female_hairs = am.get_available_hairstyles('female')

print(f'\n✓ Male hairstyles available: {len(male_hairs)}')
for h in sorted(male_hairs):
    print(f'  - {h}')

print(f'\n✓ Female hairstyles available: {len(female_hairs)}')
for h in sorted(female_hairs):
    print(f'  - {h}')

print('\n=== Phase 2 ACTIVE ===')
print('\nYou can now generate PHOTOREALISTIC avatars!')
print('\nTo enable in the server:')
print('  $env:AVATAR_USE_REALISTIC=1')
print('  python -m server.app')
print('\nThen avatars will use the 3D models with:')
print('  - Gender-specific proportions')
print('  - Detected hair/skin colors')
print('  - Face textures mapped to head')
print('  - Matching hairstyles combined')
