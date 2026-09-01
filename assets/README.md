# Phase 2 Avatar Assets Directory

This directory contains 3D models and materials for photorealistic avatar generation (Phase 2).

## Structure

```
assets/
├── avatars/
│   ├── male/
│   │   └── base.glb          # Male base humanoid model
│   └── female/
│       └── base.glb          # Female base humanoid model
├── hair/
│   ├── male/
│   │   ├── short.glb
│   │   ├── buzz.glb
│   │   ├── medium.glb
│   │   ├── textured.glb      # Curly/wavy fallback
│   │   └── slicked_back.glb
│   └── female/
│       ├── pixie_cut.glb
│       ├── short_straight.glb
│       ├── medium_straight.glb
│       ├── long_straight.glb
│       ├── long_wavy.glb
│       ├── long_curly.glb
│       ├── ponytail.glb
│       └── bun.glb
└── materials/                # Material definition templates
```

## Asset Requirements

### Base Avatar Models
- **Format**: glTF 2.0 binary (.glb)
- **Requirements**:
  - Humanoid rig with standard bones (Armature)
  - UVs mapped for head (for face texture application)
  - Material: basic white/neutral skin color
  - Scale: 1 unit = 1 cm
  - Height: ~170cm (male), ~160cm (female)

### Hairstyle Models
- **Format**: glTF 2.0 binary (.glb)
- **Requirements**:
  - Positioned on head of base avatar
  - Pre-weighted to head bone or separate bones
  - Material: basic color
  - No body collision
  - Attachment point: centered on skull

### Hair Color
Hair color is applied procedurally via material:
- Detected from facial analysis: `hair_color` (black, brown, blonde, red, gray)
- Material is modified in memory, not stored as assets

## Phase 2 Implementation Steps

1. **Acquire/Create Base Models**:
   - Male humanoid model (generic anatomy)
   - Female humanoid model (generic anatomy)
   - Can be downloaded from 3D asset marketplaces or created in Blender

2. **Create/Acquire Hairstyles**:
   - Multiple hairstyles per gender
   - Can be free assets from Sketchfab, Turbosquid, or Blender
   - Ensure compatibility with base model (same scale/rig)

3. **Implement Mesh Deformation** in `avatar_builder.py`:
   - Use `trimesh` or `pyvista` to deform mesh based on body3d_params
   - Requires: `pip install trimesh`

4. **Implement Material Application** in `avatar_builder.py`:
   - Use `pygltflib` to modify GLB materials
   - Apply skin/hair colors from facial analysis
   - Requires: `pip install pygltflib`

5. **Implement Face Texture Mapping** in `avatar_builder.py`:
   - Map extracted face crop to head UVs
   - Embed texture in GLB or reference externally

6. **Implement Hairstyle Merging** in `avatar_builder.py`:
   - Combine base avatar + hairstyle into single GLB
   - Position hairstyle on head
   - Merge materials

## How It Works

When `AVATAR_USE_REALISTIC=1`:

1. **Asset Manager** (`asset_loader.py`):
   - Loads base avatar for detected gender
   - Loads matching hairstyle
   - Caches in memory

2. **Avatar Builder** (`avatar_builder.py`):
   - Deforms base model to match detected body parameters
   - Applies skin material (from detected color)
   - Applies hair material (from detected color)
   - Maps face texture to head
   - Combines with hairstyle
   - Returns final GLB

3. **Fallback**:
   - If assets missing: gracefully falls back to Phase 1 procedural
   - If deformation fails: returns base model undeformed

## Environment Variable

Enable Phase 2 in development/production:

```bash
export AVATAR_USE_REALISTIC=1
```

Default (Phase 1 procedural): `AVATAR_USE_REALISTIC=0`

## Performance

- Asset loading: ~100-300ms first load, <1ms cached
- Mesh deformation: ~50-200ms (depends on mesh complexity)
- Material application: ~10-20ms
- Hairstyle merging: ~30-100ms
- Total Phase 2 overhead: ~200-500ms vs Phase 1: ~150ms

## References

- glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- Trimesh documentation: https://trimesh.org/
- Pygltflib documentation: https://github.com/autodesk-forks/gltf-python
