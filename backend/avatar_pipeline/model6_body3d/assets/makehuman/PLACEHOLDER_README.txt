male.glb: derived from assets/avatars/rp/rp_male.fbx (a Renderpeople rigged
character export) via a one-off conversion (three.js FBXLoader in a headless
browser to extract bind-pose positions/UVs, reassembled in Python to match
this pipeline's expected GLB shape). It is a REAL, detailed body mesh
(70209 verts) but:
  - has 6 zero-delta morph targets — it's a fixed scan, not a blendshape rig,
    so the body-shape sliders are a no-op for this asset (shoulders/hips/etc.
    won't visibly resize; same as how armLength/legLength already behave for
    every asset, since the server always sends weight 0.0 for those two).
  - uses a synthetic cylindrical UV, not the source scan's real photo UV —
    the FBX references an external diffuse texture file (sourceimages/*.jpg)
    that isn't bundled with it, and customize-face overwrites the body
    texture unconditionally anyway (makehuman_mesh._write_glb), so the
    original UV layout wouldn't have been usable regardless.
  - bind pose is a wide A/T-pose (visible as a fairly wide bounding box) —
    that's how the rig was exported, not a conversion bug.

female.glb is still the earlier crude placeholder column (no equivalent
female Renderpeople asset was provided). Regenerate via
private/tmp/.../scratchpad/fbx/{run_convert.js,build_rp_male_glb.py} if you
add more source FBX models, or replace both with the real MakeHuman/MPFB-
baked assets from scripts/generate_makehuman_avatars.py for production.
