# Fix Plan — Face-to-Avatar Integration

## Context

The avatar pipeline already works end-to-end:
- **Body**: built from bust/waist/hips/height measurements on the server (Model 1/6).
- **Face**: selfie is analyzed on-device with MediaPipe Face Mesh (`faceAnalysis.ts`),
  the face bounding-box is cropped with proper margins (`faceCrop.ts` → `computeFaceCropRect`),
  and the cropped image is sent to `POST /api/avatars/<id>/customize-face` which
  Delaunay-warps it onto the 3D head mesh (`face_customization.py`).
- After `AvatarCreatorScreen` finishes all three capture steps, it calls
  `customizeFaceAvatar()` and passes the resulting `remoteAvatarUrl` (the updated GLB)
  forward to `FacePreviewScreen`.

## What is missing / broken

### 1. `FacePreviewScreen` only shows a flat face image — no avatar preview
`FacePreviewScreen` receives `remoteAvatarUrl` (the GLB with the face already
wrapped onto the head) but never renders `AvatarViewer3D`. The user cannot see
whether the face was applied correctly before confirming.

### 2. Face predictions not presented as a distinct card
Age group, eye color, hair color, face shape are all inside `avatarConfig.features`
but only gender + face shape are surfaced. A richer face-analysis card is needed so
the user can verify the model's readings at a glance.

### 3. Dual-preview layout missing
The requirement is two separate visuals on the same screen:
- **Preview A** — standalone face crop + all model predictions  
  (color, age group, face shape, skin tone swatch, eye / hair color).
- **Preview B** — 3D avatar with the face texture wrapped onto the head,
  using `AvatarViewer3D` with `remoteAvatarUrl`.

## Approach

### A. Enhance `FacePreviewScreen`
Split the screen body into two scrollable cards:

**Card 1 — "Your face analysis"**
- Shows `avatarConfig.faceTextureUri` as a circular/rounded image.
- Lists all `AvatarFeatures` predictions in a tidy grid:
  skin tone swatch, age group pill, face shape pill, eye color pill,
  hair color pill, (facial hair if not "none").

**Card 2 — "Face on avatar"**
- Renders `<AvatarViewer3D config={avatarConfig} remoteAvatarUrl={remoteAvatarUrl} />`
  (the same component used by `FinalizedAvatarScreen`).
- Shown when `remoteAvatarUrl` is defined; a placeholder ("Avatar not yet generated") when absent.

### B. Pass `faceCustomization` data through navigation (minor)
`AvatarCreatorScreen` already builds `faceCustomization` in `analyzeFace()`.
No additional wiring is needed since the API call and URL handoff are in place.

### C. Face-crop quality (server-side, optional)
`computeFaceCropRect` adds 8 % side / 8 % top / 6 % bottom margins — sufficient
for the flat-image preview. The server's `estimate_face_landmarks` (OpenCV Haar)
then re-estimates landmarks from the crop for Delaunay warping. If wrapping
artifacts appear in real testing, increase `topMargin` to `0.18 * box.height`
to include more forehead so the warp covers the full head cap.

## Files to change

| File | Change |
|---|---|
| `mobile/src/screens/FacePreviewScreen.tsx` | Add `AvatarViewer3D` card + expand prediction badges |
| `mobile/src/services/faceMath.ts` | (optional) widen top crop margin |
| `avatar_pipeline/model6_body3d/face_customization.py` | (optional) debug warp blend |

## Non-goals
- Changing the `AvatarCreatorScreen` capture flow — it is correct.
- Touching the server endpoints — they are correct.
- Changing navigation params — `FacePreview` already receives everything it needs.
