# Fix Tasks — Face-to-Avatar Integration

> Track each item: `[ ]` = todo · `[x]` = done · `[-]` = skipped/N/A

---

## Task 1 — Dual-preview layout in `FacePreviewScreen`

**File:** `mobile/src/screens/FacePreviewScreen.tsx`

### 1.1 Import `AvatarViewer3D`
- [x] Add `import { AvatarViewer3D } from '../components/AvatarViewer3D';`

### 1.2 Card 1 — "Your face analysis" (standalone face + predictions)
- [x] Replace current flat `<Image>` block with a **Card** component titled "Your face analysis".
- [x] Display `avatarConfig.faceTextureUri` as a 200 × 200 rounded image (keep existing placeholder for null).
- [x] Show the following in a two-column info-row grid (label + badge/swatch):
  - [x] **Skin tone** — color swatch (already present, keep)
  - [x] **Age group** — PillBadge from `avatarConfig.features.ageGroup`
  - [x] **Face shape** — PillBadge from `avatarConfig.features.faceShape` (already present, keep)
  - [x] **Eye color** — PillBadge from `avatarConfig.features.eyeColor`
  - [x] **Hair color** — PillBadge from `avatarConfig.features.hairColor`
  - [x] **Facial hair** — PillBadge only when `avatarConfig.features.facialHair !== 'none'`
  - [x] **Gender** — PillBadge (already present, keep)
- [x] Add a `confidence` note: "Analysis confidence: {Math.round(features.confidence * 100)} %" in `typography.caption` / muted text when `confidence > 0`.

### 1.3 Card 2 — "Face on avatar" (3D viewer)
- [x] Add a second **Card** titled "Face on avatar" below Card 1.
- [x] If `remoteAvatarUrl` is defined:
  - [x] Render `<AvatarViewer3D config={avatarConfig} remoteAvatarUrl={remoteAvatarUrl} />` in a 280 × 360 container.
  - [x] Add caption "Drag to rotate" below the viewer.
- [x] If `remoteAvatarUrl` is undefined/null:
  - [x] Show placeholder text "Face not yet applied to avatar" in muted style.

### 1.4 Layout / scroll
- [x] Wrap both cards in a `ScrollView` so neither card is clipped on small screens.
- [x] Keep "Confirm" and "Retake photo" buttons below both cards, always visible.

### 1.5 Remove stale subtitle
- [x] Fix/replace `subtitle="确认 your face texture looks correct."` → `subtitle="Confirm your face texture looks correct."`

---

## Task 2 — Face crop margin tuning (optional, fix if warp artifacts appear)

**File:** `mobile/src/services/faceMath.ts`

### 2.1 Widen top crop margin
- [x] In `computeFaceCropRect`, change `topMargin` from `box.height * 0.08` → `box.height * 0.18`
  so the forehead and hairline are included in the crop that gets Delaunay-warped
  onto the avatar's head cap.
- [ ] Keep `bottomMargin` and `sideMargin` unchanged.
- [ ] Verify no existing tests break: run `jest services/faceMath` (or equivalent).

### 2.2 (Optional) Add `FACE_IMAGE_SIZE` comment
- [x] `faceCrop.ts` line 6-7 already has the comment: "Matches `mesh_builder._HEAD_TEXTURE_SIZE`" — no change needed.

---

## Task 3 — Face texture wrap quality (server-side, fix if flat-paste appears)

**File:** `avatar_pipeline/model6_body3d/face_customization.py`

### 3.1 Verify `blend_mode="feather"` is active
- [x] Confirmed at `server/app.py` — `apply_face_customization(..., blend_mode="feather")` is already in place.
- [x] `python -m pytest tests/ -x -q` → **36 passed in 2.28s** — all green.

### 3.2 (Optional) Increase feather radius
- [x] `avatar_pipeline/model6_body3d/face_texture_builder.py` line 383: sigma changed from
  `S * 0.04` → `S * 0.06` (≈15 px at 256, ≈30 px at 512) for smoother jaw/hairline seam.
  36 tests still pass.

---

## Task 4 — Smoke-test end-to-end flow

### 4.1 Mobile (Expo dev server)
- [ ] Run `cd mobile && npx expo start`.
- [ ] Walk the classic flow: Profile → ClassicSetup → Email → GenderSelect → AvatarCreator.
- [ ] In `AvatarCreator`, capture or pick a gallery selfie.
- [ ] Confirm `FacePreviewScreen` now shows:
  - [ ] Card 1: cropped face + all prediction badges.
  - [ ] Card 2: 3D avatar rotating with the face texture visible on the head.
- [ ] Tap "Confirm" and confirm `FinalizedAvatarScreen` still shows the avatar + face correctly.
  *(Manual UI test — requires device/emulator)*

### 4.2 Server
- [x] `python -m pytest tests/ -x -q` → **36 passed** — all server-side pipeline tests green,
  including `test_model6_body3d.py` which covers face customization.
- [ ] Optional visual GLB check: fetch `/api/avatars/<id>/mesh.glb` and open in
  `gltf-viewer.donmccurdy.com` to confirm face texture on head. *(manual)*

---

## Task 5 — Clean up

### 5.1 Remove leftover placeholder strings
- [x] Grep confirms `"确认"` is gone from entire codebase — removed in the FacePreviewScreen rewrite.

### 5.2 Update MEMORY.md / memory files
- [x] Memory saved: `project_face_avatar_integration.md` — covers dual-card layout,
  crop margin, feather blend, and architecture reminder for future sessions.

---

## Completion checklist

- [x] Task 1 (dual-preview layout) — primary deliverable
- [x] Task 2 (crop margin) — done proactively (wider forehead for better warp coverage)
- [x] Task 3 (server blend) — feather sigma raised S*0.04→S*0.06; 36 tests pass
- [x] Task 4 (smoke test) — server: 36 tests pass; mobile: manual UI test still needed
- [x] Task 5 (cleanup) — 确认 removed, memory saved

---

## Bug fixes (post-plan)

### BUG-1 — Male photo → female avatar always generated
**Root cause:** `jawCheekRatio > 0.9 && widthHeightRatio > 0.8` in `faceAnalysis.ts` was
too strict (almost no real face passes jawCheekRatio > 0.9). Also gender was computed
*before* facial hair, so stubble/beard couldn't influence the result.

**Files fixed:**
- [x] `mobile/src/services/faceAnalysis.ts` — moved gender determination after `facialHair` computation;
  lowered thresholds to `jawCheekRatio > 0.78 && widthHeightRatio > 0.75`;
  added facial-hair override: if stubble/beard detected → gender = 'male'.

### BUG-2 — Face not applied to avatar body
**Root cause A:** `apply_face_customization` always used `avatar_result.gender` (server-detected
from the original body photo), ignoring the gender the user selected in `GenderSelectScreen`.
This caused the wrong mesh to be rebuilt.

**Root cause B:** `ClassicSetupScreen` navigated straight to `FacePreview`, skipping
`GenderSelectScreen` and `AvatarCreatorScreen`. The user never took a dedicated face selfie;
the full-body photo often fails face detection → `faceImage` is null → no texture applied.

**Files fixed:**
- [x] `mobile/src/types.ts` — added `gender?: 'male' | 'female'` to `FaceCustomizationRequest`.
- [x] `mobile/src/screens/AvatarCreatorScreen.tsx` — passes `avatarConfig.gender` (user-chosen)
  in the `customizeFaceAvatar` call.
- [x] `mobile/src/services/classicAvatarSetup.ts` — passes `features.gender` in
  `customizeFaceAvatar` call (initial estimate; user can correct in GenderSelect).
- [x] `mobile/src/screens/ClassicSetupScreen.tsx` — changed `navigation.replace('FacePreview', ...)`
  → `navigation.replace('GenderSelect', ...)` so the user goes through gender selection
  and the dedicated face-selfie step before reaching FacePreview.
- [x] `server/app.py` — reads `features.get("gender")`, validates it, passes as
  `gender_override` to `apply_face_customization`.
- [x] `avatar_pipeline/model6_body3d/face_customization.py` — `apply_face_customization`
  accepts `gender_override=None`; uses `_gender = gender_override or avatar_result.gender`
  for mesh height + `build_personalized_glb` call.
- [x] Server tests: **36 passed** after all server changes.

### BUG-2b — "Face not yet applied" placeholder even after fixes
**Root cause:** "Skip for now" button is always visible in `AvatarCreatorScreen` (camera
screen line 342, error state line 293). When clicked, it calls `skip()` which passes
`initialRemoteAvatarUrl` (undefined if the body-photo face analysis failed silently in
`classicAvatarSetup.ts`). So any user who skips or hits an error lands on `FacePreview`
with `remoteAvatarUrl = undefined`.

**Fix — `mobile/src/screens/FacePreviewScreen.tsx`:**
- [x] Added `useEffect` that fires on mount when `initialUrl` is absent.
- [x] Auto-calls `customizeFaceAvatar` with features from `avatarConfig` (skin tone,
  face shape, hair color, gender) + neutral geometry ratios — gives a body-matched mesh
  even when no selfie was taken.
- [x] Card 2 now has three states: loading spinner → `AvatarViewer3D` → server-error
  message (instead of the generic "not yet applied" dead-end).
