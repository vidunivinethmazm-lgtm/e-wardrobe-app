# Finalize Avatar Flow — Body Sliders → Finalize → Finalized Avatar Page

Trackable plan for trimming the body-customization sliders and adding a "Finalize
avatar" step + new results page. Check items off (`- [ ]` → `- [x]`) as they're
implemented. **No implementation until told to start.**

## Flow summary

```
MaleAvatarScreen | FemaleAvatarScreen
  - Customize body: Shoulders / Arms / Legs / Hips-waist sliders ONLY
  - Live preview: every slider change re-renders AvatarViewer3D immediately
  - "Reset" -> back to original (detected) sizes
  - "Finalize avatar" button -> bakes current slider adjustments into avatarConfig
   -> FinalizedAvatarScreen (new)
        - "Avatar finalized" status
        - Finalized 3D avatar (read-only, no sliders)
        - Cropped face image (existing faceTextureUri) — shown as its own card
        - "Edit body" -> back to MaleAvatar/FemaleAvatar (sliders keep their values)
        - "Next" button
   -> Wardrobe -> TryOn   (unchanged)
```

## Key decisions (confirmed)
- Customize panel keeps **only** the Shoulders, Arms, Legs, Hips/waist proportion
  sliders. The Head-width slider, Height slider, Body-type preset chips, and Skin
  tone slider are **removed** from `MaleAvatarScreen` / `FemaleAvatarScreen`.
- Removed controls' values stay at their *detected* defaults (`heightOffset: 0`,
  `bodyTypeOverride: null`, `skinTone: 0`, `proportionOffsets.headWidth: 0`).
  `applyBodyAdjustments` already falls back to the detected values in this case, so
  no body-scaling logic changes are needed — only UI removal.
- New **"Finalize avatar"** button on `MaleAvatarScreen` / `FemaleAvatarScreen`:
  bakes the current `adjustedConfig` (the live output of `applyBodyAdjustments`)
  into a final `avatarConfig` and navigates to a new **`FinalizedAvatarScreen`**.
- `FinalizedAvatarScreen` shows two separate items: the finalized 3D avatar
  (read-only) and the cropped face image (`avatarConfig.faceTextureUri`), each in
  its own card.
- `FinalizedAvatarScreen` has a **"Next"** button that continues to **Wardrobe**
  (same destination as today's "Browse wardrobe").

### Slider effects (already implemented in `bodyScaling.ts`, no logic changes needed)
| Slider | `ProportionKey` | Effect on avatar |
| --- | --- | --- |
| Shoulders | `shoulderWidth` | Shoulder width / upper body broadens or narrows |
| Arms | `armLength` | Arm length/thickness proportion changes |
| Hips / Waist | `hipWidth` | Hip width and waist shape update |
| Legs | `legLength` | Leg length/thickness proportion changes |

Each slider writes to `adjustments.proportionOffsets[key]` (range -1..1, added to the
detected value and clamped). `adjustedConfig = applyBodyAdjustments(avatarConfig, adjustments)`
is recomputed via `useMemo` on every change, so `AvatarViewer3D` re-renders live —
this already works today and needs no new code, just keeping the existing wiring.

## Screens

### 1. `MaleAvatarScreen` / `FemaleAvatarScreen` — modify
- [x] Trim `PROPORTION_ROWS` to: Shoulders (`shoulderWidth`), Arms (`armLength`),
      Legs (`legLength`), Hips / waist (`hipWidth`). Remove the Head (`headWidth`)
      row.
- [x] Remove the "Height" slider block (`adjustments.heightOffset`).
- [x] Remove the "Body type" preset chip row, and the now-unused
      `BODY_TYPE_PRESETS` / `closestBodyTypePresetIndex` imports.
- [x] Remove the "Skin tone correction" slider block (`adjustments.skinTone`).
- [x] "Reset" link keeps resetting `adjustments` to `DEFAULT_BODY_ADJUSTMENTS`
      (unchanged — only the 4 visible proportion offsets are ever non-zero now).
- [x] Add a **"Finalize avatar"** `GradientButton`, placed after the "Customize
      body" card and before "Browse wardrobe":
  - On press: compute `finalConfig = applyBodyAdjustments(avatarConfig, adjustments)`
    and `navigation.navigate('FinalizedAvatar', { avatar, avatarConfig: finalConfig, remoteAvatarUrl })`.
  - Shown regardless of `remoteAvatarUrl` (even when the customize card is hidden,
    the user still needs a way to reach the finalize page).

### 2. `FinalizedAvatarScreen` (mobile/src/screens/FinalizedAvatarScreen.tsx) — new
- [x] Receives `{ avatar, avatarConfig, remoteAvatarUrl }` via route params — this
      `avatarConfig` already has the finalized (baked-in) body adjustments.
- [x] "Avatar finalized" status badge/text near the top of the screen (e.g. a
      `PillBadge` or `Header` subtitle), confirming the body config is locked.
- [x] Card 1 — "Your finalized avatar": `AvatarViewer3D` with `config={avatarConfig}`
      and `remoteAvatarUrl`; read-only (no sliders, no presets).
- [x] Card 2 — "Avatar face": `Image` from `avatarConfig.faceTextureUri` (same crop
      shown today in the "Avatar face" card on MaleAvatar/FemaleAvatar), displayed
      as its own separate card below the 3D avatar card.
- [x] "Edit body" `GradientButton` (accent variant) -> `navigation.goBack()`, returning
      to MaleAvatar/FemaleAvatar with its slider `adjustments` state untouched (still
      mounted) so the user can keep tweaking from where they left off.
- [x] "Next" `GradientButton` -> `navigation.navigate('Wardrobe', { avatar })`.

## Navigation (mobile/src/navigation/types.ts + App.tsx)
- [x] `RootStackParamList`: add
      `FinalizedAvatar: { avatar: AvatarResponse; avatarConfig: AvatarConfig; remoteAvatarUrl?: string }`.
- [x] Register `FinalizedAvatarScreen` in `App.tsx`'s `Stack.Navigator`.

## Open items to confirm while coding
- Copy/labels for the "Finalize avatar" button and `FinalizedAvatarScreen` (title,
  subtitle, card headings, "Next" button text).
- Whether the "Avatar face" card on `FinalizedAvatarScreen` should be hidden when
  `avatarConfig.faceTextureUri` is empty (mirrors the current conditional on
  MaleAvatar/FemaleAvatar).
