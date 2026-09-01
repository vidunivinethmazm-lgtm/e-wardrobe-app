# Onboarding Flow — Photo → Email → Gender → Avatar

Trackable plan for the new pre-avatar onboarding flow. Check items off (`- [ ]` → `- [x]`)
as they're implemented. **No implementation until told to start.**

## Flow summary

```
ProfileScreen (photo: gallery OR camera, pick one)
   -> EmailScreen (enter email, sent to server)
   -> GenderSelectScreen (pick Male or Female)
   -> AvatarCreatorScreen (existing guided face capture — unchanged)
   -> MaleAvatarScreen | FemaleAvatarScreen   (based on gender choice)
   -> Wardrobe -> TryOn   (unchanged)
```

Key decisions (confirmed):
- Photo step offers **both** "Choose photo" (gallery) and "Take photo" (camera) — user
  only needs to use **one**.
- Gender is **manually selected** by the user (not just face-detected) and this choice
  decides routing + body model (male.glb vs female.glb).
- Email is **submitted to the server** (new endpoint).
- Male/Female avatar pages are **new, separate screens** (not the same `AvatarScreen`
  route reused for both).

## Screens

### 1. `ProfileScreen` (mobile/src/screens/ProfileScreen.tsx) — modify
- [x] Keep existing "Choose photo" (gallery, `ImagePicker.launchImageLibraryAsync`).
- [x] Add "Take photo" button — `ImagePicker.requestCameraPermissionsAsync()` +
      `ImagePicker.launchCameraAsync()`. Either button sets the same `photo` state.
- [x] Keep "Generate my avatar": still runs `createAvatar` + `analyzeFace` + `buildAvatar`
      exactly as today, but on success navigates to **`Email`** (instead of
      `AvatarCreator`), passing `{ avatar, avatarConfig }` through.

### 2. `EmailScreen` (mobile/src/screens/EmailScreen.tsx) — new
- [x] Receives `{ avatar, avatarConfig }` via route params.
- [x] Email text input + basic format validation.
- [x] "Continue": POST email to new server endpoint, then navigate to
      **`GenderSelect`** with `{ avatar, avatarConfig }`.

### 3. `GenderSelectScreen` (mobile/src/screens/GenderSelectScreen.tsx) — new
- [x] Receives `{ avatar, avatarConfig }` via route params.
- [x] Two selectable options: **Male** / **Female**.
- [x] "Continue": override the generated `avatarConfig` with the chosen gender —
  - [x] `gender: chosen`
  - [x] `bodyAsset: BODY_ASSETS[chosen]` (male.glb / female.glb)
  - [x] `bodyScale: computeBodyScale(DEFAULT_MEASUREMENTS, chosen, features.faceShape)`
  - [x] `features.gender: chosen` (keep features consistent with the override)
- [x] Navigate to **`AvatarCreator`** with `{ avatar, avatarConfig: updatedConfig }`.
- [x] `DEFAULT_MEASUREMENTS` (currently private to `ProfileScreen`) needs to move
      somewhere shared (e.g. `services/avatarBuilder.ts`) so this screen can use it too.

### 4. `AvatarCreatorScreen` (mobile/src/screens/AvatarCreatorScreen.tsx) — small change
- [x] No change to capture/validation/face-analysis logic.
- [x] The two `navigation.replace('Avatar', ...)` calls (`skip()` and `analyze()`)
      become `navigation.replace(avatarConfig.gender === 'male' ? 'MaleAvatar' : 'FemaleAvatar', ...)`.

### 5. `MaleAvatarScreen` / `FemaleAvatarScreen` — new
- [x] `mobile/src/screens/MaleAvatarScreen.tsx` and `FemaleAvatarScreen.tsx`, each
      starting as a copy of today's `AvatarScreen.tsx` (3D viewer, body customization
      sliders, detected features, body shape, skin tone, "Browse wardrobe" / "Start over").
- [x] Registered as separate routes so each can diverge later (copy, presets, wardrobe
      sets per gender).
- [x] Remove the old `AvatarScreen.tsx` once both new screens are in place.

## Navigation (mobile/src/navigation/types.ts + App.tsx)
- [x] `RootStackParamList`: add `Email`, `GenderSelect`, `MaleAvatar`, `FemaleAvatar`
      (each carrying `{ avatar, avatarConfig, remoteAvatarUrl? }` where relevant);
      remove `Avatar`.
- [x] `Email` / `GenderSelect` params: `{ avatar: AvatarResponse; avatarConfig: AvatarConfig }`.
- [x] `Wardrobe` / `TryOn` params unchanged (still take `avatar`).
- [x] Register `EmailScreen`, `GenderSelectScreen`, `MaleAvatarScreen`,
      `FemaleAvatarScreen` in `App.tsx`'s `Stack.Navigator`; remove `AvatarScreen`.

## Server (server/app.py)
- [x] New endpoint `POST /api/users/email` — body `{ "email": "..." }`, validates format,
      persists it (lightweight storage, e.g. append to a JSONL file — new
      `server/users.py` storage module alongside `server/storage.py`).
- [x] `mobile/src/api/client.ts`: new `submitEmail(email: string): Promise<void>`.

## Open items to confirm while coding
- Exact email persistence format/location on the server (flat file vs. something else).
- Copy/labels for the Email and Gender Select screens (titles, button text).
