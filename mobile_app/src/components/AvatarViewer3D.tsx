import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';
import { ExpoWebGLRenderingContext, GLView } from 'expo-gl';
import { Renderer, TextureLoader } from 'expo-three';
import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, PanResponder, Platform, StyleSheet, Text, View } from 'react-native';
import {
  AmbientLight,
  Box3,
  BufferAttribute,
  BufferGeometry,
  ClampToEdgeWrapping,
  Color,
  DataTexture,
  DirectionalLight,
  Group,
  LinearFilter,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Object3D,
  OrthographicCamera,
  PerspectiveCamera,
  PlaneGeometry,
  SRGBColorSpace,
  Scene,
  Texture,
  Vector3,
  WebGLRenderTarget,
} from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

// `three`'s own `TextureLoader` decodes images via DOM APIs (`Image`,
// `URL.createObjectURL`) that don't exist in React Native's JS engine, so it
// silently fails to load `faceUri` on a phone (Expo Go / dev build) - the
// face texture never resolves even though no error surfaces. `expo-three`
// exports a patched `TextureLoader` that resolves the asset through
// `expo-asset-utils` and `Image.getSize`, uploading pixel data directly via
// `gl.texImage2D` - that's the one `buildHeadTexture` below needs.

import type { AvatarConfig, MorphTargetName } from '../types';
import { colors, radii, spacing, typography } from '../theme';

interface Props {
  /** Realistic avatar configuration from `services/avatarBuilder.ts`'s `buildAvatar()`. */
  config: AvatarConfig;
  /**
   * If set, a remote Avaturn-exported GLB URL. When present, this single GLB
   * is loaded and displayed as-is (centered/scaled/pivoted), and the local
   * body+tint+morph composition driven by `config` is skipped entirely.
   */
  remoteAvatarUrl?: string;
  /**
   * URL to the server-baked face texture PNG (`/api/avatars/<id>/face-texture.png`).
   * When present alongside `remoteAvatarUrl`, loaded directly via expo-three's
   * TextureLoader and applied to every mesh material — bypasses both the GLB's
   * embedded bufferView image (which three.js/GLTFLoader can't decode on native)
   * and the client-side WebGLRenderTarget compositing path (which breaks with
   * expo-gl@16 / three@0.166).
   */
  remoteTextureUrl?: string;
  /**
   * URLs of one or more garment `.glb`s (`/api/avatars/<id>/garment.glb?garment_id=...`,
   * see `garment_mesh.py`) — e.g. a t-shirt + pants worn together, or a single
   * dress. Each is built from the *same* body3d_params + height as the
   * avatar's own mesh, so it's already aligned in the same local space —
   * loaded and added as siblings of the body inside the same pivot `Group`,
   * no extra positioning needed. Their material is a flat `baseColorFactor`
   * (no embedded texture), so no TextureLoader work-around is needed here
   * the way `remoteTextureUrl` needs one.
   */
  garmentMeshUrls?: string[];
  /**
   * Texture atlas URLs parallel to `garmentMeshUrls` (same index, `null`/
   * missing entries mean "no texture for this garment" — e.g. the legacy
   * catalog garments, which are flat-color only). Loaded out-of-band via
   * expo-three's `TextureLoader` (same workaround `remoteTextureUrl` uses)
   * and applied to that garment's mesh, so an uploaded garment photo (e.g.
   * a red t-shirt) actually renders with its own photo instead of a flat
   * placeholder tint.
   */
  garmentTextureUrls?: (string | null | undefined)[];
  /**
   * When true, each `garmentMeshUrls` entry is scaled/positioned onto the
   * body via `fitGarmentToBody` instead of being added as-is — for a
   * standalone catalog garment GLB (e.g. `TSHIRT_ASSET`) that was never
   * built to align with any particular body, unlike the server-built
   * per-avatar garment GLBs `WardrobeScreen` uses (which must NOT be
   * refitted, since their alignment is already exact). Defaults to false.
   */
  autoFitGarments?: boolean;
  /**
   * Live-adjustable fit for `autoFitGarments` garments (e.g. MaleAvatarScreen's
   * sleeve-length / top-length sliders) — see `GarmentFitOptions`. Re-applied
   * to the already-loaded garment meshes whenever it changes, without
   * reloading any GLB. Ignored when `autoFitGarments` is false.
   */
  garmentFit?: GarmentFitOptions;
  /**
   * Called once garment loading finishes, so screens like `WardrobeScreen`
   * can show a clear success/error state instead of silently leaving a
   * naked avatar on a failed load.
   */
  onGarmentStatus?: (status: 'ready' | 'error', message?: string) => void;
  /**
   * A fabric photo to paint directly onto the body's own leg geometry (see
   * `applyLowerBodyFabric`) — there's no separate bottoms garment glb, so
   * "shorts"/"trousers" are done by masking the body's own triangles rather
   * than overlaying a mesh. `null`/`undefined` removes it. Re-applied live
   * whenever this or `bottomCoverage` changes, without reloading any GLB.
   */
  bottomTextureUri?: string | null;
  /** 0 (short shorts) .. 1 (full-length trousers) — how far down from the
   * hip `bottomTextureUri` is painted. Defaults to 0.5. */
  bottomCoverage?: number;
  /** Called after each `bottomTextureUri`/`bottomCoverage` application. */
  onBottomFabricStatus?: (status: 'ready' | 'error', message?: string) => void;
  /**
   * A fabric photo to paint directly onto the body's own torso + short-
   * sleeve geometry (see `applyUpperBodyFabric`) — an alternative to
   * `garmentMeshUrls`/`TSHIRT_ASSET` for a body whose T-pose doesn't match
   * what that separate garment mesh was fitted against (its sleeves either
   * fall short of the arm or gap open at the side — see that function's doc
   * comment). `null`/`undefined` removes it. Re-applied live without
   * reloading any GLB.
   */
  topFabricTextureUri?: string | null;
  /** Called after each `topFabricTextureUri` application. */
  onTopFabricStatus?: (status: 'ready' | 'error', message?: string) => void;
}

// Drag-to-rotate sensitivity: pixels of horizontal pan per radian.
const ROTATE_SPEED = 0.01;

async function loadGlbScene(source: number | string): Promise<Object3D> {
  let uri: string;
  if (typeof source === 'number') {
    const asset = Asset.fromModule(source);
    await asset.downloadAsync();
    uri = asset.localUri ?? asset.uri;
  } else {
    uri = source;
  }
  const gltf = await new GLTFLoader().loadAsync(uri);
  return gltf.scene;
}

/** Tints every mesh material in `root` with `rgb` (0-1). */
function applyTint(root: Object3D, rgb: [number, number, number]) {
  root.traverse((obj) => {
    if (!(obj instanceof Mesh)) return;

    const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
    const tinted = materials.map((mat) => {
      const cloned = mat.clone() as MeshStandardMaterial;
      cloned.color.setRGB(rgb[0], rgb[1], rgb[2]);
      cloned.needsUpdate = true;
      return cloned;
    });

    obj.material = Array.isArray(obj.material) ? tinted : tinted[0];
  });
}

/** Re-tints `root`'s mesh material clones in place with `rgb` (0-1). Mutates
 * the clones `applyTint` already made - no new allocations, safe to call on
 * every slider-drag tick. */
function retintBody(root: Object3D, rgb: [number, number, number]) {
  root.traverse((obj) => {
    if (!(obj instanceof Mesh)) return;
    const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
    for (const mat of materials) {
      const material = mat as MeshStandardMaterial;
      material.color.setRGB(rgb[0], rgb[1], rgb[2]);
      material.needsUpdate = true;
    }
  });
}

// Side length (px) of the composited head texture (matches the convention
// avatar_pipeline/model6_body3d/avatar_builder.py's _apply_face_texture uses
// for its equivalent server-side canvas).
const HEAD_TEXTURE_SIZE = 256;

// Fraction of HEAD_TEXTURE_SIZE the face photo occupies, centered - leaves a
// border of flat skin color so the photo blends into the surrounding head/
// body instead of reading as a hard-edged sticker.
const FACE_SCALE = 0.7;

let _radialAlphaMask: DataTexture | null = null;

/** A square, center-opaque/edge-transparent alpha mask (radial falloff),
 * used so the face photo's edges fade into the skin-tone background instead
 * of showing a hard rectangle. Built once and cached - it doesn't depend on
 * any per-avatar input. */
function getRadialAlphaMask(size = 128): DataTexture {
  if (_radialAlphaMask) return _radialAlphaMask;

  const data = new Uint8Array(size * size);
  const center = (size - 1) / 2;
  // Fully opaque inside ~60% of the radius, fading to transparent at the edge.
  const innerRadius = center * 0.6;
  const outerRadius = center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dist = Math.hypot(x - center, y - center);
      const t = (dist - innerRadius) / (outerRadius - innerRadius);
      const alpha = 255 * (1 - Math.min(1, Math.max(0, t)));
      data[y * size + x] = alpha;
    }
  }

  const texture = new DataTexture(data, size, size, undefined, undefined, undefined, undefined, undefined, LinearFilter, LinearFilter);
  texture.needsUpdate = true;
  _radialAlphaMask = texture;
  return texture;
}

/**
 * Composites `faceUri`'s photo onto a `skinRgb`-colored square, feathered at
 * the edges, and renders the result to a texture sized for the body model's
 * head UV island (see model6_body3d's spherical head UV / single-texel body
 * UV scheme, mirrored here: the body's pinned UV texel at (0.95, 0.95) reads
 * the flat skin-color background, while the head's full-UV spherical
 * projection shows the centered face photo).
 *
 * Uses an offscreen Three.js render (orthographic camera + WebGLRenderTarget)
 * rather than a 2D canvas API, since none is available in this RN/Expo setup.
 */
async function buildHeadTexture(
  renderer: InstanceType<typeof Renderer>,
  faceUri: string,
  skinRgb: [number, number, number]
): Promise<Texture> {
  const faceTexture = await new TextureLoader().loadAsync(faceUri);
  faceTexture.wrapS = ClampToEdgeWrapping;
  faceTexture.wrapT = ClampToEdgeWrapping;

  const scene = new Scene();
  const camera = new OrthographicCamera(0, 1, 1, 0, -1, 1);

  const background = new Mesh(
    new PlaneGeometry(1, 1),
    new MeshBasicMaterial({ color: new Color(skinRgb[0], skinRgb[1], skinRgb[2]) })
  );
  background.position.set(0.5, 0.5, 0);
  scene.add(background);

  const facePlane = new Mesh(
    new PlaneGeometry(FACE_SCALE, FACE_SCALE),
    new MeshBasicMaterial({ map: faceTexture, alphaMap: getRadialAlphaMask(), transparent: true })
  );
  facePlane.position.set(0.5, 0.5, 0.1);
  scene.add(facePlane);

  const target = new WebGLRenderTarget(HEAD_TEXTURE_SIZE, HEAD_TEXTURE_SIZE);
  const prevTarget = renderer.getRenderTarget();
  renderer.setRenderTarget(target);
  renderer.render(scene, camera);
  renderer.setRenderTarget(prevTarget);

  return target.texture;
}

/** Applies a face texture to all meshes in `root`. MeshBasicMaterial renders
 * the texture without PBR lighting — body vertices are pinned to UV (0.95,0.95)
 * which samples the skin-tone corner of the texture, so the body stays
 * skin-colored while head vertices (UV ~0.5,0.5) show the face photo. */
function applyFaceTexture(root: Object3D, texture: Texture): number {
  let meshCount = 0;
  root.traverse((obj) => {
    if (!(obj instanceof Mesh)) return;
    meshCount++;
    const mat = new MeshBasicMaterial({ map: texture });
    mat.needsUpdate = true;
    obj.material = mat;
  });
  console.log(`[AvatarViewer3D] applyFaceTexture: applied to ${meshCount} meshes`);
  return meshCount;
}

/** Loads `textureUri` as a `Texture` ready to paint onto a cylindrical-UV
 * mesh (downloading to a local file first on native, same as
 * `remoteTextureUrl`'s workaround, since a plain `http(s)://` URL can't be
 * fetched reliably by expo-three's `TextureLoader` on device). Shared by
 * `applyGarmentTexture` (upper-body garment overlay) and
 * `applyLowerBodyFabric` (fabric painted directly onto the body's own leg
 * geometry, since there's no separate bottoms garment glb). */
async function loadFabricTexture(textureUri: string): Promise<Texture> {
  let uri = textureUri;
  // Only http(s) URLs need the download-to-cache workaround — a local
  // file://, data: or blob: URI (e.g. a fabric image straight from the
  // image picker) is already loadable by expo-three's TextureLoader.
  if (Platform.OS !== 'web' && /^https?:/.test(textureUri)) {
    const localPath = (FileSystem.cacheDirectory ?? '') + `garment_tex_${Date.now()}.png`;
    const dl = await FileSystem.downloadAsync(textureUri, localPath);
    if (dl.status !== 200) throw new Error(`HTTP ${dl.status}`);
    uri = dl.uri;
  }
  const texture = await new TextureLoader().loadAsync(uri);
  texture.colorSpace = SRGBColorSpace;
  // Cylindrical UVs span [0,1] in U (full horizontal wrap). Use RepeatWrapping
  // so the seam doesn't produce a hard clamp artefact, and disable flipY so
  // the V=0 bottom/V=1 top orientation matches our UV projection.
  texture.wrapS = 1000; // THREE.RepeatWrapping
  texture.wrapT = 1000;
  texture.flipY = false;
  texture.needsUpdate = true;
  return texture;
}

/** Loads `textureUri` and applies it as every mesh's `MeshStandardMaterial.map`
 * in `root` — see `loadFabricTexture` for the cylindrical-UV assumption this
 * relies on. Throws on failure so the caller can report a clear per-garment
 * error instead of silently leaving the flat placeholder tint. */
async function applyGarmentTexture(root: Object3D, textureUri: string): Promise<void> {
  const texture = await loadFabricTexture(textureUri);

  let meshCount = 0;
  root.traverse((obj) => {
    if (!(obj instanceof Mesh)) return;
    meshCount++;
    // MeshStandardMaterial picks up the scene lights (key + ambient) set up in
    // onContextCreate, giving the garment realistic shading and depth instead
    // of the flat look of MeshBasicMaterial.
    const mat = new MeshStandardMaterial({
      map: texture,
      metalness: 0.0,
      roughness: 0.85,
    });
    mat.needsUpdate = true;
    obj.material = mat;
  });
  console.log(`[AvatarViewer3D] applyGarmentTexture: applied to ${meshCount} mesh(es)`);
}

// A standalone garment GLB (e.g. `TSHIRT_ASSET`) is exported from whatever
// scene the artist modeled it in — its own bounding box has no relationship
// to the body it's meant to be worn on. These constants describe an
// upper-body garment (crew-neck t-shirt) as a fraction of the wearer's
// standing height: COLLAR_ANCHOR is where the collar/shoulder line sits
// (fixed regardless of length adjustments below), HEIGHT_FRACTION is the
// default collar-to-hem span — calibrated by inspecting tshirt_model.glb's
// actual proportions against assets/avatars/rp/male_base_mesh.glb.
const UPPER_GARMENT_COLLAR_ANCHOR = 0.82;
const UPPER_GARMENT_HEIGHT_FRACTION = 0.32;

// male_base_mesh.glb's own T-pose width-to-height ratio — the body the
// sleeveScale=1 default was tuned against. assets/avatars/rp/*.glb bodies
// don't share a pose convention: female_base_mesh.glb's arms spread ~71%
// wider relative to its height, so fitting it with the same fixed sleeve
// scale leaves the sleeves stopping short of the arms entirely (visible as
// the shirt riding up at the shoulder with a gap of bare skin down the arm/
// side). Scaling sleeveScale by (this body's own ratio / this reference)
// makes the sleeve reach adapt to whatever body it's actually applied to,
// instead of a constant that only happens to fit the one body it was
// eyeballed against.
const REFERENCE_BODY_WIDTH_HEIGHT_RATIO = 0.9401262490423112 / 1.8115880554989434;

interface GarmentFitOptions {
  /** Garment length (collar to hem) as a fraction of the wearer's standing
   * height. Larger = the hem drops lower (oversized); smaller = a crop top.
   * The collar/shoulder line itself stays anchored in place either way. */
  heightFraction?: number;
  /** Extra multiplier applied only to the garment's horizontal (sleeve-span)
   * axis, on top of `heightFraction`'s uniform scale — this mesh has no
   * skeleton to bend a sleeve along the arm, so "sleeve length" is
   * approximated as how far the (rigid, T-posed) sleeve reaches sideways. */
  sleeveScale?: number;
}

/** Rescales and repositions `garment` (in place) so its bounding box sits on
 * `bodyBox` the way an upper-body garment (t-shirt) would be worn, using
 * only the two bounding boxes — no assumption about the garment's own
 * export scale/offset, since GLBs like `TSHIRT_ASSET` aren't pre-aligned to
 * any particular body. Recomputes from `garment`'s raw (untransformed)
 * geometry every call rather than adjusting the current scale/position, so
 * it can be re-run any number of times with new `options` (e.g. from a
 * slider) without drifting off the body. */
function fitGarmentToBody(garment: Object3D, bodyBox: Box3, options: GarmentFitOptions = {}) {
  const { heightFraction = UPPER_GARMENT_HEIGHT_FRACTION, sleeveScale: sleeveScaleMultiplier = 1 } = options;

  garment.scale.set(1, 1, 1);
  garment.position.set(0, 0, 0);

  const bodySize = bodyBox.getSize(new Vector3());
  const bodyCenter = bodyBox.getCenter(new Vector3());

  const rawBox = new Box3().setFromObject(garment);
  const rawSize = rawBox.getSize(new Vector3());
  if (rawSize.y <= 0) return;

  // Full correction (matching the reference body's reach exactly) stretches
  // this rigid, unskinned mesh enough to pull its armhole/side-seam away
  // from the body's actual armpit on a much-wider-posed body, opening a
  // visible gap at the side/back — worse than the sleeve falling a bit
  // short. Only half-correct the mismatch as a compromise between the two
  // failure modes, since there's no perfect fix without a garment mesh that
  // can actually deform to the pose.
  const bodyWidthHeightRatio = bodySize.x / bodySize.y;
  const rawAutoSleeveScale = bodyWidthHeightRatio / REFERENCE_BODY_WIDTH_HEIGHT_RATIO;
  const autoSleeveScale = 1 + (rawAutoSleeveScale - 1) * 0.5;
  const sleeveScale = sleeveScaleMultiplier * autoSleeveScale;

  const scaleY = (heightFraction * bodySize.y) / rawSize.y;
  garment.scale.set(scaleY * sleeveScale, scaleY, scaleY);

  const scaledBox = new Box3().setFromObject(garment);
  const scaledCenter = scaledBox.getCenter(new Vector3());
  const collarY = bodyBox.min.y + UPPER_GARMENT_COLLAR_ANCHOR * bodySize.y;

  garment.position.x += bodyCenter.x - scaledCenter.x;
  garment.position.y += collarY - scaledBox.max.y;
  garment.position.z += bodyCenter.z - scaledCenter.z;
}

// Fraction of standing height a hairstyle's own bounding box is rescaled to
// span, crown down — generous enough to cover both short cuts (which then
// just have empty margin at the bottom of that span) and long hair.
const HAIR_HEIGHT_FRACTION = 0.22;

/** Rescales and repositions `hair` (in place) so it sits on top of `bodyBox`
 * — its top aligned with the crown of the head, centered on the body — using
 * only the two bounding boxes, the same technique `fitGarmentToBody` uses.
 *
 * Why this is necessary: `assets/hair/*.glb` are authored in a completely
 * different unit scale from `assets/avatars/rp/*.glb` (measured directly:
 * body spans Y 0..1.81, hair spans Y 148..161 in its own raw file — roughly
 * a 100x mismatch, centimeters vs. meters). Adding hair to the body's group
 * unscaled and letting `centerAndPivot` compute a bounding box over both
 * blows `maxDim` up to ~160, which then scales the ENTIRE model (body
 * included) down by roughly that same factor — the body shrinks to an
 * imperceptible speck while the (relatively now much larger) hair becomes
 * the only visible thing, looking like a small stray blob floating on
 * screen. This was never caught before because the server-built
 * `remoteAvatarUrl` mesh always took priority over this local composition
 * path, until it was turned off. */
function fitHairToBody(hair: Object3D, bodyBox: Box3) {
  hair.scale.set(1, 1, 1);
  hair.position.set(0, 0, 0);

  const bodySize = bodyBox.getSize(new Vector3());
  const bodyCenter = bodyBox.getCenter(new Vector3());

  const rawBox = new Box3().setFromObject(hair);
  const rawSize = rawBox.getSize(new Vector3());
  if (rawSize.y <= 0) return;

  const scale = (HAIR_HEIGHT_FRACTION * bodySize.y) / rawSize.y;
  hair.scale.setScalar(scale);

  const scaledBox = new Box3().setFromObject(hair);
  const scaledCenter = scaledBox.getCenter(new Vector3());

  hair.position.x += bodyCenter.x - scaledCenter.x;
  hair.position.y += bodyBox.max.y - scaledBox.max.y;
  hair.position.z += bodyCenter.z - scaledCenter.z;
}

/**
 * Overwrites every mesh's UV attribute in `root` with a cylindrical
 * projection computed from its own vertex positions (azimuth around the
 * vertical axis -> U, height -> V), replacing whatever UV layout the GLB
 * shipped with.
 *
 * Why: a downloaded garment like `TSHIRT_ASSET` has a real multi-panel UV
 * unwrap (front/back/sleeves/collar packed into separate regions of 0-1
 * UV space — verified by inspecting tshirt_model.glb directly). Mapping a
 * single uploaded fabric photo through `applyGarmentTexture` onto that
 * layout scatters unrelated crops of the photo across each panel — a
 * correctly-framed print on the front torso, but a warped, unrelated-looking
 * slice of the same image on the sleeves/back, which reads as a "ghost"
 * duplicate. A computed cylindrical UV instead varies smoothly with actual
 * 3D position, so neighboring geometry (torso into sleeve) samples
 * neighboring parts of the source image instead of jumping to a disjoint UV
 * island. Only appropriate for a standalone, unrigged garment fitted via
 * `fitGarmentToBody` — never for the server-built garments in WardrobeScreen,
 * which already ship a real cylindrical UV matching their flat-color paint.
 */
function applyCylindricalUV(root: Object3D) {
  root.updateMatrixWorld(true);
  root.traverse((obj) => {
    if (!(obj instanceof Mesh)) return;
    const position = obj.geometry.attributes.position;
    if (!position) return;

    const world = new Vector3();
    const points: Vector3[] = new Array(position.count);
    let minY = Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < position.count; i++) {
      world.fromBufferAttribute(position, i).applyMatrix4(obj.matrixWorld);
      points[i] = world.clone();
      minY = Math.min(minY, world.y);
      maxY = Math.max(maxY, world.y);
    }
    const heightRange = maxY - minY || 1;

    const uv = new Float32Array(position.count * 2);
    for (let i = 0; i < position.count; i++) {
      const p = points[i];
      uv[i * 2] = Math.atan2(p.z, p.x) / (Math.PI * 2) + 0.5;
      uv[i * 2 + 1] = (p.y - minY) / heightRange;
    }
    obj.geometry.setAttribute('uv', new BufferAttribute(uv, 2));
  });
}

// Name tag on the child Mesh `applyLowerBodyFabric` adds to the body, so a
// later call (coverage slider moving, a new fabric photo) can find and
// remove the previous one before rebuilding.
const LOWER_GARMENT_NAME = 'lowerGarmentFabric';

// Name tag for `applyUpperBodyFabric`'s child mesh (see LOWER_GARMENT_NAME).
const UPPER_GARMENT_NAME = 'upperGarmentFabric';

// Fractions of standing height (body's own rest frame) bounding the
// "torso + short sleeve" triangle mask below. Calibrated empirically against
// assets/avatars/rp/female_base_mesh.glb by binning its vertices into height
// slices and measuring width per slice: the torso column is a roughly
// constant ~0.27-0.40 wide from the hip up to ~75% height, then the T-posed
// arms spike to ~1.75-1.8 wide between ~77-83% height (the shoulder/sleeve
// band) before narrowing into the neck/head above ~87%. TORSO_HALF_WIDTH and
// SLEEVE_REACH are fractions of height (not width) since a T-pose's overall
// arm-span varies enormously by asset/pose, while shoulder width scales much
// more consistently with height across humanoid figures.
const UPPER_FABRIC_HEM_FRACTION = 0.5;
const UPPER_FABRIC_COLLAR_FRACTION = 0.75;
const UPPER_FABRIC_SLEEVE_MIN_FRACTION = 0.74;
const UPPER_FABRIC_SLEEVE_MAX_FRACTION = 0.87;
const UPPER_FABRIC_TORSO_HALF_WIDTH_FRACTION = 0.105;
const UPPER_FABRIC_SLEEVE_REACH_FRACTION = 0.22;

// Fractions of standing height (in the body's own rest frame) bounding the
// "legs" triangle mask below — hip/crotch line down to the ankle. There's no
// per-limb vertex grouping on this mesh, so anything in this height range
// counts as "leg" (a hanging hand/forearm at the top of the range would too,
// but arms don't reach anywhere near ankle height, so most of the range is
// unambiguous).
const LOWER_GARMENT_HIP_FRACTION = 0.5;
const LOWER_GARMENT_ANKLE_FRACTION = 0.04;

/** Composes `node`'s matrix with each ancestor's up to (but not including)
 * `ancestor`, giving `node`'s transform relative to `ancestor` regardless of
 * `ancestor`'s own current position/scale — used so `applyLowerBodyFabric`
 * measures the body's rest-frame proportions independent of whatever height/
 * width scale is currently applied to the body's own top-level transform. */
function localMatrixRelativeToAncestor(node: Object3D, ancestor: Object3D): Matrix4 {
  const matrix = new Matrix4();
  let current: Object3D | null = node;
  while (current && current !== ancestor) {
    current.updateMatrix();
    matrix.premultiply(current.matrix);
    current = current.parent;
  }
  return matrix;
}

/**
 * Paints `textureUri` directly onto `body`'s own leg triangles — "shorts" vs
 * "trousers" have no dedicated garment glb the way the upper-body
 * TSHIRT_ASSET does, so instead of overlaying a separate mesh this extracts
 * the leg-height triangle subset of the body's own geometry into a new child
 * Mesh (added to `body`, so it automatically tracks the body's own height/
 * width scaling), gives it a computed cylindrical UV (see
 * `applyCylindricalUV`), and applies the fabric there — the rest of the
 * body's own material (flat skin tint) is untouched.
 *
 * `coverage` 0..1 selects how far down from the hip the fabric extends: 0 ~
 * short shorts (just below the hip), 1 ~ full-length trousers (down to the
 * ankle). Re-derives the split from `body`'s own rest-frame geometry every
 * call — removing any previously-added lower-garment mesh first — so this
 * can be re-run any number of times as `coverage` changes from a slider
 * without drifting.
 */
async function applyLowerBodyFabric(body: Object3D, textureUri: string, coverage: number): Promise<void> {
  for (const child of [...body.children]) {
    if (child.name === LOWER_GARMENT_NAME) body.remove(child);
  }

  const texture = await loadFabricTexture(textureUri);

  // Body-relative (not world-space) positions, so this doesn't shift if the
  // body is re-parented or its own scale changes between calls.
  const meshData: { mesh: Mesh; matrix: Matrix4 }[] = [];
  let minY = Infinity;
  let maxY = -Infinity;
  const probe = new Vector3();
  body.traverse((obj) => {
    if (!(obj instanceof Mesh)) return;
    const position = obj.geometry.attributes.position;
    if (!position) return;
    const matrix = localMatrixRelativeToAncestor(obj, body);
    meshData.push({ mesh: obj, matrix });
    for (let i = 0; i < position.count; i++) {
      probe.fromBufferAttribute(position, i).applyMatrix4(matrix);
      minY = Math.min(minY, probe.y);
      maxY = Math.max(maxY, probe.y);
    }
  });
  const bodyHeight = maxY - minY;
  if (!(bodyHeight > 0)) return;

  const hipY = minY + LOWER_GARMENT_HIP_FRACTION * bodyHeight;
  const ankleY = minY + LOWER_GARMENT_ANKLE_FRACTION * bodyHeight;
  const coverageBottomY = hipY - coverage * (hipY - ankleY);

  const legPositions: number[] = [];
  const legIndex: number[] = [];
  const remap = new Map<number, number>();
  let globalVertexOffset = 0;
  const v = new Vector3();

  for (const { mesh, matrix } of meshData) {
    const position = mesh.geometry.attributes.position;
    const sourceIndex = mesh.geometry.index;
    const triangleCount = sourceIndex ? sourceIndex.count / 3 : position.count / 3;
    remap.clear();

    const vertexIndex = (i: number): number => {
      let mapped = remap.get(i);
      if (mapped === undefined) {
        v.fromBufferAttribute(position, i).applyMatrix4(matrix);
        legPositions.push(v.x, v.y, v.z);
        mapped = globalVertexOffset++;
        remap.set(i, mapped);
      }
      return mapped;
    };

    for (let t = 0; t < triangleCount; t++) {
      const ia = sourceIndex ? sourceIndex.getX(t * 3) : t * 3;
      const ib = sourceIndex ? sourceIndex.getX(t * 3 + 1) : t * 3 + 1;
      const ic = sourceIndex ? sourceIndex.getX(t * 3 + 2) : t * 3 + 2;

      let inLeg = true;
      for (const idx of [ia, ib, ic]) {
        v.fromBufferAttribute(position, idx).applyMatrix4(matrix);
        if (v.y < coverageBottomY || v.y > hipY) {
          inLeg = false;
          break;
        }
      }
      if (!inLeg) continue;

      legIndex.push(vertexIndex(ia), vertexIndex(ib), vertexIndex(ic));
    }
  }
  if (legIndex.length === 0) return;

  const legGeometry = new BufferGeometry();
  legGeometry.setAttribute('position', new BufferAttribute(new Float32Array(legPositions), 3));
  legGeometry.setIndex(legIndex);
  legGeometry.computeVertexNormals();

  const legMesh = new Mesh(
    legGeometry,
    new MeshStandardMaterial({ map: texture, metalness: 0.0, roughness: 0.85 })
  );
  legMesh.name = LOWER_GARMENT_NAME;
  body.add(legMesh);
  applyCylindricalUV(legMesh);
}

/**
 * Paints `textureUri` directly onto `body`'s own torso + short-sleeve
 * triangles — the same technique as `applyLowerBodyFabric`, applied to the
 * upper body instead of the legs.
 *
 * Why this exists alongside `TSHIRT_ASSET`/`fitGarmentToBody`: that's a
 * separate, rigid (unskinned) garment scan being scaled/positioned onto a
 * body it was never modeled for. On a body whose T-pose spreads its arms
 * much wider relative to its height than the body that scan was eyeballed
 * against (confirmed: female_base_mesh.glb's arm-span/height ratio is ~71%
 * wider than male_base_mesh.glb's), no amount of scaling reconciles "sleeves
 * reach the arm" with "the armhole seals against the body" at once — the
 * mismatch always shows as either short sleeves or a gap at the side/back.
 * Painting directly onto the body's own geometry has no such mismatch: it's
 * physically the same surface being wrapped, so it can't gap or float
 * regardless of the body's proportions or pose.
 *
 * The mask (torso band + a separate, wider band at shoulder height for the
 * sleeve) was calibrated by inspecting female_base_mesh.glb's own per-height
 * vertex-width profile — see the UPPER_FABRIC_* constants' comment. It
 * assumes a T-pose (arms spread sideways, roughly constant height along
 * their length) to place the sleeve band; a body posed with arms hanging
 * down (e.g. male_base_mesh.glb) wouldn't have much arm geometry fall inside
 * a shoulder-height band, so this is intended for T-posed bodies specifically
 * rather than as a universal replacement for the garment-mesh approach.
 */
async function applyUpperBodyFabric(body: Object3D, textureUri: string): Promise<void> {
  for (const child of [...body.children]) {
    if (child.name === UPPER_GARMENT_NAME) body.remove(child);
  }

  const texture = await loadFabricTexture(textureUri);

  const meshData: { mesh: Mesh; matrix: Matrix4 }[] = [];
  let minY = Infinity;
  let maxY = -Infinity;
  const probe = new Vector3();
  body.traverse((obj) => {
    if (!(obj instanceof Mesh)) return;
    const position = obj.geometry.attributes.position;
    if (!position) return;
    const matrix = localMatrixRelativeToAncestor(obj, body);
    meshData.push({ mesh: obj, matrix });
    for (let i = 0; i < position.count; i++) {
      probe.fromBufferAttribute(position, i).applyMatrix4(matrix);
      minY = Math.min(minY, probe.y);
      maxY = Math.max(maxY, probe.y);
    }
  });
  const bodyHeight = maxY - minY;
  if (!(bodyHeight > 0)) return;

  const hemY = minY + UPPER_FABRIC_HEM_FRACTION * bodyHeight;
  const collarY = minY + UPPER_FABRIC_COLLAR_FRACTION * bodyHeight;
  const sleeveMinY = minY + UPPER_FABRIC_SLEEVE_MIN_FRACTION * bodyHeight;
  const sleeveMaxY = minY + UPPER_FABRIC_SLEEVE_MAX_FRACTION * bodyHeight;
  const torsoHalfWidth = UPPER_FABRIC_TORSO_HALF_WIDTH_FRACTION * bodyHeight;
  const sleeveReach = UPPER_FABRIC_SLEEVE_REACH_FRACTION * bodyHeight;

  const inMask = (x: number, y: number): boolean => {
    const inTorso = y >= hemY && y <= collarY && Math.abs(x) <= torsoHalfWidth;
    const inSleeve = y >= sleeveMinY && y <= sleeveMaxY && Math.abs(x) <= sleeveReach;
    return inTorso || inSleeve;
  };

  const fabricPositions: number[] = [];
  const fabricIndex: number[] = [];
  const remap = new Map<number, number>();
  let globalVertexOffset = 0;
  const v = new Vector3();

  for (const { mesh, matrix } of meshData) {
    const position = mesh.geometry.attributes.position;
    const sourceIndex = mesh.geometry.index;
    const triangleCount = sourceIndex ? sourceIndex.count / 3 : position.count / 3;
    remap.clear();

    const vertexIndex = (i: number): number => {
      let mapped = remap.get(i);
      if (mapped === undefined) {
        v.fromBufferAttribute(position, i).applyMatrix4(matrix);
        fabricPositions.push(v.x, v.y, v.z);
        mapped = globalVertexOffset++;
        remap.set(i, mapped);
      }
      return mapped;
    };

    for (let t = 0; t < triangleCount; t++) {
      const ia = sourceIndex ? sourceIndex.getX(t * 3) : t * 3;
      const ib = sourceIndex ? sourceIndex.getX(t * 3 + 1) : t * 3 + 1;
      const ic = sourceIndex ? sourceIndex.getX(t * 3 + 2) : t * 3 + 2;

      let included = true;
      for (const idx of [ia, ib, ic]) {
        v.fromBufferAttribute(position, idx).applyMatrix4(matrix);
        if (!inMask(v.x, v.y)) {
          included = false;
          break;
        }
      }
      if (!included) continue;

      fabricIndex.push(vertexIndex(ia), vertexIndex(ib), vertexIndex(ic));
    }
  }
  if (fabricIndex.length === 0) return;

  const fabricGeometry = new BufferGeometry();
  fabricGeometry.setAttribute('position', new BufferAttribute(new Float32Array(fabricPositions), 3));
  fabricGeometry.setIndex(fabricIndex);
  fabricGeometry.computeVertexNormals();

  const fabricMesh = new Mesh(
    fabricGeometry,
    new MeshStandardMaterial({ map: texture, metalness: 0.0, roughness: 0.85 })
  );
  fabricMesh.name = UPPER_GARMENT_NAME;
  body.add(fabricMesh);
  applyCylindricalUV(fabricMesh);
}

/** Sets morph-target influences on every mesh in `root` that has a populated
 * `morphTargetDictionary` (from gltf.meshes[0].extras.targetNames). No-op for
 * GLBs without morph targets. */
function applyBodyMorphs(root: Object3D, morphWeights: Record<MorphTargetName, number>) {
  root.traverse((obj) => {
    if (!(obj instanceof Mesh)) return;
    if (!obj.morphTargetDictionary || !obj.morphTargetInfluences) return;
    for (const [name, weight] of Object.entries(morphWeights) as [MorphTargetName, number][]) {
      const index = obj.morphTargetDictionary[name];
      if (index !== undefined) obj.morphTargetInfluences[index] = weight;
    }
  });
}

/** Centers `model` on the origin and wraps it in a `Group` ("pivot") scaled so
 * its largest dimension maps to 1.6 world units — shared by the body model
 * and a single remote Avaturn GLB. */
function centerAndPivot(model: Object3D): Group {
  const box = new Box3().setFromObject(model);
  const size = box.getSize(new Vector3());
  const center = box.getCenter(new Vector3());
  model.position.sub(center);

  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const pivot = new Group();
  pivot.add(model);
  pivot.scale.setScalar(1.6 / maxDim);
  return pivot;
}

/** Stable identity for a `garmentMeshUrls`/`garmentTextureUrls` pair — used
 * to tell whether the live-update effect actually needs to reload garments,
 * since the arrays themselves are new references every render. */
function garmentKeyFor(
  garmentMeshUrls: string[] | undefined,
  garmentTextureUrls: (string | null | undefined)[] | undefined
): string {
  return `${garmentMeshUrls?.join('|') ?? ''}::${garmentTextureUrls?.join('|') ?? ''}`;
}

/** Loads each `garmentMeshUrls` entry, fits/textures it, and adds it to
 * `model` — shared by `onContextCreate`'s initial load and the live-update
 * effect that re-runs this when `garmentMeshUrls`/`garmentTextureUrls`
 * change, so toggling a garment doesn't need a full component remount (see
 * that effect's comment for why remounting specifically is worth avoiding
 * here). Returns the loaded garment Object3Ds so the caller can track them
 * for later removal/re-fitting, and an error message if any garment or its
 * texture failed. */
async function loadAndFitGarments(
  model: Group,
  bodyBox: Box3,
  garmentMeshUrls: string[],
  garmentTextureUrls: (string | null | undefined)[] | undefined,
  autoFitGarments: boolean | undefined,
  garmentFit: GarmentFitOptions | undefined
): Promise<{ garments: Object3D[]; error?: string }> {
  const loadedGarments: Object3D[] = [];
  let garmentError: string | undefined;
  for (const [index, url] of garmentMeshUrls.entries()) {
    try {
      const garmentModel = await loadGlbScene(url);
      if (autoFitGarments) {
        // A standalone catalog garment (e.g. a t-shirt GLB) isn't
        // pre-aligned to any body, and its own UV layout doesn't match
        // the simple cylindrical scheme `applyGarmentTexture` assumes
        // — recompute one before fitting it onto this body.
        applyCylindricalUV(garmentModel);
        fitGarmentToBody(garmentModel, bodyBox, garmentFit);
      }
      // Otherwise, the garment glb was built server-side from this
      // same avatar's body3d_params + height (see
      // garment_mesh.build_garment_glb), so it's already aligned with
      // the body in the same local space — added as-is.
      model.add(garmentModel);
      loadedGarments.push(garmentModel);

      const textureUrl = garmentTextureUrls?.[index];
      if (textureUrl) {
        try {
          await applyGarmentTexture(garmentModel, textureUrl);
        } catch (texErr) {
          console.error('[AvatarViewer3D] garment texture load failed', textureUrl, texErr);
          garmentError = `Garment texture failed to load: ${
            texErr instanceof Error ? texErr.message : String(texErr)
          }`;
        }
      }
    } catch (err) {
      console.error('[AvatarViewer3D] garment load failed', url, err);
      garmentError = `Garment model failed to load: ${err instanceof Error ? err.message : String(err)}`;
    }
  }
  return { garments: loadedGarments, error: garmentError };
}

/**
 * Renders a realistic GLB avatar built by `buildAvatar()`, tinted with the
 * wearer's detected skin color. Drag horizontally to spin the model around
 * its vertical axis. Face preview images are shown by the avatar screens in a
 * separate card, never as the 3D viewer background.
 */
export function AvatarViewer3D({
  config,
  remoteAvatarUrl,
  remoteTextureUrl,
  garmentMeshUrls,
  garmentTextureUrls,
  autoFitGarments,
  garmentFit,
  onGarmentStatus,
  bottomTextureUri,
  bottomCoverage,
  onBottomFabricStatus,
  topFabricTextureUri,
  onTopFabricStatus,
}: Props) {
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [texStatus, setTexStatus] = useState<string>('');
  const pivotRef = useRef<Group | null>(null);
  const committedRotationRef = useRef(0);
  const bodyRef = useRef<Object3D | null>(null);
  const modelRef = useRef<Group | null>(null);
  const garmentsRef = useRef<Object3D[]>([]);
  // Identifies which garmentMeshUrls/garmentTextureUrls set is currently
  // loaded into garmentsRef, so the live-update effect below (which shares
  // this identity check) doesn't redundantly reload the same garments right
  // after onContextCreate's own initial load finishes.
  const appliedGarmentKeyRef = useRef<string | null>(null);
  const bodyBoxRef = useRef<Box3 | null>(null);
  const rendererRef = useRef<InstanceType<typeof Renderer> | null>(null);

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: (_event, gesture) => Math.abs(gesture.dx) > 2,
      onPanResponderMove: (_event, gesture) => {
        const pivot = pivotRef.current;
        if (pivot) {
          pivot.rotation.y = committedRotationRef.current + gesture.dx * ROTATE_SPEED;
        }
      },
      onPanResponderRelease: (_event, gesture) => {
        committedRotationRef.current += gesture.dx * ROTATE_SPEED;
      },
    })
  ).current;

  const onContextCreate = async (gl: ExpoWebGLRenderingContext) => {
    const renderer = new Renderer({ gl });
    renderer.setSize(gl.drawingBufferWidth, gl.drawingBufferHeight);
    renderer.setClearColor(0x000000, 0);
    rendererRef.current = renderer;

    const scene = new Scene();
    const camera = new PerspectiveCamera(35, gl.drawingBufferWidth / gl.drawingBufferHeight, 0.1, 100);
    camera.position.set(0, 0, 3.2);

    scene.add(new AmbientLight(0xffffff, 0.7));
    const keyLight = new DirectionalLight(0xffffff, 0.9);
    keyLight.position.set(1.5, 3, 2);
    scene.add(keyLight);

    try {
      let body: Object3D;
      const extras: Object3D[] = []; // hair — only populated on the local (non-remote) path

      if (remoteAvatarUrl) {
        body = await loadGlbScene(remoteAvatarUrl);
        if (remoteTextureUrl) {
          try {
            setTexStatus('Loading texture…');
            let texUri = remoteTextureUrl;
            if (Platform.OS !== 'web') {
              // On native, download to a local file:// URI first — expo-three's
              // TextureLoader can't reliably fetch http:// on device.
              const localPath = (FileSystem.cacheDirectory ?? '') + 'face_tex.png';
              const dl = await FileSystem.downloadAsync(remoteTextureUrl, localPath);
              if (dl.status !== 200) throw new Error(`HTTP ${dl.status}`);
              texUri = dl.uri;
            }
            const tex = await new TextureLoader().loadAsync(texUri);
            tex.colorSpace = SRGBColorSpace;
            tex.needsUpdate = true;
            console.log('[AvatarViewer3D] face texture loaded, applying...');
            const n = applyFaceTexture(body, tex);
            setTexStatus(n > 0 ? `Face texture applied (${n} mesh)` : 'Texture loaded but 0 meshes found');
          } catch (err) {
            console.error('[AvatarViewer3D] remote face texture load failed', err);
            setTexStatus(`Texture error: ${err instanceof Error ? err.message : String(err)}`);
          }
        }
        // Apply height and morph-target body scaling to the remote mesh too.
        // applyBodyMorphs is a no-op when the server GLB has no morph targets,
        // but height scale and body-type width scale always take effect.
        applyBodyMorphs(body, config.bodyScale.morphWeights);
        // Approximate shoulder/hip width from the bodyType weight (–1 slim → 1 plus).
        const widthScale = 1 + config.bodyScale.morphWeights.bodyType * 0.08;

        body.scale.y *= config.bodyScale.heightScale;
        body.scale.x *= widthScale;
        body.scale.z *= widthScale;
      } else {
        // assets/hair/long.glb (the female hair asset) is a wide/round
        // shape rather than the narrow "hangs down the back" silhouette its
        // name implies — scaled to fit the head via fitHairToBody, it balls
        // up into a dome that swallows the whole face. Skip hair entirely
        // for the female avatar until a real female hairstyle asset exists;
        // male is unaffected.
        const includeHair = config.gender !== 'female';

        const [loadedBody, hair] = await Promise.all([
          loadGlbScene(config.bodyAsset),
          includeHair ? loadGlbScene(config.hairAsset) : Promise.resolve(null),
        ]);

        // Face-photo compositing (buildHeadTexture + applyFaceTexture) is
        // disabled for the local body mesh: it relies on the old body's
        // vertices all being pinned to a single UV corner so the face photo
        // only shows up on head vertices. assets/avatars/rp/*.glb has real
        // per-part UVs, so the same texture bleeds onto hands/arms/torso
        // wherever their UVs happen to land inside the face-photo region.
        // Flat tint only, for now.
        applyTint(loadedBody, config.skinColor);
        applyBodyMorphs(loadedBody, config.bodyScale.morphWeights);
        loadedBody.scale.y *= config.bodyScale.heightScale;

        if (hair) {
          const [hr, hg, hb] = config.features.hairRgb;
          applyTint(hair, [hr / 255, hg / 255, hb / 255]);

          // assets/hair/*.glb ships in a completely different unit scale
          // from assets/avatars/rp/*.glb (see fitHairToBody's doc comment) —
          // rescale and reposition it onto the body's own bounding box
          // instead of assuming its raw coordinates are usable as-is.
          const bodyBoxForHair = new Box3().setFromObject(loadedBody);
          fitHairToBody(hair, bodyBoxForHair);
          extras.push(hair);
        }

        body = loadedBody;
      }
      bodyRef.current = body;

      const model = new Group();
      model.add(body);
      for (const extra of extras) model.add(extra);
      modelRef.current = model;

      // bodyBoxRef is needed by the live-update effects (garmentFit and the
      // garment add/remove effect below) regardless of whether a garment is
      // worn from the start — a garment toggled on later still needs it.
      const bodyBox = new Box3().setFromObject(body);
      bodyBoxRef.current = bodyBox;

      // Garments (e.g. TSHIRT_ASSET) are kept in a separate `garmentsRef`
      // (not folded into `bodyRef`) so `applyFaceTexture`/`retintBody` below
      // — which traverse `bodyRef.current` — never touch their material.
      if (garmentMeshUrls && garmentMeshUrls.length > 0) {
        const { garments, error } = await loadAndFitGarments(
          model,
          bodyBox,
          garmentMeshUrls,
          garmentTextureUrls,
          autoFitGarments,
          garmentFit
        );
        garmentsRef.current = garments;
        onGarmentStatus?.(error ? 'error' : 'ready', error);
      } else {
        garmentsRef.current = [];
      }
      // Marks this garment set as already applied, so the live-update effect
      // (which shares this same key) doesn't immediately redo this same work
      // the moment `status` flips to 'ready' below.
      appliedGarmentKeyRef.current = garmentKeyFor(garmentMeshUrls, garmentTextureUrls);

      // Pivot scale (1.6 / bounding-box max dimension) is computed once here
      // from the model's INITIAL geometry. Live morph/height updates from
      // body-customization sliders (see the effect below) change the mesh's
      // actual bounding box but don't recompute this scale, so the avatar may
      // appear marginally larger/smaller at slider extremes. Acceptable for a
      // live-preview control; recomputing would mean re-parenting `model`
      // into a new pivot, which would also reset in-progress drag rotation.
      const pivot = centerAndPivot(model);
      scene.add(pivot);

      pivotRef.current = pivot;
      setStatus('ready');
    } catch (err) {
      console.error('[AvatarViewer3D] onContextCreate failed', err);
      setStatus('error');
    }

    const render = () => {
      requestAnimationFrame(render);
      renderer.render(scene, camera);
      gl.endFrameEXP();
    };
    render();
  };

  // Live-update the already-mounted body when `config` changes (e.g. from
  // body-customization sliders), without reloading GLBs. `status` is in the
  // deps so a config change that arrives before the initial load finishes
  // (bodyRef still null) is picked up once loading completes.
  useEffect(() => {
    const body = bodyRef.current;
    if (!body) return;

    applyBodyMorphs(body, config.bodyScale.morphWeights);
    body.scale.y = config.bodyScale.heightScale;

    if (remoteAvatarUrl) {
      const widthScale = 1 + config.bodyScale.morphWeights.bodyType * 0.08;
      body.scale.x = widthScale;
      body.scale.z = widthScale;
      // Auto-fit garments (see `fitGarmentToBody`) were deliberately scaled
      // to a fraction of the body's size, not a 1:1 copy of it — only the
      // legacy pre-aligned server garments (WardrobeScreen) track the body's
      // scale directly.
      if (!autoFitGarments) {
        for (const garment of garmentsRef.current) {
          garment.scale.copy(body.scale);
        }
      }
      return;
    }

    // Face-photo compositing is disabled for the local body mesh (see the
    // matching note in onContextCreate) — always a flat re-tint.
    retintBody(body, config.skinColor);
  }, [remoteAvatarUrl, config.bodyScale, config.skinColor, config.faceTextureUri, status]);

  // Live-update autoFit garments (sleeve-length / top-length sliders) without
  // reloading their GLB or touching the already-applied texture — re-derived
  // from scratch each time by `fitGarmentToBody`, so no drift across edits.
  useEffect(() => {
    if (!autoFitGarments) return;
    const bodyBox = bodyBoxRef.current;
    if (!bodyBox) return;
    for (const garment of garmentsRef.current) {
      fitGarmentToBody(garment, bodyBox, garmentFit);
    }
  }, [autoFitGarments, garmentFit?.heightFraction, garmentFit?.sleeveScale, status]);

  // Live-update which garments are worn (e.g. MaleAvatarScreen/
  // FemaleAvatarScreen's "Top" switch, or a newly-picked fabric photo)
  // without remounting this component. This deliberately does NOT reload
  // the whole avatar via a `key` change the way it used to: remounting reran
  // `onContextCreate` from scratch, including re-resolving `config.bodyAsset`
  // via `Asset.fromModule` — for a recently-added/renamed local asset (see
  // avatarBuilder.ts's `female_base_mesh_v2.glb` note) that repeated
  // re-resolution hit a stale Metro/browser asset-registry entry often
  // enough to occasionally load the WRONG body file on remount. Adding/
  // removing garments in place on the already-loaded body sidesteps that
  // entirely, and is also strictly better UX (no reload flicker).
  useEffect(() => {
    if (status !== 'ready') return;
    const model = modelRef.current;
    const bodyBox = bodyBoxRef.current;
    if (!model || !bodyBox) return;

    const key = garmentKeyFor(garmentMeshUrls, garmentTextureUrls);
    if (key === appliedGarmentKeyRef.current) return; // already applied (e.g. onContextCreate's own initial load)
    appliedGarmentKeyRef.current = key;

    for (const garment of garmentsRef.current) {
      model.remove(garment);
    }
    garmentsRef.current = [];

    if (!garmentMeshUrls || garmentMeshUrls.length === 0) {
      onGarmentStatus?.('ready');
      return;
    }

    let cancelled = false;
    loadAndFitGarments(model, bodyBox, garmentMeshUrls, garmentTextureUrls, autoFitGarments, garmentFit).then(
      ({ garments, error }) => {
        if (cancelled) {
          // This garment set was superseded by another change before it
          // finished loading — drop it instead of leaving it in the scene.
          for (const garment of garments) model.remove(garment);
          return;
        }
        garmentsRef.current = garments;
        onGarmentStatus?.(error ? 'error' : 'ready', error);
      }
    );
    return () => {
      cancelled = true;
    };
  }, [garmentKeyFor(garmentMeshUrls, garmentTextureUrls), status]);

  // Live-update the "bottoms" fabric paint (see `applyLowerBodyFabric`) — no
  // GLB to reload here, so this alone (no `key` remount) handles both first
  // application and later coverage/photo changes.
  useEffect(() => {
    const body = bodyRef.current;
    if (!body) return;

    if (!bottomTextureUri) {
      for (const child of [...body.children]) {
        if (child.name === LOWER_GARMENT_NAME) body.remove(child);
      }
      return;
    }

    applyLowerBodyFabric(body, bottomTextureUri, bottomCoverage ?? 0.5)
      .then(() => onBottomFabricStatus?.('ready'))
      .catch((err) => {
        console.error('[AvatarViewer3D] lower-body fabric failed', err);
        onBottomFabricStatus?.('error', err instanceof Error ? err.message : String(err));
      });
  }, [bottomTextureUri, bottomCoverage, status]);

  // Live-update the "top" fabric paint (see `applyUpperBodyFabric`) — same
  // pattern as the bottoms effect above, no GLB/remount needed.
  useEffect(() => {
    const body = bodyRef.current;
    if (!body) return;

    if (!topFabricTextureUri) {
      for (const child of [...body.children]) {
        if (child.name === UPPER_GARMENT_NAME) body.remove(child);
      }
      return;
    }

    applyUpperBodyFabric(body, topFabricTextureUri)
      .then(() => onTopFabricStatus?.('ready'))
      .catch((err) => {
        console.error('[AvatarViewer3D] upper-body fabric failed', err);
        onTopFabricStatus?.('error', err instanceof Error ? err.message : String(err));
      });
  }, [topFabricTextureUri, status]);

  const glView = <GLView style={styles.glView} onContextCreate={onContextCreate} />;

  return (
    <View style={styles.container} {...panResponder.panHandlers}>
      {glView}
      {status !== 'ready' && (
        <View style={styles.overlay} pointerEvents="none">
          {status === 'loading' ? (
            <ActivityIndicator color={colors.primary} />
          ) : (
            <Text style={typography.body}>Couldn't load the 3D model.</Text>
          )}
        </View>
      )}
      {texStatus ? (
        <View style={styles.texOverlay} pointerEvents="none">
          <Text style={styles.texStatus}>{texStatus}</Text>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: radii.md,
    overflow: 'hidden',
    backgroundColor: colors.background,
  },
  glView: {
    flex: 1,
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.md,
  },
  texOverlay: {
    position: 'absolute',
    bottom: 4,
    left: 4,
    right: 4,
  },
  texStatus: {
    fontSize: 10,
    color: '#fff',
    backgroundColor: 'rgba(0,0,0,0.55)',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    overflow: 'hidden',
  },
});
