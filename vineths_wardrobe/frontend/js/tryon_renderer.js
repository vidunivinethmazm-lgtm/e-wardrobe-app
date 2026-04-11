/**
 * eWardrobeAI — Three.js / WebGL Avatar Renderer
 * Stage 4: Animation & Real-Time Rendering
 *
 * Architecture
 * ────────────
 * Scene graph:
 *   Scene
 *   ├── AmbientLight
 *   ├── DirectionalLight  (key)
 *   ├── DirectionalLight  (fill)
 *   ├── HemisphereLight
 *   └── AvatarGroup
 *       ├── BaseMesh      (Blender .glb — rigged with Mixamo skeleton)
 *       └── ClothingMesh* (one per garment .glb, attached to skeleton)
 *
 * Face Texture Pipeline
 * ─────────────────────
 * Backend sends face_texture_b64 (512×512 PNG, base64-encoded).
 * We decode it as a THREE.Texture and assign to the head mesh material.
 * UV mapping is pre-baked into the Blender avatar .glb.
 *
 * Avatar Scaling
 * ──────────────
 * scaleParams from BodyCalibrator are applied per bone group:
 *   Hips → global Y scale (height)
 *   Spine → chest / waist X scale
 *   Shoulders → shoulder X scale
 *   Thighs  → leg Y scale
 *   Head    → head uniform scale
 *
 * Mixamo Animations
 * ─────────────────
 * Animation clips are baked into the base avatar .glb and accessed via
 * AnimationMixer. Cross-fade transitions use mixer.clipAction().crossFadeTo().
 */

(function () {
  'use strict';

  // ── Constants ──────────────────────────────────────────────────────────────
  const ANIMATION_FADE_DURATION = 0.5;  // seconds
  const DEMO_AVATAR_HEIGHT      = 1.75; // metres (base avatar neutral height)
  const CAMERA_NEAR             = 0.1;
  const CAMERA_FAR              = 100;

  // Bone name mappings for Mixamo rig
  const BONE_MAP = {
    globalY:   ['mixamorigHips'],
    shoulderX: ['mixamorigLeftShoulder', 'mixamorigRightShoulder'],
    chestX:    ['mixamorigSpine1', 'mixamorigSpine2'],
    waistX:    ['mixamorigSpine'],
    hipX:      ['mixamorigHips'],
    legY:      ['mixamorigLeftUpLeg', 'mixamorigRightUpLeg',
                 'mixamorigLeftLeg',  'mixamorigRightLeg'],
    headScale: ['mixamorigHead', 'mixamorigNeck'],
  };

  // ── Renderer State ─────────────────────────────────────────────────────────
  let renderer, scene, camera, controls;
  let avatarGroup    = null;
  let avatarMixer    = null;
  let currentAction  = null;
  let animFrameId    = null;
  let clock;
  let payloadCache   = null;
  let _onLoadComplete = null;

  // ── Initialise Three.js Scene ──────────────────────────────────────────────
  function init() {
    const canvas = document.getElementById('renderer-canvas');
    if (!canvas) return;

    clock = new THREE.Clock();

    // Renderer
    renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha:     false,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type    = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace   = THREE.SRGBColorSpace;
    renderer.toneMapping        = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0F0F13);
    scene.fog = new THREE.Fog(0x0F0F13, 8, 20);

    // Camera — must be created before resizeRenderer()
    camera = new THREE.PerspectiveCamera(45, getAspect(), CAMERA_NEAR, CAMERA_FAR);
    camera.position.set(0, 1.6, 3.5);
    resizeRenderer();

    // Orbit Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping   = true;
    controls.dampingFactor   = 0.08;
    controls.target.set(0, 1.0, 0);
    controls.minDistance     = 1.0;
    controls.maxDistance     = 8.0;
    controls.maxPolarAngle   = Math.PI * 0.85;

    // Lighting
    setupLighting();

    // Ground plane
    setupGround();

    // Start render loop
    animate();

    // Responsive resize
    window.addEventListener('resize', resizeRenderer);

    // Hide loading overlay
    hideLoadingOverlay();

    console.log('[eWardrobeRenderer] Three.js scene initialised.');
  }

  function setupLighting() {
    // Ambient
    const ambient = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambient);

    // Key light (front-right)
    const keyLight = new THREE.DirectionalLight(0xfff5e0, 1.8);
    keyLight.position.set(2, 4, 3);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(2048, 2048);
    keyLight.shadow.camera.near = 0.5;
    keyLight.shadow.camera.far  = 20;
    keyLight.shadow.bias = -0.001;
    scene.add(keyLight);

    // Fill light (left)
    const fillLight = new THREE.DirectionalLight(0xc5d8ff, 0.8);
    fillLight.position.set(-3, 2, 1);
    scene.add(fillLight);

    // Rim light (back)
    const rimLight = new THREE.DirectionalLight(0x9988ff, 0.5);
    rimLight.position.set(0, 3, -4);
    scene.add(rimLight);

    // Hemisphere sky/ground
    const hemi = new THREE.HemisphereLight(0x6060ff, 0x443322, 0.3);
    scene.add(hemi);
  }

  function setupGround() {
    const geo = new THREE.CircleGeometry(3, 64);
    const mat = new THREE.MeshStandardMaterial({
      color:     0x1A1A24,
      roughness: 0.9,
      metalness: 0.1,
    });
    const ground = new THREE.Mesh(geo, mat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    // Subtle grid
    const grid = new THREE.GridHelper(6, 24, 0x2E2E3E, 0x2E2E3E);
    grid.position.y = 0.001;
    scene.add(grid);
  }

  // ── Animation Loop ─────────────────────────────────────────────────────────
  function animate() {
    animFrameId = requestAnimationFrame(animate);
    const delta = clock.getDelta();
    if (avatarMixer) avatarMixer.update(delta);
    controls.update();
    renderer.render(scene, camera);
  }

  // ── Load Render Payload ────────────────────────────────────────────────────
  function loadPayload(payload, onComplete) {
    if (!payload) return;
    payloadCache = payload;

    showLoadingOverlay('Loading avatar…');
    clearAvatarGroup();
    _onLoadComplete = onComplete || null;

    avatarGroup = new THREE.Group();
    scene.add(avatarGroup);

    const loader = new THREE.GLTFLoader();

    // Check if avatar GLB path is valid (demo fallback: use placeholder geometry)
    const avatarPath = payload.avatarGlbPath;
    if (avatarPath && !avatarPath.includes('undefined')) {
      loader.load(
        avatarPath,
        gltf => onAvatarLoaded(gltf, payload),
        xhr => {
          const pct = (xhr.loaded / (xhr.total || 1) * 100).toFixed(0);
          updateLoadingText(`Loading avatar… ${pct}%`);
        },
        err => {
          console.warn('[Renderer] Avatar GLB not found, using demo mesh:', err.message);
          loadDemoAvatar(payload);
        }
      );
    } else {
      loadDemoAvatar(payload);
    }
  }

  function onAvatarLoaded(gltf, payload) {
    const avatarScene = gltf.scene;
    avatarScene.traverse(node => {
      if (node.isMesh) {
        node.castShadow    = true;
        node.receiveShadow = true;
      }
    });

    avatarGroup.add(avatarScene);

    // Apply bone scale transforms
    applyBoneScales(avatarScene, payload.scaleParams);

    // Apply face texture to head mesh
    if (payload.faceTextureB64) {
      applyFaceTexture(avatarScene, payload.faceTextureB64);
    }

    // Setup animation mixer
    if (gltf.animations && gltf.animations.length) {
      avatarMixer = new THREE.AnimationMixer(avatarScene);
      playAnimation(payload.animation);
    }

    // Load clothing assets
    loadClothingAssets(payload.clothingAssets, avatarScene, gltf.animations);

    // Centre avatar
    centreAvatar(avatarScene);
    hideLoadingOverlay();
    if (_onLoadComplete) { _onLoadComplete(); _onLoadComplete = null; }

    console.log(
      `[Renderer] Avatar loaded. Outfit: "${payload.outfitName}"  ` +
      `Clothing meshes: ${payload.clothingAssets.length}`
    );
  }

  // ── Demo Avatar (placeholder when .glb not found) ─────────────────────────
  function loadDemoAvatar(payload) {
    console.log('[Renderer] Using demo placeholder avatar.');

    // Simple capsule body
    const bodyGeo = new THREE.CapsuleGeometry(0.22, 1.1, 8, 16);
    const bodyMat = new THREE.MeshStandardMaterial({
      color:     0x888899,
      roughness: 0.8,
    });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = 0.85;
    body.castShadow = true;
    avatarGroup.add(body);

    // Head sphere
    const headGeo = new THREE.SphereGeometry(0.18, 32, 32);
    const headMat = new THREE.MeshStandardMaterial({ roughness: 0.7 });

    // Apply face texture to head if available
    if (payload.faceTextureB64) {
      const img     = new Image();
      img.src       = `data:image/png;base64,${payload.faceTextureB64}`;
      img.onload    = () => {
        const tex  = new THREE.Texture(img);
        tex.needsUpdate   = true;
        tex.colorSpace    = THREE.SRGBColorSpace;
        headMat.map       = tex;
        headMat.needsUpdate = true;
      };
    } else {
      headMat.color.set(0xFFCBA4);
    }

    const head = new THREE.Mesh(headGeo, headMat);
    head.position.y = 1.72;
    head.castShadow = true;
    avatarGroup.add(head);

    // Apply scale
    if (payload.scaleParams) {
      const s = payload.scaleParams;
      avatarGroup.scale.set(
        s.shoulderX || 1,
        s.globalY   || 1,
        s.chestX    || 1,
      );
    }

    // Demo rotation animation (substitute for Mixamo)
    const animKey = payload.animation?.clipName || '';
    if (animKey.includes('Walk') || animKey.includes('Catwalk')) {
      startDemoWalkAnimation();
    } else if (animKey.includes('Turn')) {
      startDemoRotateAnimation();
    }

    // Load clothing colour quads as placeholders
    loadDemoClothing(payload.clothingAssets);
    hideLoadingOverlay();
    if (_onLoadComplete) { _onLoadComplete(); _onLoadComplete = null; }
  }

  function loadDemoClothing(clothingAssets) {
    if (!clothingAssets || !avatarGroup) return;
    clothingAssets.forEach((asset, i) => {
      const colour = asset.colourHex || '#6C63FF';
      const geo    = new THREE.BoxGeometry(0.48, 0.55, 0.25);
      const mat    = new THREE.MeshStandardMaterial({
        color:     colour,
        roughness: 0.6,
        metalness: 0.05,
        transparent: true,
        opacity:   0.88,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.y = 0.85 + (i * 0.02);  // slight offset per layer
      mesh.castShadow  = true;
      avatarGroup.add(mesh);
    });
  }

  let _demoAnimReq = null;
  function startDemoWalkAnimation() {
    let t = 0;
    function step() {
      t += 0.025;
      if (avatarGroup) {
        avatarGroup.position.y = Math.sin(t * 2) * 0.04;
        avatarGroup.rotation.z = Math.sin(t) * 0.03;
      }
      _demoAnimReq = requestAnimationFrame(step);
    }
    if (_demoAnimReq) cancelAnimationFrame(_demoAnimReq);
    step();
  }

  function startDemoRotateAnimation() {
    let t = 0;
    function step() {
      t += 0.015;
      if (avatarGroup) {
        avatarGroup.rotation.y = t;
      }
      _demoAnimReq = requestAnimationFrame(step);
    }
    if (_demoAnimReq) cancelAnimationFrame(_demoAnimReq);
    step();
  }

  // ── Bone-Level Scale Application ──────────────────────────────────────────
  function applyBoneScales(avatarScene, scaleParams) {
    if (!scaleParams) return;

    avatarScene.traverse(node => {
      if (!node.isBone) return;
      const name = node.name;

      Object.entries(BONE_MAP).forEach(([paramKey, boneNames]) => {
        if (boneNames.some(bn => name.toLowerCase().includes(bn.toLowerCase()))) {
          const scale = scaleParams[paramKey] || 1.0;
          if (paramKey === 'globalY' || paramKey === 'legY') {
            node.scale.y = scale;
          } else if (paramKey === 'headScale') {
            node.scale.setScalar(scale);
          } else {
            node.scale.x = scale;
          }
        }
      });
    });
  }

  // ── Face Texture Application ───────────────────────────────────────────────
  function applyFaceTexture(avatarScene, textureB64) {
    const img    = new Image();
    img.src      = `data:image/png;base64,${textureB64}`;
    img.onload   = () => {
      const texture          = new THREE.Texture(img);
      texture.needsUpdate    = true;
      texture.colorSpace     = THREE.SRGBColorSpace;
      texture.flipY          = false;  // Blender UV convention

      // Find head mesh (named 'Head' or 'head' in the Blender export)
      avatarScene.traverse(node => {
        if (node.isMesh) {
          const n = node.name.toLowerCase();
          if (n.includes('head') || n.includes('face')) {
            if (Array.isArray(node.material)) {
              node.material[0].map        = texture;
              node.material[0].needsUpdate = true;
            } else {
              node.material.map        = texture;
              node.material.needsUpdate = true;
            }
            console.log(`[Renderer] Face texture applied to mesh: ${node.name}`);
          }
        }
      });
    };
  }

  // ── Clothing Asset Loader ──────────────────────────────────────────────────
  function loadClothingAssets(assets, avatarScene, avatarAnimations) {
    if (!assets || !assets.length) return;

    const loader = new THREE.GLTFLoader();
    assets
      .sort((a, b) => a.layerOrder - b.layerOrder)
      .forEach(asset => {
        if (!asset.assetExists) {
          console.log(`[Renderer] Clothing asset not found, skipping: ${asset.assetPath}`);
          return;
        }
        loader.load(
          asset.assetPath,
          gltf => {
            const clothMesh = gltf.scene;
            clothMesh.traverse(node => {
              if (node.isMesh) {
                node.castShadow    = true;
                node.receiveShadow = true;
                // Tint with garment colour
                if (node.material) {
                  node.material = node.material.clone();
                  node.material.color = new THREE.Color(asset.colourHex || '#FFFFFF');
                }
              }
            });

            avatarGroup.add(clothMesh);

            // Retarget clothing animations to avatar skeleton
            if (gltf.animations && gltf.animations.length && avatarMixer) {
              gltf.animations.forEach(clip => {
                THREE.AnimationUtils.makeClipAdditive(clip);
                avatarMixer.clipAction(clip).play();
              });
            }

            console.log(`[Renderer] Clothing loaded: ${asset.garmentId}`);
          },
          undefined,
          err => console.warn(`[Renderer] Clothing load error (${asset.assetPath}):`, err)
        );
      });
  }

  // ── Animation Control ──────────────────────────────────────────────────────
  function playAnimation(animConfig) {
    if (!avatarMixer || !animConfig) return;

    const clipName = animConfig.clipName;
    const clips    = avatarMixer._root.animations || [];
    const clip     = THREE.AnimationClip.findByName(clips, clipName);

    if (!clip) {
      console.warn(`[Renderer] Animation clip not found: ${clipName}`);
      return;
    }

    const newAction = avatarMixer.clipAction(clip);
    newAction.timeScale = animConfig.timeScale || 1.0;
    newAction.loop      = animConfig.loop ? THREE.LoopRepeat : THREE.LoopOnce;

    if (currentAction && currentAction !== newAction) {
      newAction.reset().play();
      currentAction.crossFadeTo(
        newAction,
        animConfig.fadeDuration || ANIMATION_FADE_DURATION,
        false
      );
    } else {
      newAction.play();
    }
    currentAction = newAction;
  }

  function setAnimation(key) {
    if (!avatarMixer) {
      // Demo mode: key → animation function
      const DEMO_ANIM_MAP = {
        walk:    startDemoWalkAnimation,
        rotate:  startDemoRotateAnimation,
        catwalk: startDemoWalkAnimation,
      };
      if (_demoAnimReq) { cancelAnimationFrame(_demoAnimReq); _demoAnimReq = null; }
      if (DEMO_ANIM_MAP[key]) DEMO_ANIM_MAP[key]();
      else if (avatarGroup) avatarGroup.rotation.y = 0;
      return;
    }
    const MIXAMO_MAP = {
      idle:    'Mixamo_Idle',
      walk:    'Mixamo_Walking',
      rotate:  'Mixamo_TurnLeft',
      pose_t:  'Mixamo_TPose',
      pose_a:  'Mixamo_APose',
      catwalk: 'Mixamo_CatwalkWalk',
    };
    playAnimation({
      clipName:     MIXAMO_MAP[key] || 'Mixamo_Idle',
      loop:         !['pose_t', 'pose_a'].includes(key),
      timeScale:    key === 'catwalk' ? 0.85 : 1.0,
      fadeDuration: ANIMATION_FADE_DURATION,
    });
  }

  // ── Scene Utilities ────────────────────────────────────────────────────────
  function centreAvatar(avatarScene) {
    const box  = new THREE.Box3().setFromObject(avatarScene);
    const size = box.getSize(new THREE.Vector3());
    const cent = box.getCenter(new THREE.Vector3());
    avatarScene.position.x -= cent.x;
    avatarScene.position.z -= cent.z;
    avatarScene.position.y -= box.min.y;

    // Adjust camera distance based on avatar height
    const avatarHeight = size.y;
    const dist = avatarHeight * 1.8;
    camera.position.set(0, avatarHeight * 0.55, dist);
    controls.target.set(0, avatarHeight * 0.5, 0);
    controls.update();
  }

  function clearAvatarGroup() {
    if (avatarMixer) { avatarMixer.stopAllAction(); avatarMixer = null; }
    if (_demoAnimReq) { cancelAnimationFrame(_demoAnimReq); _demoAnimReq = null; }
    if (avatarGroup)  { scene.remove(avatarGroup); avatarGroup = null; }
    currentAction = null;
  }

  function resetScene() {
    clearAvatarGroup();
    payloadCache = null;
  }

  // ── Responsive Canvas ──────────────────────────────────────────────────────
  function getAspect() {
    const canvas = document.getElementById('renderer-canvas');
    if (!canvas) return 1;
    return canvas.clientWidth / canvas.clientHeight;
  }

  function resizeRenderer() {
    const canvas = document.getElementById('renderer-canvas');
    if (!canvas || !renderer || !camera) return;
    const w = canvas.clientWidth  || canvas.offsetWidth  || 800;
    const h = canvas.clientHeight || canvas.offsetHeight || 600;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  // ── Loading Overlay ────────────────────────────────────────────────────────
  function showLoadingOverlay(msg) {
    const el = document.getElementById('loading-overlay');
    if (el) {
      el.style.display = 'flex';
      updateLoadingText(msg || 'Loading…');
    }
  }
  function hideLoadingOverlay() {
    const el = document.getElementById('loading-overlay');
    if (el) el.style.display = 'none';
  }
  function updateLoadingText(msg) {
    const el = document.getElementById('loading-text');
    if (el) el.textContent = msg;
  }

  // ── Public API ─────────────────────────────────────────────────────────────
  window.eWardrobeRenderer = {
    loadPayload,
    setAnimation,
    resetScene,
  };

  // ── Bootstrap ──────────────────────────────────────────────────────────────
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    init();
  } else {
    document.addEventListener('DOMContentLoaded', init);
  }

})();
