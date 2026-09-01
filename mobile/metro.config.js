// eslint-disable-next-line @typescript-eslint/no-var-requires
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Register 3D model formats as binary assets so `require('../assets/avatars/male.glb')`
// resolves to a bundled asset (used by the realistic avatar viewer/builder).
config.resolver.assetExts.push('glb', 'gltf', 'bin');

module.exports = config;
