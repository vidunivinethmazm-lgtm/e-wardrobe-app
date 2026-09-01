import { Asset } from 'expo-asset';

/**
 * Resolves a `require()`'d module id (e.g. a bundled `.glb`) to a URI that
 * `AvatarViewer3D`'s `garmentMeshUrls`/`remoteAvatarUrl` props can load —
 * those are typed as remote-looking URL strings, but a downloaded local
 * asset's `file://` URI works exactly the same way.
 */
export async function resolveModuleUri(moduleId: number): Promise<string> {
  const asset = Asset.fromModule(moduleId);
  await asset.downloadAsync();
  return asset.localUri ?? asset.uri;
}
