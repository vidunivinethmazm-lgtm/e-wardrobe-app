import sys
import numpy as np
from pygltflib import GLTF2

path = sys.argv[1]
gltf = GLTF2.load(path)
blob = gltf.binary_blob()
prim = gltf.meshes[0].primitives[0]
acc = gltf.accessors[prim.attributes.POSITION]
bv = gltf.bufferViews[acc.bufferView]
offset = bv.byteOffset + (acc.byteOffset or 0)
count = acc.count
pos = np.frombuffer(blob[offset:offset + count * 12], dtype=np.float32).reshape(count, 3)

y = pos[:, 1]
height = y.max() - y.min()
ymin = y.min()


def cross_section_width(y_frac_lo, y_frac_hi):
    lo, hi = ymin + y_frac_lo * height, ymin + y_frac_hi * height
    band = (y >= lo) & (y <= hi)
    x_band, z_band = pos[band, 0], pos[band, 2]
    # Exclude hands/arms (relaxed A-pose puts them far from center at torso
    # height) - keep only the central torso cluster, same fix as before.
    torso = np.abs(x_band) < 0.3
    if torso.sum() == 0:
        return float("nan")
    radial = np.hypot(x_band[torso], z_band[torso])
    return radial.max() * 2  # diameter


# Approximate height bands for bust/waist/hip on this rig (consistent with
# the 0.45-0.55 hip band used previously; bust sits higher, waist between).
bust = cross_section_width(0.62, 0.68)
waist = cross_section_width(0.53, 0.58)
hip = cross_section_width(0.45, 0.50)

name = path.split("/")[-1].replace(".glb", "")
print(f"{name:28} height={height:.4f}  bust={bust:.4f}  waist={waist:.4f}  hip={hip:.4f}  "
      f"bust-hip={bust-hip:+.4f}  waist-bust={waist-bust:+.4f}  waist-hip={waist-hip:+.4f}")
