"""
Derives assets/avatars/rp/female_base_mesh_v2.glb as a reshaped COPY of
male_base_mesh.glb, instead of the previous unrelated Sketchfab female scan
— so the male and female avatars share the same topology/pose/style, and
"female" reads as a body-shape variant of the same character rather than a
completely different model.

male_base_mesh.glb ("kakashi") is a low-poly stylized humanoid: standing,
arms hanging with hands resting around hip/upper-thigh height (verified by
rendering an x/z and y/z scatter of its raw vertex positions — see
`_debug_render` below), local axes X=width, Y=depth, Z=height (0 at the
feet, ~1.81 at the top of the head; the Sketchfab Y-up conversion happens at
the node-transform level, not in the vertex data itself, so this script
works directly on the raw accessor floats).

Approach: for the main body mesh only (a small unrelated "Sphere" prop mesh
is copied through untouched), scale each vertex's (x, y) horizontally by a
height-dependent factor built from a handful of anatomical control points
(hip/waist/bust/shoulder), smoothly interpolated so there's no visible seam
between bands. The scale is additionally gated down to ~0 for vertices far
from the central column (|x| large) so it fades out over the arms/hands
instead of also shoving them sideways — those vertices happen to sit at the
same height as the hip/waist band in this particular pose.
"""

import struct
import numpy as np
from pygltflib import GLTF2, BufferView, Accessor

SRC = "assets/avatars/rp/male_base_mesh.glb"
DST = "assets/avatars/rp/female_base_mesh_v2.glb"
BODY_MESH_NAME = "kakashi_Default OBJ.001_0"

# (height fraction t, width scale, depth scale) control points for the main
# body mesh, piecewise-linearly interpolated over t. t=0 is the feet, t=1
# the top of the head. Kept close to 1.0 above the shoulders so the face is
# never touched.
CONTROL_POINTS = [
    (0.00, 1.00, 1.00),  # feet
    (0.30, 1.00, 1.00),  # shin — unaffected
    (0.38, 1.14, 1.08),  # hip start
    (0.46, 1.45, 1.22),  # hip peak — wider hips
    (0.55, 0.68, 0.74),  # waist — tapered in
    (0.62, 0.84, 0.90),  # underbust
    (0.68, 0.95, 1.45),  # bust — fuller chest (front+back, symmetric)
    (0.76, 0.80, 0.92),  # shoulder — narrower
    (0.84, 1.00, 1.00),  # neck and up — unaffected (face/head untouched)
    (1.00, 1.00, 1.00),
]

# Horizontal (|x|) gating so the deformation only affects the torso/hip
# column, fading to a no-op before it reaches the arms/hands — which, in
# this mesh's resting pose, hang at the same height as the hip/waist band.
# Wide enough that straps/gear crossing the torso move with the reshape
# instead of staying frozen in the original silhouette.
TORSO_FULL_RADIUS = 0.22
TORSO_FADE_RADIUS = 0.34


def get_accessor_data(g, acc_idx):
    acc = g.accessors[acc_idx]
    bv = g.bufferViews[acc.bufferView]
    blob = g.binary_blob()
    offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    comp_type_map = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
    type_count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc.type]
    fmt = comp_type_map[acc.componentType]
    size = struct.calcsize(fmt)
    stride = bv.byteStride or (size * type_count)
    data = np.zeros((acc.count, type_count), dtype=np.float64)
    for i in range(acc.count):
        base = offset + i * stride
        for j in range(type_count):
            data[i, j] = struct.unpack_from("<" + fmt, blob, base + j * size)[0]
    return data


def torso_weight(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    w = 1.0 - (ax - TORSO_FULL_RADIUS) / (TORSO_FADE_RADIUS - TORSO_FULL_RADIUS)
    w = np.clip(w, 0.0, 1.0)
    # smoothstep, so the fade-out itself has no visible kink
    return w * w * (3 - 2 * w)


def spine_curve(t: np.ndarray, y: np.ndarray, weight: np.ndarray, bins=30) -> np.ndarray:
    """Smoothed torso-weighted mean Y per height bin, so the depth scale is
    applied relative to the body's own natural front-back curve rather than
    a fixed y=0 (which would otherwise shift the whole silhouette forward or
    back wherever the curve departs from zero)."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    means = np.zeros(bins)
    have_data = np.zeros(bins, dtype=bool)
    for i in range(bins):
        m = (t >= edges[i]) & (t < edges[i + 1]) & (weight > 0.5)
        if m.sum() >= 5:
            means[i] = np.average(y[m], weights=weight[m])
            have_data[i] = True
    # fill bins with too little torso data from their nearest valid neighbor
    valid_idx = np.flatnonzero(have_data)
    if len(valid_idx) == 0:
        return np.zeros_like(t)
    for i in range(bins):
        if not have_data[i]:
            nearest = valid_idx[np.argmin(np.abs(valid_idx - i))]
            means[i] = means[nearest]
    return np.interp(t, centers, means)


def feminize(pos: np.ndarray) -> np.ndarray:
    x, y, z = pos[:, 0].copy(), pos[:, 1].copy(), pos[:, 2].copy()
    zmin, zmax = z.min(), z.max()
    t = (z - zmin) / (zmax - zmin)

    cps = np.array(CONTROL_POINTS)
    width_scale = np.interp(t, cps[:, 0], cps[:, 1])
    depth_scale = np.interp(t, cps[:, 0], cps[:, 2])

    weight = torso_weight(x)
    eff_width = 1.0 + weight * (width_scale - 1.0)
    eff_depth = 1.0 + weight * (depth_scale - 1.0)

    cy = spine_curve(t, y, weight)

    new_x = x * eff_width
    new_y = (y - cy) * eff_depth + cy
    return np.stack([new_x, new_y, z], axis=1)


def pack_positions(values: np.ndarray) -> bytes:
    out = bytearray()
    for row in values:
        out += struct.pack("<fff", float(row[0]), float(row[1]), float(row[2]))
    return bytes(out)


def main():
    g = GLTF2().load(SRC)
    blob = bytearray(g.binary_blob())

    body_mesh = next(m for m in g.meshes if m.name == BODY_MESH_NAME)
    prim = body_mesh.primitives[0]
    pos_acc_idx = prim.attributes.POSITION
    pos = get_accessor_data(g, pos_acc_idx)
    new_pos = feminize(pos)

    print("original bounds", pos.min(axis=0), pos.max(axis=0))
    print("feminized bounds", new_pos.min(axis=0), new_pos.max(axis=0))

    acc = g.accessors[pos_acc_idx]
    bv = g.bufferViews[acc.bufferView]
    offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    packed = pack_positions(new_pos)
    if len(packed) != acc.count * 12:
        raise RuntimeError("packed size mismatch")
    blob[offset : offset + len(packed)] = packed

    # Recomputed bounds so viewers/loaders that trust accessor min/max
    # (three.js's GLTFLoader does, for the initial bounding box) see the
    # new silhouette instead of the original one.
    acc.min = new_pos.min(axis=0).tolist()
    acc.max = new_pos.max(axis=0).tolist()

    g.set_binary_blob(bytes(blob))
    g.save(DST)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
