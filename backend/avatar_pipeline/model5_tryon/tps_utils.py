"""
Model 5 — Virtual Try-On: thin-plate-spline (TPS) warping utilities.

Two implementations are provided:

- `tps_warp` (numpy + OpenCV): classical, non-differentiable backward warp
  given explicit source/destination control points. Used by predict.py's
  `try_on_avatar` to warp a clothing image onto Model 4's stylized avatar
  using only geometry (pose keypoints) — no trained model required.

- `TPSGridGen` + `bilinear_sampler` (TensorFlow, differentiable): the
  CP-VTON-style learned warp used inside the GMM (architecture.build_gmm).
  The GMM predicts where a fixed grid of control points should move to;
  `TPSGridGen` turns that into a dense per-pixel sampling grid, and
  `bilinear_sampler` resamples the clothing image/mask through that grid.
"""

import cv2
import numpy as np

# `tensorflow` is imported lazily inside TPSGridGen/bilinear_sampler so that
# `tps_warp` (the classical numpy/OpenCV path used by predict.py's
# try_on_avatar) works without a TensorFlow install.


# ---------------------------------------------------------------------------
# Classical TPS (numpy / OpenCV) — used for avatar try-on (predict.py)
# ---------------------------------------------------------------------------

def _tps_kernel(r):
    """U(r) = r^2 * log(r), with U(0) := 0."""
    r_safe = np.where(r == 0, 1.0, r)
    return np.where(r == 0, 0.0, (r_safe ** 2) * np.log(r_safe))


def _tps_solve(dst_points, values):
    """Solves for TPS weights `w` (N,) and affine params `a` (3,) such that
    f(dst_points[i]) ~= values[i], where
    f(x, y) = a[0] + a[1]*x + a[2]*y + sum_i w[i] * U(|| (x,y) - dst_points[i] ||)
    """
    n = len(dst_points)
    diff = dst_points[:, None, :] - dst_points[None, :, :]
    r = np.sqrt((diff ** 2).sum(-1))
    k = _tps_kernel(r)
    p = np.hstack([np.ones((n, 1)), dst_points])

    l = np.zeros((n + 3, n + 3))
    l[:n, :n] = k
    l[:n, n:] = p
    l[n:, :n] = p.T

    rhs = np.zeros(n + 3)
    rhs[:n] = values

    params = np.linalg.solve(l + np.eye(n + 3) * 1e-6, rhs)
    return params[:n], params[n:]


def tps_warp(image, src_points, dst_points, output_size, border_value=0):
    """Backward-warps `image` so that `dst_points` (coordinates in the
    OUTPUT image) sample from `src_points` (corresponding coordinates in
    `image`).

    image: HxWx3 or HxWx1 array (the clothing image or its mask)
    src_points, dst_points: (N, 2) arrays of (x, y) pixel coordinates, N >= 3
    output_size: (height, width) of the result
    Returns an array of shape (*output_size, channels).
    """
    src_points = np.asarray(src_points, dtype=np.float64)
    dst_points = np.asarray(dst_points, dtype=np.float64)
    h_out, w_out = output_size

    wx, ax = _tps_solve(dst_points, src_points[:, 0])
    wy, ay = _tps_solve(dst_points, src_points[:, 1])

    grid_y, grid_x = np.mgrid[0:h_out, 0:w_out].astype(np.float64)
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (H*W, 2)

    diff = pts[:, None, :] - dst_points[None, :, :]
    r = np.sqrt((diff ** 2).sum(-1))
    u = _tps_kernel(r)  # (H*W, N)

    map_x = ax[0] + ax[1] * pts[:, 0] + ax[2] * pts[:, 1] + u @ wx
    map_y = ay[0] + ay[1] * pts[:, 0] + ay[2] * pts[:, 1] + u @ wy

    map_x = map_x.reshape(h_out, w_out).astype(np.float32)
    map_y = map_y.reshape(h_out, w_out).astype(np.float32)

    return cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


# ---------------------------------------------------------------------------
# Differentiable TPS (TensorFlow) — used inside the GMM (training)
# ---------------------------------------------------------------------------

class TPSGridGen:
    """Precomputes everything that depends only on a fixed regular
    `grid_size` x `grid_size` grid of TARGET control points in [-1, 1], so
    that at call time it only needs the GMM's predicted SOURCE control
    points to produce a dense (B, H, W, 2) sampling grid.

    Follows the formulation used in CP-VTON's TpsGridGen (originally
    PyTorch), adapted to TensorFlow.
    """

    def __init__(self, target_height, target_width, grid_size=5):
        import tensorflow as tf

        self.h = target_height
        self.w = target_width
        self.n = grid_size * grid_size

        axis = np.linspace(-1, 1, grid_size)
        gx, gy = np.meshgrid(axis, axis)
        target_control_points = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)  # (N, 2)

        k = self._kernel(target_control_points, target_control_points)
        p = np.hstack([np.ones((self.n, 1)), target_control_points])
        l = np.zeros((self.n + 3, self.n + 3))
        l[: self.n, : self.n] = k
        l[: self.n, self.n :] = p
        l[self.n :, : self.n] = p.T
        l_inv = np.linalg.inv(l + np.eye(self.n + 3) * 1e-6)
        self.l_inv = tf.constant(l_inv, dtype=tf.float32)  # (N+3, N+3)

        gy2, gx2 = np.mgrid[0:target_height, 0:target_width].astype(np.float64)
        gx2 = gx2 / (target_width - 1) * 2 - 1
        gy2 = gy2 / (target_height - 1) * 2 - 1
        out_points = np.stack([gx2.ravel(), gy2.ravel()], axis=1)  # (H*W, 2)

        u = self._kernel(out_points, target_control_points)  # (H*W, N)
        p_out = np.hstack([np.ones((out_points.shape[0], 1)), out_points])  # (H*W, 3)
        # (H*W, N+3)
        self.out_basis = tf.constant(np.hstack([u, p_out]), dtype=tf.float32)

    @staticmethod
    def _kernel(p1, p2):
        d = np.sqrt(((p1[:, None, :] - p2[None, :, :]) ** 2).sum(-1))
        d_safe = np.where(d == 0, 1.0, d)
        return np.where(d == 0, 0.0, (d_safe ** 2) * np.log(d_safe))

    def __call__(self, source_control_points):
        """source_control_points: (B, N, 2) predicted positions (in [-1, 1])
        that each fixed target control point should map to.
        Returns a sampling grid (B, H, W, 2) in [-1, 1], (x, y) order.
        """
        import tensorflow as tf

        batch_size = tf.shape(source_control_points)[0]
        zeros = tf.zeros((batch_size, 3, 2), dtype=tf.float32)
        y = tf.concat([source_control_points, zeros], axis=1)  # (B, N+3, 2)

        params = tf.matmul(tf.tile(self.l_inv[None], (batch_size, 1, 1)), y)  # (B, N+3, 2)
        mapped = tf.matmul(tf.tile(self.out_basis[None], (batch_size, 1, 1)), params)  # (B, H*W, 2)
        return tf.reshape(mapped, (batch_size, self.h, self.w, 2))


def bilinear_sampler(image, grid):
    """Differentiable bilinear sampling (spatial-transformer style).

    image: (B, Hin, Win, C)
    grid: (B, Hout, Wout, 2), (x, y) in [-1, 1]
    Returns (B, Hout, Wout, C).
    """
    import tensorflow as tf

    shape = tf.shape(image)
    b, h_in, w_in = shape[0], shape[1], shape[2]

    x = (grid[..., 0] + 1.0) * tf.cast(w_in - 1, tf.float32) / 2.0
    y = (grid[..., 1] + 1.0) * tf.cast(h_in - 1, tf.float32) / 2.0

    x0 = tf.floor(x)
    x1 = x0 + 1.0
    y0 = tf.floor(y)
    y1 = y0 + 1.0

    x0c = tf.clip_by_value(x0, 0, tf.cast(w_in - 1, tf.float32))
    x1c = tf.clip_by_value(x1, 0, tf.cast(w_in - 1, tf.float32))
    y0c = tf.clip_by_value(y0, 0, tf.cast(h_in - 1, tf.float32))
    y1c = tf.clip_by_value(y1, 0, tf.cast(h_in - 1, tf.float32))

    def gather(xc, yc):
        idx = tf.stack([tf.cast(yc, tf.int32), tf.cast(xc, tf.int32)], axis=-1)
        return tf.gather_nd(image, idx, batch_dims=1)

    ia = gather(x0c, y0c)
    ib = gather(x0c, y1c)
    ic = gather(x1c, y0c)
    id_ = gather(x1c, y1c)

    wa = ((x1 - x) * (y1 - y))[..., None]
    wb = ((x1 - x) * (y - y0))[..., None]
    wc = ((x - x0) * (y1 - y))[..., None]
    wd = ((x - x0) * (y - y0))[..., None]

    return wa * ia + wb * ib + wc * ic + wd * id_
