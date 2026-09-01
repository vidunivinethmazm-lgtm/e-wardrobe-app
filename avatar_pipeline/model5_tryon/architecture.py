"""
Model 5 — Virtual Try-On: GMM (geometric matching) + TOM (try-on module) +
PatchGAN discriminator, following the CP-VTON / VITON two-stage design.

Stage 1 — GMM (`build_gmm`): looks at the clothing image/mask and the
target `person_repr` (pose + body silhouette + skin, see pose_repr.py) and
predicts a thin-plate-spline warp that aligns the clothing to the target
body — without ever seeing the person's actual photo, so it can't cheat by
copying pixels.

Stage 2 — TOM (`build_tom`): a U-Net that takes (person_repr, warped_cloth,
warped_cloth_mask) and predicts a rendered body layer plus a composition
mask. The final image blends the warped clothing over the rendered body:
    output = mask * warped_cloth + (1 - mask) * rendered_person

`build_tom_discriminator` is a PatchGAN used to add an adversarial loss on
top of TOM's L1/perceptual losses ("...GAN pipeline").

Train both stages on a paired dataset (VITON / VITON-HD) via train.py. For
inference onto Model 4's stylized avatar, predict.py instead uses the
classical (non-learned) `tps_utils.tps_warp` driven directly by pose
keypoints — see predict.py's docstring for why.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers

from .pose_repr import PERSON_REPR_CHANNELS
from .tps_utils import TPSGridGen, bilinear_sampler

GRID_SIZE = 5  # 5x5 TPS control point grid (CP-VTON default)
IMG_HEIGHT = 256
IMG_WIDTH = 192


def _conv_block(x, filters, stride=2, batchnorm=True):
    x = layers.Conv2D(filters, 4, strides=stride, padding="same")(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(0.2)(x)


def _correlate(inputs):
    """Normalized correlation between every spatial location of two equally
    shaped feature maps. (B,H,W,C) x2 -> (B,H,W,H*W)."""
    f_a, f_b = inputs
    f_a = tf.math.l2_normalize(f_a, axis=-1)
    f_b = tf.math.l2_normalize(f_b, axis=-1)
    shape = tf.shape(f_a)
    b, h, w, c = shape[0], shape[1], shape[2], shape[3]
    f_a = tf.reshape(f_a, (b, h * w, c))
    f_b = tf.reshape(f_b, (b, h * w, c))
    corr = tf.matmul(f_a, f_b, transpose_b=True)  # (B, HW, HW)
    return tf.reshape(corr, (b, h, w, h * w))


def build_gmm(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, grid_size=GRID_SIZE,
               person_repr_channels=PERSON_REPR_CHANNELS):
    """{"person_repr": (H,W,C_p), "cloth": (H,W,3), "cloth_mask": (H,W,1)}
    -> "control_points": (grid_size*grid_size, 2) in [-1, 1] — the predicted
    SOURCE control points for `tps_utils.TPSGridGen`.
    """
    person_in = layers.Input(shape=(img_height, img_width, person_repr_channels), name="person_repr")
    cloth_in = layers.Input(shape=(img_height, img_width, 3), name="cloth")
    mask_in = layers.Input(shape=(img_height, img_width, 1), name="cloth_mask")

    def tower(x):
        for filters in (64, 128, 256, 256):
            x = _conv_block(x, filters)
        return x

    person_feat = tower(person_in)
    cloth_feat = tower(layers.Concatenate()([layers.Rescaling(1.0 / 255.0)(cloth_in), mask_in]))

    corr = layers.Lambda(_correlate)([person_feat, cloth_feat])
    x = _conv_block(corr, 256, stride=1)
    x = _conv_block(x, 128, stride=2)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    offsets = layers.Dense(grid_size * grid_size * 2, activation="tanh", name="offsets")(x)
    control_points = layers.Reshape((grid_size * grid_size, 2), name="control_points")(offsets)

    return Model([person_in, cloth_in, mask_in], control_points, name="gmm")


def apply_gmm_warp(cloth, cloth_mask, control_points, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, grid_size=GRID_SIZE):
    """Warps `cloth`/`cloth_mask` (B,H,W,C) using the GMM's predicted
    `control_points` (B, grid_size*grid_size, 2). Returns
    (warped_cloth, warped_cloth_mask), same shapes as the inputs.

    Builds a fresh `TPSGridGen` each call — cheap (pure numpy precompute),
    and keeps this function stateless/side-effect free for use in both the
    GMM and TOM training loops.
    """
    grid_gen = TPSGridGen(img_height, img_width, grid_size=grid_size)
    grid = grid_gen(control_points)
    warped_cloth = bilinear_sampler(cloth, grid)
    warped_mask = bilinear_sampler(cloth_mask, grid)
    return warped_cloth, warped_mask


def build_gmm_with_warp(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, grid_size=GRID_SIZE,
                          person_repr_channels=PERSON_REPR_CHANNELS):
    """Wraps `build_gmm` so the model directly outputs the warped
    cloth/mask (concatenated on the channel axis: 3 + 1 = 4 channels),
    which is what `train.py --stage gmm` supervises with an L1 loss against
    (cloth_on_person, cloth_on_person_mask).

    Returns (gmm, wrapper) — `gmm` for saving/loading and reuse inside the
    TOM training loop, `wrapper` for `model.fit`.
    """
    gmm = build_gmm(img_height, img_width, grid_size, person_repr_channels)
    person_in, cloth_in, mask_in = gmm.inputs
    control_points = gmm.output

    warped_cloth, warped_mask = apply_gmm_warp(cloth_in, mask_in, control_points, img_height, img_width, grid_size)
    warped = layers.Concatenate(name="warped_cloth_and_mask")([warped_cloth, warped_mask])
    wrapper = Model([person_in, cloth_in, mask_in], warped, name="gmm_with_warp")
    return gmm, wrapper


def build_tom(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, person_repr_channels=PERSON_REPR_CHANNELS):
    """{"person_repr": (H,W,C_p), "warped_cloth": (H,W,3), "warped_cloth_mask": (H,W,1)}
    -> {"rendered_person": (H,W,3) in [0,255], "composition_mask": (H,W,1) in [0,1]}

    Final composite (computed by the caller / loss function):
        output = composition_mask * warped_cloth + (1 - composition_mask) * rendered_person
    """
    person_in = layers.Input(shape=(img_height, img_width, person_repr_channels), name="person_repr")
    cloth_in = layers.Input(shape=(img_height, img_width, 3), name="warped_cloth")
    mask_in = layers.Input(shape=(img_height, img_width, 1), name="warped_cloth_mask")

    x = layers.Concatenate()([person_in, layers.Rescaling(1.0 / 255.0)(cloth_in), mask_in])

    skips = []
    for filters in (64, 128, 256, 512, 512):
        x = _conv_block(x, filters)
        skips.append(x)

    skips = skips[:-1][::-1]  # 4 skip connections, deepest-to-shallowest reversed -> shallow order
    for filters, skip in zip((512, 256, 128, 64), skips):
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Concatenate()([x, skip])

    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    rendered = layers.Conv2D(3, 4, padding="same", activation="sigmoid")(x)
    rendered = layers.Rescaling(255.0, name="rendered_person")(rendered)
    composition_mask = layers.Conv2D(1, 4, padding="same", activation="sigmoid", name="composition_mask")(x)

    return Model([person_in, cloth_in, mask_in], [rendered, composition_mask], name="tom")


def build_tom_discriminator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, person_repr_channels=PERSON_REPR_CHANNELS):
    """PatchGAN: {"person_repr": (H,W,C_p), "image": (H,W,3)} -> patch logits.
    `image` is either a real ground-truth person photo or TOM's composite
    output, both conditioned on the same `person_repr`.
    """
    person_in = layers.Input(shape=(img_height, img_width, person_repr_channels), name="person_repr")
    img_in = layers.Input(shape=(img_height, img_width, 3), name="image")

    x = layers.Concatenate()([person_in, layers.Rescaling(1.0 / 255.0)(img_in)])
    for i, filters in enumerate((64, 128, 256, 512)):
        x = _conv_block(x, filters, batchnorm=(i != 0))
    out = layers.Conv2D(1, 4, padding="same", name="patch_logits")(x)

    return Model([person_in, img_in], out, name="tom_discriminator")
