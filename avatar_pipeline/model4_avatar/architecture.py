"""
Model 4 — Avatar Generation: conditional VAE-GAN architecture.

- Encoder: avatar image -> (z_mean, z_log_var) for a latent code z.
- Decoder ("generator"): (z, condition) -> RGBA avatar image. `condition` is
  the concatenation of Model 1's body-shape one-hot, Model 2's pose vector,
  and Model 3's skin RGB (see condition_utils.build_condition_vector).
- Discriminator: (image, condition) -> real/fake logit.

Why VAE-GAN rather than a plain GAN or plain VAE: a plain VAE trained with
only pixel reconstruction + KL loss tends to produce blurry images; a plain
GAN is powerful but notoriously unstable to train from scratch (mode
collapse, no reliable convergence signal). The VAE-GAN (Larsen et al., 2016)
combines both — the reconstruction term anchors training and gives every
input image a corresponding latent code, while the adversarial term
sharpens the decoder's output. train.py implements the custom training loop.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers

from .condition_utils import CONDITION_DIM

LATENT_DIM = 64
IMG_SIZE = 128
IMG_CHANNELS = 4  # RGBA — alpha channel marks the avatar silhouette so
# Model 5 can composite clothing over a transparent background.


class Sampling(layers.Layer):
    """Reparameterization trick: z = mu + exp(0.5*logvar) * epsilon."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(img_size=IMG_SIZE, latent_dim=LATENT_DIM, condition_dim=CONDITION_DIM):
    img_in = layers.Input(shape=(img_size, img_size, IMG_CHANNELS), name="image")
    cond_in = layers.Input(shape=(condition_dim,), name="condition")

    x = layers.Rescaling(1.0 / 255.0)(img_in)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    cond_feat = layers.Dense(64, activation="relu")(cond_in)
    x = layers.Concatenate()([x, cond_feat])
    x = layers.Dense(256, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    return Model([img_in, cond_in], [z_mean, z_log_var], name="encoder")


def build_decoder(img_size=IMG_SIZE, latent_dim=LATENT_DIM, condition_dim=CONDITION_DIM):
    z_in = layers.Input(shape=(latent_dim,), name="z")
    cond_in = layers.Input(shape=(condition_dim,), name="condition")

    x = layers.Concatenate()([z_in, cond_in])
    init_size = img_size // 16  # 4 upsampling stages of x2 -> img_size
    x = layers.Dense(init_size * init_size * 256, activation="relu")(x)
    x = layers.Reshape((init_size, init_size, 256))(x)

    for filters in (128, 64, 32):
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(IMG_CHANNELS, 4, strides=2, padding="same", activation="sigmoid")(x)
    out = layers.Rescaling(255.0, name="rgba_0_255")(x)
    return Model([z_in, cond_in], out, name="decoder")


def build_discriminator(img_size=IMG_SIZE, condition_dim=CONDITION_DIM):
    img_in = layers.Input(shape=(img_size, img_size, IMG_CHANNELS), name="image")
    cond_in = layers.Input(shape=(condition_dim,), name="condition")

    x = layers.Rescaling(1.0 / 255.0)(img_in)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    cond_feat = layers.Dense(64, activation="relu")(cond_in)
    x = layers.Concatenate()([x, cond_feat])
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(0.2)(x)
    out = layers.Dense(1, name="logit")(x)
    return Model([img_in, cond_in], out, name="discriminator")
