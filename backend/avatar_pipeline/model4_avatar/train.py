"""
Model 4 — Avatar Generation: training script for the conditional VAE-GAN.

1. Generate the synthetic training set (paper-doll avatars + condition
   vectors):
    python -m avatar_pipeline.model4_avatar.synthetic_avatars \
        --output_dir data/model4_avatar --n_samples 4000

2. Train:
    python -m avatar_pipeline.model4_avatar.train

Each training step:
  (a) updates the discriminator on real vs. decoder-reconstructed images
      (both conditioned on the same condition vector as the real image), then
  (b) updates the encoder+decoder on reconstruction loss + KL divergence +
      adversarial loss (fool the discriminator).

Every `--sample_every` epochs, a grid of avatars sampled from the prior
(z ~ N(0, I)) at a handful of fixed condition vectors is written to
`<output_dir>/samples/`, which is the main way to judge training progress —
generative models don't have a single scalar "accuracy".
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from .architecture import (
    IMG_SIZE,
    LATENT_DIM,
    Sampling,
    build_decoder,
    build_discriminator,
    build_encoder,
)
from .data_pipeline import make_dataset


class CVAEGAN(tf.keras.Model):
    def __init__(self, encoder, decoder, discriminator, kl_weight=1e-3, adv_weight=0.1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.sampling = Sampling()
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight

        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.gen_adv_loss_tracker = tf.keras.metrics.Mean(name="gen_adv_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")

    @property
    def metrics(self):
        return [
            self.recon_loss_tracker,
            self.kl_loss_tracker,
            self.gen_adv_loss_tracker,
            self.disc_loss_tracker,
        ]

    def compile(self, gen_optimizer, disc_optimizer):
        super().compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

    def train_step(self, data):
        images, conditions = data

        # --- Discriminator step ---
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder([images, conditions])
            z = self.sampling([z_mean, z_log_var])
            fake_images = self.decoder([z, conditions])

            real_logits = self.discriminator([images, conditions])
            fake_logits = self.discriminator([fake_images, conditions])

            d_loss_real = tf.keras.losses.binary_crossentropy(
                tf.ones_like(real_logits), real_logits, from_logits=True
            )
            d_loss_fake = tf.keras.losses.binary_crossentropy(
                tf.zeros_like(fake_logits), fake_logits, from_logits=True
            )
            d_loss = tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake)

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.disc_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        # --- Encoder + decoder step ---
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder([images, conditions])
            z = self.sampling([z_mean, z_log_var])
            fake_images = self.decoder([z, conditions])

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(images - fake_images), axis=[1, 2, 3])
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )

            fake_logits = self.discriminator([fake_images, conditions])
            gen_adv_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    tf.ones_like(fake_logits), fake_logits, from_logits=True
                )
            )

            total_loss = recon_loss + self.kl_weight * kl_loss + self.adv_weight * gen_adv_loss

        trainable = self.encoder.trainable_weights + self.decoder.trainable_weights
        grads = tape.gradient(total_loss, trainable)
        self.gen_optimizer.apply_gradients(zip(grads, trainable))

        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.gen_adv_loss_tracker.update_state(gen_adv_loss)
        self.disc_loss_tracker.update_state(d_loss)

        return {m.name: m.result() for m in self.metrics}


class SampleAvatarCallback(tf.keras.callbacks.Callback):
    """Every `every` epochs, decode a fixed set of condition vectors from
    fresh prior samples z ~ N(0, I) and save an image grid for visual
    inspection."""

    def __init__(self, decoder, sample_conditions, output_dir, every=10):
        super().__init__()
        self.decoder = decoder
        self.sample_conditions = tf.constant(sample_conditions, dtype=tf.float32)
        self.output_dir = output_dir
        self.every = every
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every != 0:
            return
        n = self.sample_conditions.shape[0]
        z = tf.random.normal((n, LATENT_DIM))
        images = self.decoder([z, self.sample_conditions], training=False).numpy()
        images = np.clip(images, 0, 255).astype("uint8")

        grid = Image.new("RGBA", (IMG_SIZE * n, IMG_SIZE))
        for i in range(n):
            grid.paste(Image.fromarray(images[i], "RGBA"), (i * IMG_SIZE, 0))
        grid.save(os.path.join(self.output_dir, f"epoch_{epoch + 1:04d}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/model4_avatar")
    parser.add_argument("--output_dir", default="saved_models/model4_avatar")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--kl_weight", type=float, default=1e-3)
    parser.add_argument("--adv_weight", type=float, default=0.1)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--n_samples_grid", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_ds, val_ds = make_dataset(args.data_dir, img_size=args.img_size, batch_size=args.batch_size)

    encoder = build_encoder(img_size=args.img_size)
    decoder = build_decoder(img_size=args.img_size)
    discriminator = build_discriminator(img_size=args.img_size)
    encoder.summary()
    decoder.summary()
    discriminator.summary()

    model = CVAEGAN(encoder, decoder, discriminator, kl_weight=args.kl_weight, adv_weight=args.adv_weight)
    model.compile(
        gen_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        disc_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
    )

    # Fix a handful of condition vectors (from the validation split) for
    # consistent before/after-training visual comparisons.
    sample_conditions = next(iter(val_ds.unbatch().batch(args.n_samples_grid)))[1].numpy()
    sample_cb = SampleAvatarCallback(
        decoder, sample_conditions, os.path.join(args.output_dir, "samples"), every=args.sample_every
    )

    model.fit(train_ds, epochs=args.epochs, callbacks=[sample_cb])

    encoder.save(os.path.join(args.output_dir, "encoder.keras"))
    decoder.save(os.path.join(args.output_dir, "decoder.keras"))
    discriminator.save(os.path.join(args.output_dir, "discriminator.keras"))
    decoder.export(os.path.join(args.output_dir, "decoder_savedmodel"))

    print(f"Saved model artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
