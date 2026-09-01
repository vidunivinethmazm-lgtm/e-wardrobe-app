"""
Model 5 — Virtual Try-On: training script for the GMM and TOM stages.

1. Preprocess a raw VITON / VITON-HD download:
    python -m avatar_pipeline.model5_tryon.preprocess_viton \
        --raw_dir path/to/viton --output_dir data/model5_tryon

2. Train the GMM (geometric matching / TPS warp):
    python -m avatar_pipeline.model5_tryon.train --stage gmm

3. Train the TOM (try-on module + PatchGAN), using the GMM from step 2:
    python -m avatar_pipeline.model5_tryon.train --stage tom \
        --gmm_model_path saved_models/model5_tryon/gmm.keras

The GMM is trained with a plain L1 loss against the ground-truth clothing
region (cloth_on_person / cloth_on_person_mask). The TOM is trained with a
combination of L1 + VGG perceptual loss + composition-mask regularization +
adversarial loss from a PatchGAN discriminator (the "...GAN" part of the
pipeline) — implemented as a custom train_step (TOMGAN), analogous to
Model 4's CVAEGAN.
"""

import argparse
import os

import tensorflow as tf
from tensorflow.keras import Model

from .architecture import (
    GRID_SIZE,
    IMG_HEIGHT,
    IMG_WIDTH,
    apply_gmm_warp,
    build_gmm_with_warp,
    build_tom,
    build_tom_discriminator,
)
from .data_pipeline import make_dataset


def _gmm_xy(sample):
    x = {
        "person_repr": sample["person_repr"],
        "cloth": sample["cloth"],
        "cloth_mask": sample["cloth_mask"],
    }
    y = tf.concat([sample["cloth_on_person"], sample["cloth_on_person_mask"]], axis=-1)
    return x, y


def build_vgg_feature_extractor(img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(img_height, img_width, 3))
    vgg.trainable = False
    layer_names = ["block1_conv2", "block2_conv2", "block3_conv3"]
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return Model(vgg.input, outputs, name="vgg_features")


class TOMGAN(tf.keras.Model):
    """Custom training loop for the TOM + PatchGAN discriminator. The GMM is
    pretrained and frozen — only used to produce `warped_cloth`/`warped_mask`
    on the fly from (person_repr, cloth, cloth_mask)."""

    def __init__(self, gmm, tom, discriminator, vgg,
                 l1_weight=1.0, perceptual_weight=1.0, mask_weight=1.0, adv_weight=0.1):
        super().__init__()
        self.gmm = gmm
        self.gmm.trainable = False
        self.tom = tom
        self.discriminator = discriminator
        self.vgg = vgg
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.mask_weight = mask_weight
        self.adv_weight = adv_weight

        self.l1_loss_tracker = tf.keras.metrics.Mean(name="l1_loss")
        self.perceptual_loss_tracker = tf.keras.metrics.Mean(name="perceptual_loss")
        self.mask_loss_tracker = tf.keras.metrics.Mean(name="mask_loss")
        self.gen_adv_loss_tracker = tf.keras.metrics.Mean(name="gen_adv_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")

    @property
    def metrics(self):
        return [
            self.l1_loss_tracker,
            self.perceptual_loss_tracker,
            self.mask_loss_tracker,
            self.gen_adv_loss_tracker,
            self.disc_loss_tracker,
        ]

    def compile(self, gen_optimizer, disc_optimizer):
        super().compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

    def _warp_cloth(self, person_repr, cloth, cloth_mask):
        control_points = self.gmm([person_repr, cloth, cloth_mask], training=False)
        return apply_gmm_warp(cloth, cloth_mask, control_points)

    def _vgg_perceptual_loss(self, a, b):
        a_feats = self.vgg(tf.keras.applications.vgg16.preprocess_input(a))
        b_feats = self.vgg(tf.keras.applications.vgg16.preprocess_input(b))
        return tf.add_n([tf.reduce_mean(tf.abs(fa - fb)) for fa, fb in zip(a_feats, b_feats)])

    def train_step(self, data):
        person_repr = data["person_repr"]
        cloth = data["cloth"]
        cloth_mask = data["cloth_mask"]
        person = data["person"]

        warped_cloth, warped_mask = self._warp_cloth(person_repr, cloth, cloth_mask)

        # --- Discriminator step ---
        with tf.GradientTape() as tape:
            rendered, comp_mask = self.tom([person_repr, warped_cloth, warped_mask], training=True)
            composite = comp_mask * warped_cloth + (1.0 - comp_mask) * rendered

            real_logits = self.discriminator([person_repr, person], training=True)
            fake_logits = self.discriminator([person_repr, composite], training=True)

            d_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.ones_like(real_logits), real_logits, from_logits=True)
            ) + tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_logits), fake_logits, from_logits=True)
            )

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.disc_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        # --- Generator (TOM) step ---
        with tf.GradientTape() as tape:
            rendered, comp_mask = self.tom([person_repr, warped_cloth, warped_mask], training=True)
            composite = comp_mask * warped_cloth + (1.0 - comp_mask) * rendered

            l1_loss = tf.reduce_mean(tf.abs(composite - person))
            perceptual_loss = self._vgg_perceptual_loss(composite, person)
            mask_loss = tf.reduce_mean(tf.abs(comp_mask - warped_mask / 255.0))

            fake_logits = self.discriminator([person_repr, composite], training=False)
            gen_adv_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.ones_like(fake_logits), fake_logits, from_logits=True)
            )

            total_loss = (
                self.l1_weight * l1_loss
                + self.perceptual_weight * perceptual_loss
                + self.mask_weight * mask_loss
                + self.adv_weight * gen_adv_loss
            )

        grads = tape.gradient(total_loss, self.tom.trainable_weights)
        self.gen_optimizer.apply_gradients(zip(grads, self.tom.trainable_weights))

        self.l1_loss_tracker.update_state(l1_loss)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.mask_loss_tracker.update_state(mask_loss)
        self.gen_adv_loss_tracker.update_state(gen_adv_loss)
        self.disc_loss_tracker.update_state(d_loss)

        return {m.name: m.result() for m in self.metrics}


def train_gmm(args):
    train_ds, val_ds = make_dataset(args.data_dir, batch_size=args.batch_size)
    train_ds = train_ds.map(_gmm_xy)
    val_ds = val_ds.map(_gmm_xy)

    gmm, wrapper = build_gmm_with_warp(grid_size=GRID_SIZE)
    wrapper.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mae")
    wrapper.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
    ]
    wrapper.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    gmm.save(os.path.join(args.output_dir, "gmm.keras"))
    gmm.export(os.path.join(args.output_dir, "gmm_savedmodel"))
    print(f"Saved GMM to {args.output_dir}")


def train_tom(args):
    train_ds, val_ds = make_dataset(args.data_dir, batch_size=args.batch_size)

    gmm = tf.keras.models.load_model(args.gmm_model_path)
    tom = build_tom()
    discriminator = build_tom_discriminator()
    vgg = build_vgg_feature_extractor()

    model = TOMGAN(gmm, tom, discriminator, vgg)
    model.compile(
        gen_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        disc_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
    )
    model.fit(train_ds, epochs=args.epochs)

    tom.save(os.path.join(args.output_dir, "tom.keras"))
    discriminator.save(os.path.join(args.output_dir, "discriminator.keras"))
    tom.export(os.path.join(args.output_dir, "tom_savedmodel"))
    print(f"Saved TOM to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["gmm", "tom"], required=True)
    parser.add_argument("--data_dir", default="data/model5_tryon")
    parser.add_argument("--output_dir", default="saved_models/model5_tryon")
    parser.add_argument("--gmm_model_path", default="saved_models/model5_tryon/gmm.keras")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.stage == "gmm":
        train_gmm(args)
    else:
        train_tom(args)


if __name__ == "__main__":
    main()
