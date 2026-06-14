#!/usr/bin/env python3
"""GAN"""
import tensorflow as tf


def train_step(self, useless_argument):
        """GAN"""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as discr_tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)
                discr_loss = self.discriminator.loss(
                    self.discriminator(real_samples, training=True),
                    self.discriminator(fake_samples, training=True)
                )
            discr_gradients = discr_tape.gradient(discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradients, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as gen_tape:
            fake_samples = self.get_fake_sample(training=True)
            gen_loss = self.generator.loss(
                self.discriminator(fake_samples, training=False)
            )
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
