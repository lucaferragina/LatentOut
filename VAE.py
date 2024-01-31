import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder():
    def build(layers_dim, encoder_inputs):
        x = layers.Dense(layers_dim[1],activation="relu")(encoder_inputs)
        x = layers.BatchNormalization()(x)
        for i in range(2, len(layers_dim)):
            x = layers.Dense(layers_dim[i],activation="relu")(x)
            x = layers.BatchNormalization()(x)

        z_mean = layers.Dense(layers_dim[-1], name="z_mean")(x)
        z_log_var = layers.Dense(layers_dim[-1], name="z_log_var")(x)
        mv = Model(encoder_inputs, [z_mean, z_log_var])
        z = Sampling()([z_mean, z_log_var])
        return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


class Decoder():
    def build(layers_dim, latent_inputs):
        x = layers.Dense(layers_dim[1], activation="relu")(latent_inputs)
        x = layers.BatchNormalization()(x)
        for i in range(2,len(layers_dim)):
            x = layers.Dense(layers_dim[i], activation="relu")(x)
            x = layers.BatchNormalization()(x)

        return keras.Model(latent_inputs, x, name="decoder")


class VariationalAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VariationalAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= data.shape[1]
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            beta = 10**(-4) # TODO: beta as a parameter
            total_loss = reconstruction_loss + beta*kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
        }