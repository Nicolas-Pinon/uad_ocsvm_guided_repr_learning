import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, MaxPooling2D, UpSampling2D, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, ReLU
import cvxpy as cp
from cvxpylayers.tensorflow import CvxpyLayer


class Autoencoder(tf.keras.Model):
    def __init__(self, latent_dim=32):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder layers
        self.encoder_conv1 = Conv2D(4, (5, 5), activation=None, padding='same')
        self.encoder_bn1 = BatchNormalization()
        self.encoder_leaky_relu1 = LeakyReLU()
        self.encoder_max_pool1 = MaxPooling2D(pool_size=(2, 2))

        self.encoder_conv2 = Conv2D(8, (5, 5), activation=None, padding='same')
        self.encoder_bn2 = BatchNormalization()
        self.encoder_leaky_relu2 = LeakyReLU()
        self.encoder_max_pool2 = MaxPooling2D(pool_size=(2, 2))

        self.flatten = Flatten()
        self.encoder_latent = Dense(latent_dim, activation=None)

        # Decoder layers
        self.decoder_dense = Dense(7 * 7 * 8, activation=None)
        self.decoder_reshape = Reshape((7 , 7, 8))

        self.decoder_upsample1 = UpSampling2D(size=(2, 2))
        self.decoder_conv2d_transpose1 = Conv2DTranspose(8, (5, 5), activation=None, padding='same')
        self.decoder_bn1 = BatchNormalization()
        self.decoder_leaky_relu1 = LeakyReLU()

        self.decoder_upsample2 = UpSampling2D(size=(2, 2))
        self.decoder_conv2d_transpose2 = Conv2DTranspose(4, (5, 5), activation=None, padding='same')
        self.decoder_bn2 = BatchNormalization()
        self.decoder_leaky_relu2 = LeakyReLU()

        self.output_layer = Conv2DTranspose(1, (5, 5), activation='sigmoid', padding='same')

    def encoder(self, inputs):
        # Encoder part
        x = self.encoder_conv1(inputs)
        x = self.encoder_bn1(x)
        x = self.encoder_leaky_relu1(x)
        x = self.encoder_max_pool1(x)

        x = self.encoder_conv2(x)
        x = self.encoder_bn2(x)
        x = self.encoder_leaky_relu2(x)
        x = self.encoder_max_pool2(x)

        x = self.flatten(x)
        latent = self.encoder_latent(x)

        return latent

    def decoder(self, latent):
        # Decoder part
        x = self.decoder_dense(latent)
        x = self.decoder_reshape(x)

        x = self.decoder_upsample1(x)
        x = self.decoder_conv2d_transpose1(x)
        x = self.decoder_bn1(x)
        x = self.decoder_leaky_relu1(x)

        x = self.decoder_upsample2(x)
        x = self.decoder_conv2d_transpose2(x)
        x = self.decoder_bn2(x)
        x = self.decoder_leaky_relu2(x)

        decoded = self.output_layer(x)

        return decoded

    def call(self, inputs):
        # Forward pass: call encoder and decoder
        latent = self.encoder(inputs)
        decoded = self.decoder(latent)
        return decoded

    def train_step(self, x):

        with tf.GradientTape() as tape:
            # Forward pass
            x_hat = self(x, training=True)
            # Compute loss
            mse_recons_loss = tf.reduce_mean(tf.square(x - x_hat))

        # Compute gradients
        gradients = tape.gradient(mse_recons_loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Return a dictionary mapping metric names to current value
        return {"mse_recons_loss": mse_recons_loss}

    def test_step(self, x):
        # Forward pass (inference mode)
        x_hat = self(x, training=False)
        # Compute loss
        mse_recons_loss = tf.reduce_mean(tf.square(x - x_hat))

        # Return a dictionary mapping metric names to current value
        return {"mse_recons_loss": mse_recons_loss}



class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim=32, beta_kl=1):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl

        # Encoder layers
        self.encoder_conv1 = Conv2D(4, (5, 5), activation=None, padding='same')
        self.encoder_bn1 = BatchNormalization()
        self.encoder_leaky_relu1 = LeakyReLU()
        self.encoder_max_pool1 = MaxPooling2D(pool_size=(2, 2))

        self.encoder_conv2 = Conv2D(8, (5, 5), activation=None, padding='same')
        self.encoder_bn2 = BatchNormalization()
        self.encoder_leaky_relu2 = LeakyReLU()
        self.encoder_max_pool2 = MaxPooling2D(pool_size=(2, 2))

        self.flatten = Flatten()
        self.encoder_mean = Dense(latent_dim, activation=None)
        self.encoder_log_var = Dense(latent_dim, activation=None)

        # Decoder layers
        self.decoder_dense = Dense(7 * 7 * 8, activation=None)
        self.decoder_reshape = Reshape((7 , 7, 8))

        self.decoder_upsample1 = UpSampling2D(size=(2, 2))
        self.decoder_conv2d_transpose1 = Conv2DTranspose(8, (5, 5), activation=None, padding='same')
        self.decoder_bn1 = BatchNormalization()
        self.decoder_leaky_relu1 = LeakyReLU()

        self.decoder_upsample2 = UpSampling2D(size=(2, 2))
        self.decoder_conv2d_transpose2 = Conv2DTranspose(4, (5, 5), activation=None, padding='same')
        self.decoder_bn2 = BatchNormalization()
        self.decoder_leaky_relu2 = LeakyReLU()

        self.output_layer = Conv2DTranspose(1, (5, 5), activation='sigmoid', padding='same')

    def encoder(self, inputs):
        # Encoder part
        x = self.encoder_conv1(inputs)
        x = self.encoder_bn1(x)
        x = self.encoder_leaky_relu1(x)
        x = self.encoder_max_pool1(x)

        x = self.encoder_conv2(x)
        x = self.encoder_bn2(x)
        x = self.encoder_leaky_relu2(x)
        x = self.encoder_max_pool2(x)

        x = self.flatten(x)
        mean = self.encoder_mean(x)
        log_var = self.encoder_log_var(x)

        return mean, log_var

    def reparameterize(self, mean, log_var):
        # Reparameterization trick: z = mean + std * eps
        eps = tf.random.normal(shape=tf.shape(mean))
        std = tf.exp(0.5 * log_var)
        z = mean + std * eps
        return z

    def decoder(self, z):
        # Decoder part
        x = self.decoder_dense(z)
        x = self.decoder_reshape(x)

        x = self.decoder_upsample1(x)
        x = self.decoder_conv2d_transpose1(x)
        x = self.decoder_bn1(x)
        x = self.decoder_leaky_relu1(x)

        x = self.decoder_upsample2(x)
        x = self.decoder_conv2d_transpose2(x)
        x = self.decoder_bn2(x)
        x = self.decoder_leaky_relu2(x)

        decoded = self.output_layer(x)
        return decoded

    def call(self, inputs):
        # Forward pass: call encoder, reparameterize, and decoder
        mean, log_var = self.encoder(inputs)
        z = self.reparameterize(mean, log_var)
        decoded = self.decoder(z)
        return decoded, mean, log_var

    def compute_kl_loss(self, mean, log_var):
        # Compute the KL divergence loss
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1)
        return kl_loss

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Forward pass
            x_hat, mean, log_var = self(x, training=True)
            # Reconstruction loss (Mean Squared Error)
            mse_recons_loss = tf.reduce_mean(tf.square(x - x_hat))
            # KL divergence loss
            kl_loss = tf.reduce_mean(self.compute_kl_loss(mean, log_var))
            # Total loss is reconstruction loss + KL divergence loss
            total_loss = mse_recons_loss + self.beta_kl * kl_loss

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Return a dictionary mapping metric names to current value
        return {"total_loss": total_loss, "mse_recons_loss": mse_recons_loss, "kl_loss": kl_loss}

    def test_step(self, x):
        # Forward pass (inference mode)
        x_hat, mean, log_var = self(x, training=False)
        # Compute losses
        mse_recons_loss = tf.reduce_mean(tf.square(x - x_hat))
        kl_loss = tf.reduce_mean(self.compute_kl_loss(mean, log_var))
        total_loss = mse_recons_loss + self.beta_kl * kl_loss

        # Return a dictionary mapping metric names to current value
        return {"total_loss": total_loss, "mse_recons_loss": mse_recons_loss, "kl_loss": kl_loss}


class SiameseAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim=32, similarity_coeff=0.1):
        super(SiameseAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.similarity_coeff = similarity_coeff

        self.encoder_conv1 = Conv2D(4, (5, 5), activation=None, padding='same')
        self.encoder_bn1 = BatchNormalization()
        self.encoder_leaky_relu1 = LeakyReLU()
        self.encoder_max_pool1 = MaxPooling2D(pool_size=(2, 2))

        self.encoder_conv2 = Conv2D(8, (5, 5), activation=None, padding='same')
        self.encoder_bn2 = BatchNormalization()
        self.encoder_leaky_relu2 = LeakyReLU()
        self.encoder_max_pool2 = MaxPooling2D(pool_size=(2, 2))

        self.flatten = Flatten()
        self.encoder_latent = Dense(latent_dim, activation=None)

        self.decoder_dense = Dense(7 * 7 * 8, activation=None)
        self.decoder_reshape = Reshape((7 , 7, 8))

        self.decoder_upsample1 = UpSampling2D(size=(2, 2))
        self.decoder_conv2d_transpose1 = Conv2DTranspose(8, (5, 5), activation=None, padding='same')
        self.decoder_bn1 = BatchNormalization()
        self.decoder_leaky_relu1 = LeakyReLU()

        self.decoder_upsample2 = UpSampling2D(size=(2, 2))
        self.decoder_conv2d_transpose2 = Conv2DTranspose(4, (5, 5), activation=None, padding='same')
        self.decoder_bn2 = BatchNormalization()
        self.decoder_leaky_relu2 = LeakyReLU()

        self.output_layer = Conv2DTranspose(1, (5, 5), activation='sigmoid', padding='same')

    def encoder(self, inputs):
        # Shared Encoder part
        x = self.encoder_conv1(inputs)
        x = self.encoder_bn1(x)
        x = self.encoder_leaky_relu1(x)
        x = self.encoder_max_pool1(x)

        x = self.encoder_conv2(x)
        x = self.encoder_bn2(x)
        x = self.encoder_leaky_relu2(x)
        x = self.encoder_max_pool2(x)

        x = self.flatten(x)
        latent = self.encoder_latent(x)

        return latent

    def decoder(self, latent):
        # Decoder for input1
        x = self.decoder_dense(latent)
        x = self.decoder_reshape(x)

        x = self.decoder_upsample1(x)
        x = self.decoder_conv2d_transpose1(x)
        x = self.decoder_bn1(x)
        x = self.decoder_leaky_relu1(x)

        x = self.decoder_upsample2(x)
        x = self.decoder_conv2d_transpose2(x)
        x = self.decoder_bn2(x)
        x = self.decoder_leaky_relu2(x)

        decoded = self.output_layer(x)
        return decoded

    def call(self, inputs):
        # Forward pass for both inputs
        input1, input2 = tf.split(inputs, 2, axis=0)
        # Encode both inputs
        latent1 = self.encoder(input1)
        latent2 = self.encoder(input2)

        # Decode each input separately
        decoded1 = self.decoder(latent1)
        decoded2 = self.decoder(latent2)

        return decoded1, decoded2, latent1, latent2

    @staticmethod
    def compute_contrastive_loss(latent1, latent2, margin=1.0):
        # Contrastive loss (L2 distance) between the latent vectors
        return tf.reduce_mean(tf.square(latent1 - latent2))  # TODO this is different from what i did for jean zad WMH : no every pair is tracked, it's randomly sampled

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            input1, input2 = tf.split(inputs, 2, axis=0)
            # Forward pass
            decoded1, decoded2, latent1, latent2 = self(inputs, training=True)

            # Reconstruction loss (Mean Squared Error)
            mse_recons_loss1 = tf.reduce_mean(tf.square(input1 - decoded1))
            mse_recons_loss2 = tf.reduce_mean(tf.square(input2 - decoded2))
            mse_recons_loss = 0.5*mse_recons_loss1 + 0.5*mse_recons_loss2

            # Contrastive loss (optional)
            contrastive_loss = self.compute_contrastive_loss(latent1, latent2)

            # Total loss is reconstruction loss + contrastive loss
            total_loss = mse_recons_loss + self.similarity_coeff * contrastive_loss

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Return a dictionary mapping metric names to current value
        return {
            "total_loss": total_loss,
            "mse_recons_loss": mse_recons_loss,
            "contrastive_loss": contrastive_loss,
        }

    def test_step(self, inputs):
        # Unpack the data
        input1, input2 = tf.split(inputs, 2, axis=0)
        # Forward pass
        decoded1, decoded2, latent1, latent2 = self(inputs, training=False)

        # Compute losses
        mse_recons_loss1 = tf.reduce_mean(tf.square(input1 - decoded1))
        mse_recons_loss2 = tf.reduce_mean(tf.square(input2 - decoded2))
        mse_recons_loss = 0.5 * mse_recons_loss1 + 0.5 * mse_recons_loss2

        contrastive_loss = self.compute_contrastive_loss(latent1, latent2)
        total_loss = mse_recons_loss + self.similarity_coeff * contrastive_loss

        # Return a dictionary mapping metric names to current value
        return {
            "total_loss": total_loss,
            "mse_recons_loss": mse_recons_loss,
            "contrastive_loss": contrastive_loss,
        }

