import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, MaxPooling2D, UpSampling2D, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, ReLU

from ocsvm_guidance_tf import OCSVMGuidedAutoencoderBase


class OCSVMguidedAutoencoder(OCSVMGuidedAutoencoderBase):
    """OgAE for Experiment 1 (MNIST-C). OCSVM-guidance arguments are documented in OCSVMGuidedAutoencoderBase.

    Defaults are the best Experiment 1 hyperparameters (lambda = 1e-2, nu = 0.03, gamma = 1e-2), with the decision
    function given by the mean of the last M = 10 OC-SVMs. Use BetaSchedule for the expander -> expander + compactor
    schedule. Must be compiled with run_eagerly=True.
    """

    def __init__(self, batch_size_train, batch_size_valid, latent_dim=32, n_last_ocsvms=10, **ogae_kwargs):
        super().__init__(batch_size_train, batch_size_valid, n_last_ocsvms=n_last_ocsvms, **ogae_kwargs)
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
        self.decoder_reshape = Reshape((7, 7, 8))

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
        x = self.encoder_conv1(inputs)
        x = self.encoder_bn1(x)
        x = self.encoder_leaky_relu1(x)
        x = self.encoder_max_pool1(x)

        x = self.encoder_conv2(x)
        x = self.encoder_bn2(x)
        x = self.encoder_leaky_relu2(x)
        x = self.encoder_max_pool2(x)

        x = self.flatten(x)
        return self.encoder_latent(x)

    def decoder(self, latent):
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

        return self.output_layer(x)


class DeepSVDDAutoEncoderHard(tf.keras.Model):
    # Deep SVDD : On MNIST, we use a CNN with two
    # modules, 8 × (5 × 5 × 1)-filters followed by 4 × (5 × 5 × 1)-
    # filters, and a final dense layer of 32 uni

    # This implementation does not have weight decay (but the other methods do not have also)

    def __init__(self, center, latent_dim=32, batch_norm=True, balance_coeff=0.1):
        super(DeepSVDDAutoEncoderHard, self).__init__()
        self.latent_dim = latent_dim
        self.center = center
        self.balance_coeff = balance_coeff  # will be applied to MSE

        # Encoder layers
        self.encoder_conv1 = Conv2D(4, (5, 5), activation=None, padding='same', use_bias=False)  # DeepSVDD does not use bias
        self.encoder_bn1 = BatchNormalization() if batch_norm else lambda x: x
        self.encoder_leaky_relu1 = LeakyReLU()
        self.encoder_max_pool1 = MaxPooling2D(pool_size=(2, 2))

        self.encoder_conv2 = Conv2D(8, (5, 5), activation=None, padding='same', use_bias=False)  # DeepSVDD does not use bias
        self.encoder_bn2 = BatchNormalization() if batch_norm else lambda x: x
        self.encoder_leaky_relu2 = LeakyReLU()
        self.encoder_max_pool2 = MaxPooling2D(pool_size=(2, 2))

        self.flatten = Flatten()
        self.encoder_latent = Dense(latent_dim, activation=None, use_bias=False)  # DeepSVDD does not use bias

        # Decoder layers
        self.decoder_dense = Dense(7 * 7 * 8, activation=None)
        self.decoder_reshape = Reshape((7 , 7, 8))

        self.decoder_upsample1 = UpSampling2D(size=(2, 2))
        self.decoder_conv2d_transpose1 = Conv2DTranspose(8, (5, 5), activation=None, padding='same', use_bias=False)  # DeepSVDD does not use bias
        # Bias could be used in the decoder (no collapse, the decoder does not lead to the latent space) but we perform the symmetry
        # of the encoder (no info in papers implementing this method)
        self.decoder_bn1 = BatchNormalization()
        self.decoder_leaky_relu1 = LeakyReLU()

        self.decoder_upsample2 = UpSampling2D(size=(2, 2))
        self.decoder_conv2d_transpose2 = Conv2DTranspose(4, (5, 5), activation=None, padding='same', use_bias=False)  # DeepSVDD does not use bias
        self.decoder_bn2 = BatchNormalization()
        self.decoder_leaky_relu2 = LeakyReLU()
        # sigmoid is prohibited in Deep SVDD, but only in the encoder
        self.output_layer = Conv2DTranspose(1, (5, 5), activation='sigmoid', padding='same', use_bias=False)  # DeepSVDD does not use bias

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

        l2_norm_squared_z_center = tf.reduce_sum((latent - self.center)**2, axis=-1)

        return l2_norm_squared_z_center, decoded

    def compute_ano_score(self, inputs):

        l2_norm_squared_z_center, decoded = self(inputs)
        return - l2_norm_squared_z_center

    def train_step(self, x):

        with tf.GradientTape() as tape:
            # Forward pass
            l2_norm_squared_z_center, x_hat= self(x, training=True)
            # MSE loss :
            mse_recons_loss = tf.reduce_mean(tf.square(x - x_hat))
            # Hard Deep SVDD, MSE between center and data points :
            mse_z_center = tf.reduce_mean(l2_norm_squared_z_center)  # MSE is the mean of the squared L2 norm
            #total loss :
            total_loss = self.balance_coeff * mse_recons_loss +  mse_z_center

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Return a dictionary mapping metric names to current value
        return {"total_loss": total_loss, "mse_recons_loss": mse_recons_loss, "mse_z_center": mse_z_center}

    def test_step(self, x):
        # Forward pass (inference mode)
        l2_norm_squared_z_center, x_hat = self(x, training=False)
        # MSE loss :
        mse_recons_loss = tf.reduce_mean(tf.square(x - x_hat))
        # Hard Deep SVDD, MSE between center and data points :
        mse_z_center = tf.reduce_mean(l2_norm_squared_z_center)  # MSE is the mean of the squared L2 norm
        # total loss :
        total_loss = self.balance_coeff * mse_recons_loss + mse_z_center

        # Return a dictionary mapping metric names to current value
        return {"total_loss": total_loss, "mse_recons_loss": mse_recons_loss, "mse_z_center": mse_z_center}


class DeepSVDDVariationalAutoEncoderHard(tf.keras.Model):
    def __init__(self, latent_dim=32, batch_norm=True, balance_coeff=0.1, beta_kl=1):
        super(DeepSVDDVariationalAutoEncoderHard, self).__init__()
        self.latent_dim = latent_dim
        self.center = tf.Variable(tf.zeros(latent_dim), trainable=False, dtype=tf.float32)  # the 0 center will not be used and will be updated as soon as the first batch comes
        self.balance_coeff = balance_coeff  # for MSE
        self.beta_kl = beta_kl  # for KL divergence

        # Encoder layers
        self.encoder_conv1 = Conv2D(4, (5, 5), activation=None, padding='same', use_bias=False)
        self.encoder_bn1 = BatchNormalization() if batch_norm else lambda x: x
        self.encoder_leaky_relu1 = LeakyReLU()
        self.encoder_max_pool1 = MaxPooling2D(pool_size=(2, 2))

        self.encoder_conv2 = Conv2D(8, (5, 5), activation=None, padding='same', use_bias=False)
        self.encoder_bn2 = BatchNormalization() if batch_norm else lambda x: x
        self.encoder_leaky_relu2 = LeakyReLU()
        self.encoder_max_pool2 = MaxPooling2D(pool_size=(2, 2))

        self.flatten = Flatten()
        self.encoder_mean = Dense(latent_dim, activation=None, use_bias=False)
        self.encoder_log_var = Dense(latent_dim, activation=None, use_bias=False)

        # Decoder layers
        self.decoder_dense = Dense(7 * 7 * 8, activation=None)
        self.decoder_reshape = Reshape((7, 7, 8))

        self.decoder_upsample1 = UpSampling2D(size=(2, 2))
        self.decoder_conv2d_transpose1 = Conv2DTranspose(8, (5, 5), activation=None, padding='same', use_bias=False)
        self.decoder_bn1 = BatchNormalization()
        self.decoder_leaky_relu1 = LeakyReLU()

        self.decoder_upsample2 = UpSampling2D(size=(2, 2))
        self.decoder_conv2d_transpose2 = Conv2DTranspose(4, (5, 5), activation=None, padding='same', use_bias=False)
        self.decoder_bn2 = BatchNormalization()
        self.decoder_leaky_relu2 = LeakyReLU()
        # sigmoid is prohibited in Deep SVDD, but only in the encoder
        self.output_layer = Conv2DTranspose(1, (5, 5), activation='sigmoid', padding='same', use_bias=False)

    def encoder(self, inputs):
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
        eps = tf.random.normal(shape=tf.shape(mean))
        std = tf.exp(0.5 * log_var)
        z = mean + std * eps
        return z

    def decoder(self, z):
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


    def compute_ano_score(self, inputs):
        latent, _ = self.encoder(inputs)
        l2_norm_squared_z_center = tf.reduce_sum((latent-self.center)**2, axis=-1)

        return - l2_norm_squared_z_center

    def call(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        return self.decoder(z)

    def train_step(self, x):
        with tf.GradientTape() as tape:
            mean, log_var = self.encoder(x)
            z = self.reparameterize(mean, log_var)
            # Update center c to be the mean of the current batch
            self.center.assign(tf.reduce_mean(z, axis=0))
            x_hat = self.decoder(z)
            mse_recons_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(x, x_hat))
            kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
            mse_z_center = tf.reduce_mean(tf.keras.losses.mean_squared_error(z, self.center))

            total_loss = self.balance_coeff * (mse_recons_loss + self.beta_kl * kl_loss) + mse_z_center

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": total_loss, "mse_reconstruction_loss": mse_recons_loss, "kl_loss": kl_loss, "mse_z_center": mse_z_center}

    def test_step(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decoder(z)
        mse_recons_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(x, x_hat))
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        mse_z_center = tf.reduce_mean(tf.keras.losses.mean_squared_error(z, self.center))
        total_loss = self.balance_coeff * (mse_recons_loss + self.beta_kl * kl_loss) + mse_z_center
        return {"loss": total_loss, "mse_reconstruction_loss": mse_recons_loss, "kl_loss": kl_loss, "mse_z_center": mse_z_center}


class DeepSVDDEncoderHard(tf.keras.Model):
    # Deep SVDD : On MNIST, we use a CNN with two
    # modules, 8 × (5 × 5 × 1)-filters followed by 4 × (5 × 5 × 1)-
    # filters, and a final dense layer of 32 uni

    # This implementation does not have weight decay (but the other methods do not have also)

    def __init__(self, center, latent_dim=32, batch_norm=True):
        super(DeepSVDDEncoderHard, self).__init__()
        self.latent_dim = latent_dim
        self.center = center

        # Encoder layers
        self.encoder_conv1 = Conv2D(4, (5, 5), activation=None, padding='same', use_bias=False)  # DeepSVDD does not use bias
        self.encoder_bn1 = BatchNormalization() if batch_norm else lambda x: x
        self.encoder_leaky_relu1 = LeakyReLU()
        self.encoder_max_pool1 = MaxPooling2D(pool_size=(2, 2))

        self.encoder_conv2 = Conv2D(8, (5, 5), activation=None, padding='same', use_bias=False)  # DeepSVDD does not use bias
        self.encoder_bn2 = BatchNormalization() if batch_norm else lambda x: x
        self.encoder_leaky_relu2 = LeakyReLU()
        self.encoder_max_pool2 = MaxPooling2D(pool_size=(2, 2))

        self.flatten = Flatten()
        self.encoder_latent = Dense(latent_dim, activation=None, use_bias=False)  # DeepSVDD does not use bias

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

    def call(self, inputs):
        # Forward pass: call encoder and decoder
        latent = self.encoder(inputs)

        l2_norm_squared_z_center = tf.reduce_sum((latent - self.center)**2, axis=-1)

        return l2_norm_squared_z_center

    def compute_ano_score(self, inputs):

        return - self(inputs)

    def train_step(self, x):

        with tf.GradientTape() as tape:
            # Forward pass
            l2_norm_squared_z_center = self(x, training=True)
            # 1st and only term of Hard Deep SVDD, MSE between center and data points :
            mse_z_center = tf.reduce_mean(l2_norm_squared_z_center)  # MSE is the mean of the squared L2 norm

        # Compute gradients
        gradients = tape.gradient(mse_z_center, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Return a dictionary mapping metric names to current value
        return {"mse_z_center": mse_z_center}

    def test_step(self, x):
        # Forward pass (inference mode)
        l2_norm_squared_z_center = self(x, training=False)
        # Compute loss
        mse_z_center = tf.reduce_mean(l2_norm_squared_z_center)

        # Return a dictionary mapping metric names to current value
        return {"mse_z_center": mse_z_center}


class DeepSVDDEncoderSoft(tf.keras.Model):
    # Deep SVDD : On MNIST, we use a CNN with two
    # modules, 8 × (5 × 5 × 1)-filters followed by 4 × (5 × 5 × 1)-
    # filters, and a final dense layer of 32 uni

    # This implementation does not have weight decay (but the other methods do not have also)

    def __init__(self, center, nu, latent_dim=32, batch_norm=True):
        super(DeepSVDDEncoderSoft, self).__init__()
        self.latent_dim = latent_dim
        self.center = center
        self.nu = nu
        self.R = tf.Variable(1e-4, trainable=True, dtype=tf.float32)
        self.train_r_only = False

        # Encoder layers
        self.encoder_conv1 = Conv2D(4, (5, 5), activation=None, padding='same', use_bias=False)  # DeepSVDD does not use bias
        self.encoder_bn1 = BatchNormalization() if batch_norm else lambda x: x
        self.encoder_leaky_relu1 = LeakyReLU()
        self.encoder_max_pool1 = MaxPooling2D(pool_size=(2, 2))

        self.encoder_conv2 = Conv2D(8, (5, 5), activation=None, padding='same', use_bias=False)  # DeepSVDD does not use bias
        self.encoder_bn2 = BatchNormalization() if batch_norm else lambda x: x
        self.encoder_leaky_relu2 = LeakyReLU()
        self.encoder_max_pool2 = MaxPooling2D(pool_size=(2, 2))

        self.flatten = Flatten()
        self.encoder_latent = Dense(latent_dim, activation=None, use_bias=False)  # DeepSVDD does not use bias

        self.relu = ReLU()

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

    def call(self, inputs):
        # Forward pass: call encoder and decoder
        latent = self.encoder(inputs)

        l2_norm_squared_z_center = tf.reduce_sum((latent - self.center)**2, axis=-1)

        return l2_norm_squared_z_center

    def compute_ano_score(self, inputs):

        return self(inputs)

    def train_step(self, x):

        trainable_vars_except_R = [var for var in self.trainable_variables if var is not self.R]

        with tf.GradientTape() as tape:
            # Forward pass
            l2_norm_squared_z_center = self(x, training=True)
            # Soft Deep SVDD is radius squared and then distance to the radius if not inside radius
            second_term = (1/self.nu) * tf.reduce_mean(self.relu(l2_norm_squared_z_center - self.R**2))
            objective = self.R**2 + second_term

        if self.train_r_only:
            # Compute gradients
            gradients = tape.gradient(objective, [self.R])
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, [self.R]))
        else:
            # Compute gradients
            gradients = tape.gradient(objective, trainable_vars_except_R)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars_except_R))

        # Return a dictionary mapping metric names to current value
        return {"objective": objective.numpy(), "second_term": second_term.numpy(), "R": self.R.numpy()}

    def test_step(self, x):
        # Forward pass
        l2_norm_squared_z_center = self(x, training=False)
        # Soft Deep SVDD is radius squared and then distance to the radius if not inside radius
        second_term = (1 / self.nu) * tf.reduce_mean(self.relu(l2_norm_squared_z_center - self.R ** 2))
        objective = self.R ** 2 + second_term

        # Return a dictionary mapping metric names to current value
        return {"objective": objective, "second_term": second_term, "R": self.R}

