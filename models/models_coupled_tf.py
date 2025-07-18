import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, MaxPooling2D, UpSampling2D, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, ReLU
import cvxpy as cp
from cvxpylayers.tensorflow import CvxpyLayer



class OCSVMguidedAutoencoder(tf.keras.Model):

    def __init__(self, batch_size_train, batch_size_valid, latent_dim=32, ocsvm_coeff=0.1, nu_ocsvm_coeff=0.03,
                 gamma_rbf_coeff="scale", jz_mode="StopGradLoss", jean_zad_linear=False):
        super(OCSVMguidedAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Shared Encoder layers
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

        # OC-SVM :
        self.dtype_ = tf.keras.mixed_precision.global_policy().name.replace("mixed_", "")
        self.ocsvm_coeff = tf.constant(ocsvm_coeff, dtype=self.dtype_)
        # OCSVM hyperparameters coeffs :
        self.nu = tf.constant(nu_ocsvm_coeff, dtype=self.dtype_)
        if gamma_rbf_coeff == "auto":
            gamma_rbf_coeff = tf.constant(1/(latent_dim * 2 * 2), dtype=self.dtype_)   # This corresponds to the "auto" param of the OneClassSVM (sklearn)
            self.gamma_rbf_coeff = tf.constant(gamma_rbf_coeff, dtype=self.dtype_)
        elif gamma_rbf_coeff == "scale":  # default param in sklearn
            self.gamma_rbf_coeff = "scale"
        elif isinstance(gamma_rbf_coeff, int) or isinstance(gamma_rbf_coeff, float):  # user specified gamma_rbf
            self.gamma_rbf_coeff = tf.constant(gamma_rbf_coeff, dtype=self.dtype_)
        elif isinstance(gamma_rbf_coeff, str):
            raise Exception(gamma_rbf_coeff + " not implemented or non-valid")
        self.jz_mode = jz_mode
        self.linear = jean_zad_linear

        # CVXPyLayers OC-SVM dual problem :
        n_train, n_valid = batch_size_train, batch_size_valid
        alpha_sv_train = cp.Variable(n_train//2)  # no need to set non-negative, it is enforced in the constraints
        k_z_sqrt_train = cp.Parameter((n_train//2, n_train//2), PSD=True)  # SQRT of kernel matrix k_z of z, i.e. k_z_sqrt @ k_z_sqrt = k_z, sqrt of k_z is also PSD
        # /!\ every n is //2 to separate the z into two : z_sv used to compute support vector (the cvx optim problem) and z_loss used to enforce the ocsvm objective
        alpha_sv_valid = cp.Variable(n_valid//2)
        k_z_sqrt_valid = cp.Parameter((n_valid//2, n_valid//2), PSD=True)

        constraints_train = [cp.sum(alpha_sv_train) == (nu_ocsvm_coeff*(n_train//2))]  # sum of alpha_i = nu * n
        constraints_train += [alpha_sv_train[i] >= 0 for i in range(n_train//2)]   # 0 <= alpha_i <= 1 (left part)
        constraints_train += [alpha_sv_train[i] <= 1 for i in range(n_train//2)]  # 0 <= alpha_i <= 1 (right part)

        constraints_valid = [cp.sum(alpha_sv_valid) == (nu_ocsvm_coeff*(n_valid//2))]  # sum of alpha_i = nu * n
        # We actually solve a scaled problem (http://ntur.lib.ntu.edu.tw/bitstream/246246/155217/1/09.pdf), with alpha_scaled = nu * n * alpha
        constraints_valid += [alpha_sv_valid[i] >= 0 for i in range(n_valid//2)]   # 0 <= alpha_i <= 1 (left part)
        constraints_valid += [alpha_sv_valid[i] <= 1 for i in range(n_valid//2)]  # 0 <= alpha_i <= 1 (right part)

        # sum_i_j (Ks @ alpha)_i_j == ||Ks @ alpha||_2^2 == alpha.T @ Ks.T @ Ks @ alpha == alpha.T @ K @ alpha
        self.ocsvm_layer_train = CvxpyLayer(cp.Problem(cp.Minimize(0.5 * cp.sum_squares(k_z_sqrt_train @ alpha_sv_train)), constraints_train),
                                            parameters=[k_z_sqrt_train], variables=[alpha_sv_train])
        self.ocsvm_layer_valid = CvxpyLayer(cp.Problem(cp.Minimize(0.5 * cp.sum_squares(k_z_sqrt_valid @ alpha_sv_valid)), constraints_valid),
                                            parameters=[k_z_sqrt_valid], variables=[alpha_sv_valid])

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

    def solve_ocsvm_problem(self, latent, training):
        # Identification of n and d
        n_subjects = tf.constant(tf.cast(tf.shape(latent)[0], self.dtype_))
        z_dim = tf.constant(tf.cast(tf.shape(latent)[1], self.dtype_))
        # Split the batch in two : one for solving ocsvm, one for loss computation
        z_sv, _ = tf.split(latent, num_or_size_splits=2, axis=0)  # z_loss not used here !
        # Standardisation:
        z_sv = (z_sv - tf.reduce_mean(z_sv, axis=0))/tf.math.reduce_std(z_sv, axis=0)
        # Gamma computation if needed
        if str(self.gamma_rbf_coeff) == "scale":
            gamma_rbf_coeff = 1 / (z_dim * tf.stop_gradient(tf.math.reduce_variance(z_sv)))
            gamma_rbf_coeff = tf.constant(1e32, self.dtype_) if tf.math.is_inf(gamma_rbf_coeff) else gamma_rbf_coeff  # in case of variance 0 (collapse) just to avoid nan
        else:
            gamma_rbf_coeff = self.gamma_rbf_coeff
        # Computation of the K (kernel) matrix and its square root, parameters of the optim problem
        if self.linear:
            k_z_sv = tf.tensordot(z_sv, tf.transpose(z_sv), axes=1)
        else:
            l2_dist_z_sv_i_j = tf.reduce_sum((z_sv[:, None, ...] - z_sv[None, ...]) ** 2, axis=-1)  # sum is over z dimension (norm l2 dim z)
            k_z_sv = tf.exp(-gamma_rbf_coeff * l2_dist_z_sv_i_j)  # Kernel matrix K of z_sv with RBF kernel, i.e. K_i_j = <z_i,z_j> = exp( - gamma (z_i - z_j)**2)
        num_stability_coeff = tf.constant(1e-8 / gamma_rbf_coeff, self.dtype_) if not self.linear else tf.constant(1e-8)
        k_z_sqrt_sv = tf.linalg.sqrtm(tf.cast(k_z_sv + num_stability_coeff * tf.eye(n_subjects // 2), tf.float64))
        # Computing support vectors from OC-SVM problem
        if training:
            alpha_sv, = self.ocsvm_layer_train(k_z_sqrt_sv)
        else:  # validation:
            alpha_sv, = self.ocsvm_layer_valid(k_z_sqrt_sv)
        alpha_sv = tf.cast(alpha_sv, self.dtype_)  # CVXPylayer outputs float64
        alpha_sv = alpha_sv / (self.nu * n_subjects / 2)  # important to return to the unscaled problem

        return alpha_sv, k_z_sv

    def compute_ocsvm_objective(self, alpha_sv, latent, k_z_sv):
        # Identification of n and d
        n_subjects = tf.constant(tf.cast(tf.shape(latent)[0], self.dtype_))
        z_dim = tf.constant(tf.cast(tf.shape(latent)[1], self.dtype_))
        # Split the batch in two : one for solving ocsvm, one for loss computation
        z_sv, z_loss = tf.split(latent, num_or_size_splits=2, axis=0)
        # Standardisation:
        z_sv, z_loss = (z_sv - tf.reduce_mean(z_sv, axis=0))/tf.math.reduce_std(z_sv, axis=0), (z_loss - tf.reduce_mean(z_loss, axis=0))/tf.math.reduce_std(z_loss, axis=0)
        # Gamma computation if needed
        if str(self.gamma_rbf_coeff) == "scale":
            gamma_rbf_coeff = 1 / (z_dim * tf.stop_gradient(tf.math.reduce_variance(z_sv)))
            gamma_rbf_coeff = tf.constant(1e32, self.dtype_) if tf.math.is_inf(gamma_rbf_coeff) else gamma_rbf_coeff  # in case of variance 0 (collapse) just to avoid nan
        else:
            gamma_rbf_coeff = self.gamma_rbf_coeff
        # Computation of the K (kernel) matrix and its square root, parameters of the optim problem
        if self.linear:
            k_z_sv_loss = tf.tensordot(z_sv, tf.transpose(z_loss), axes=1)
        else:
            if "FullGrad" in self.jz_mode:
                l2_dist_z_sv_loss_i_j = tf.reduce_sum((z_sv[:, None, ...] - z_loss[None, ...]) ** 2, axis=-1)  # sum is over z dimension (norm l2 dim z)
            elif "StopGradSV" in self.jz_mode:
                l2_dist_z_sv_loss_i_j = tf.reduce_sum((tf.stop_gradient(z_sv[:, None, ...]) - z_loss[None, ...]) ** 2, axis=-1)  # sum is over z dimension (norm l2 dim z)
            elif "StopGradLoss" in self.jz_mode:
                l2_dist_z_sv_loss_i_j = tf.reduce_sum((z_sv[:, None, ...] - tf.stop_gradient(z_loss[None, ...])) ** 2, axis=-1)  # sum is over z dimension (norm l2 dim z)
            k_z_sv_loss = tf.exp(
                -gamma_rbf_coeff * l2_dist_z_sv_loss_i_j)  # "Kernel" matrix K_sv_loss of distance z_sv to z_loss with RBF kernel, i.e. K_i_j = <z_sv_i,z_loss_j> = exp( - gamma (z_sv_i - z_loss_j)**2)
        # sv_arb_idx = tf.math.argmin((alpha_sv - 1/(self.nu*n_subjects))**2)  # we search for an 0 < alpha_i < 1/(nu x n/2), arbitrarily we will take the closest to 1/(2 x nu x n/2) = 1/(nu x n)
        # e_j_arb_idx = tf.concat((tf.zeros(sv_arb_idx), tf.ones(1), tf.zeros(alpha_sv.shape - sv_arb_idx - 1)), axis=0)
        # rho = alpha_sv[None,] @ k_z_sv @ e_j_arb_idx[..., None]
        sv_all = (alpha_sv - 1 / (self.nu * n_subjects)) ** 2 < (1 / (self.nu * n_subjects) - 1e-6) ** 2  # the middle is 1/nu*n, low bound 0, high bound 2/nu*n, small tolerance eps 1e-6
        e_j_all = tf.cast(sv_all, self.dtype_)  # sum of e_j_all will be the number of SV, to obtain the mean
        rho_mean = 1 / (tf.reduce_sum(e_j_all)) * alpha_sv[None,] @ k_z_sv @ e_j_all[..., None]  # need to insert dimensions for correct multiplication, equivalent to a.T @ K @ e_j
        rho = rho_mean
        if "StopGradSV" in self.jz_mode or "*" in self.jz_mode:
            decision_functions = (tf.stop_gradient(alpha_sv[None,]) @ k_z_sv_loss - tf.stop_gradient(rho)) * self.nu * (
                        n_subjects / 2)  # Another de-normalization is necessary because of the scaled problem
        else:
            decision_functions = (alpha_sv[None,] @ k_z_sv_loss - rho) * self.nu * (n_subjects / 2)  # Another de-normalization is necessary because of the scaled problem
        # minus sign because deci_func is neg for outliers, relu to penalize only outliers (applied on z_loss !),  (nu as the upper bound of outliers seems the natural normalizing coefficient)
        ocsvm_objective = (1 / self.nu) * tf.nn.relu(-decision_functions) @ (tf.ones(alpha_sv[..., None].shape))  # sum to n so need but sparse so no need to divide by n

        return tf.squeeze(ocsvm_objective)

    def call(self, inputs, training=False, inference=True):
        # Forward pass
        latent = self.encoder(inputs)
        decoded = self.decoder(latent)
        if inference:
            return decoded, latent
        # Solve OC-SVM problem
        alpha_sv, k_z_sv = self.solve_ocsvm_problem(latent, training=training)
        return decoded, latent, alpha_sv, k_z_sv

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            # Forward pass
            decoded, latent, alpha_sv, k_z_sv = self(inputs, training=True, inference=False)

            # Reconstruction loss (Mean Squared Error)
            mse_recons_loss = tf.reduce_mean(tf.square(inputs - decoded))

            # OC-SVM objective
            ocsvm_objective = self.compute_ocsvm_objective(alpha_sv, latent, k_z_sv)

            # Total loss is reconstruction loss + contrastive loss
            total_loss = mse_recons_loss + self.ocsvm_coeff * ocsvm_objective

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # STD for monitoring :
        pairwise_distances = tf.norm(tf.expand_dims(latent, 1) - tf.expand_dims(latent, 0), axis=-1)
        mean_pairwise_distance = tf.reduce_mean(pairwise_distances)
        std_z = tf.math.reduce_std(latent)

        # Return a dictionary mapping metric names to current value
        return {
            "total_loss": total_loss,
            "mse_recons_loss": mse_recons_loss,
            "ocsvm_objective": ocsvm_objective,
            "mean_pairwise_distance": mean_pairwise_distance,
            "std_z": std_z
        }

    def test_step(self, inputs):
        # Forward pass
        decoded, latent, alpha_sv, k_z_sv = self(inputs, training=False, inference=False)

        # Compute losses
        mse_recons_loss = tf.reduce_mean(tf.square(inputs - decoded))

        # OC-SVM objective
        ocsvm_objective = self.compute_ocsvm_objective(alpha_sv, latent, k_z_sv)
        total_loss = mse_recons_loss + self.ocsvm_coeff * ocsvm_objective

        # STD for monitoring :
        pairwise_distances = tf.norm(tf.expand_dims(latent, 1) - tf.expand_dims(latent, 0), axis=-1)
        mean_pairwise_distance = tf.reduce_mean(pairwise_distances)
        std_z = tf.math.reduce_std(latent)

        # Return a dictionary mapping metric names to current value
        return {
            "total_loss": total_loss,
            "mse_recons_loss": mse_recons_loss,
            "ocsvm_objective": ocsvm_objective,
            "mean_pairwise_distance": mean_pairwise_distance,
            "std_z": std_z
        }


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
        # sigmoid is probibited in deep SVDD but in the encoder only
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
            mse_z_center = tf.reduce_mean(l2_norm_squared_z_center)  # For those in the back not following : MSE is the mean of the squared L2 norm (squared error)
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
        mse_z_center = tf.reduce_mean(l2_norm_squared_z_center)  # For those in the back not following : MSE is the mean of the squared L2 norm (squared error)
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
        # sigmoid is probibited in deep SVDD but in the encoder only
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
            mse_z_center = tf.reduce_mean(l2_norm_squared_z_center)  # For those in the back not following : MSE is the mean of the squared L2 norm (squared error)

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

