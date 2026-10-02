"""OCSVM-guidance of autoencoder training (OgAE), TensorFlow implementation.

Notation follows the paper (Sec. III-B, eq. 7): each batch z is split into z_sv, used to solve the OC-SVM dual
problem (alpha*, rho*), and z_L (here `z_loss`), on which misclassified samples are penalized. The penalty is split into
  - the expander term (weight beta1): gradient flows through z_sv (and alpha*, rho*), z_loss is stopped,
  - the compactor term (weight beta2): gradient flows through z_loss only, z_sv, alpha* and rho* are stopped.
Both terms have the same forward value, so the loss value does not depend on (beta1, beta2): only the gradient does
(beta1 * g_expander + beta2 * g_compactor).

The OC-SVM layer (cvxpylayers) runs outside the TF graph: models must be compiled with `run_eagerly=True`.
"""
import collections

import cvxpy as cp
import tensorflow as tf
from cvxpylayers.tensorflow import CvxpyLayer


def build_ocsvm_layer(n, nu_ocsvm_coeff):
    """CVXPyLayers OC-SVM dual problem for n samples"""
    alpha_sv = cp.Variable(n)  # no need to set non-negative, it is enforced in the constraints
    k_z_sqrt = cp.Parameter((n, n), PSD=True)  # SQRT of kernel matrix k_z of z, i.e. k_z_sqrt @ k_z_sqrt = k_z, sqrt of k_z is also PSD
    constraints = [cp.sum(alpha_sv) == (nu_ocsvm_coeff * n)]  # sum of alpha_i = nu * n
    # We actually solve a scaled problem (http://ntur.lib.ntu.edu.tw/bitstream/246246/155217/1/09.pdf), with alpha_scaled = nu * n * alpha
    constraints += [alpha_sv >= 0, alpha_sv <= 1]  # 0 <= alpha_i <= 1
    # sum_i_j (Ks @ alpha)_i_j == ||Ks @ alpha||_2^2 == alpha.T @ Ks.T @ Ks @ alpha == alpha.T @ K @ alpha
    return CvxpyLayer(cp.Problem(cp.Minimize(0.5 * cp.sum_squares(k_z_sqrt @ alpha_sv)), constraints),
                      parameters=[k_z_sqrt], variables=[alpha_sv])


class OCSVMGuidedAutoencoderBase(tf.keras.Model):
    """Autoencoder trained with reconstruction + lambda * OCSVM-guidance loss. Subclasses define encoder() and decoder().

    Args:
        batch_size_train, batch_size_valid: batch sizes, split in two (z_sv and z_loss).
        ocsvm_coeff: lambda, weight of the OCSVM-guidance term.
        nu_ocsvm_coeff: nu of the OC-SVM.
        gamma_rbf_coeff: RBF gamma, a number, "scale" or "auto" (as in sklearn).
        beta1, beta2: weights of the expander and compactor gradients (see module docstring), can be changed during
            training with set_betas() or BetaSchedule.
        differentiate_dual: if False, the expander gradient does not flow through alpha and rho.
        standardize_z: standardize z_sv and z_loss (each with its own batch statistics) before the kernel.
        linear_kernel: linear kernel instead of the RBF kernel.
        n_last_ocsvms: number M of OC-SVMs of the last training iterations kept for decision_function() (0: none,
            a final OC-SVM must then be trained on the latent representations).
    """

    def __init__(self, batch_size_train, batch_size_valid, ocsvm_coeff=1e-2, nu_ocsvm_coeff=0.03, gamma_rbf_coeff=1e-2,
                 beta1=1.0, beta2=0.0, differentiate_dual=True, standardize_z=True, linear_kernel=False,
                 n_last_ocsvms=0, **kwargs):
        super().__init__(**kwargs)
        # OC-SVM :
        self.dtype_ = tf.keras.mixed_precision.global_policy().name.replace("mixed_", "")
        self.ocsvm_coeff = tf.constant(ocsvm_coeff, dtype=self.dtype_)
        # OCSVM hyperparameters coeffs :
        self.nu = tf.constant(nu_ocsvm_coeff, dtype=self.dtype_)
        if gamma_rbf_coeff in ("auto", "scale"):  # "scale" is the default param in sklearn, both depend on z and are computed at each batch
            self.gamma_rbf_coeff = gamma_rbf_coeff
        elif isinstance(gamma_rbf_coeff, (int, float)):  # user specified gamma_rbf
            self.gamma_rbf_coeff = tf.constant(gamma_rbf_coeff, dtype=self.dtype_)
        else:
            raise ValueError(str(gamma_rbf_coeff) + " not implemented or non-valid")
        # Expander (beta1) and compactor (beta2) weights, variables so they can be changed during training
        self.beta1 = tf.Variable(beta1, trainable=False, dtype=self.dtype_, name="beta1")
        self.beta2 = tf.Variable(beta2, trainable=False, dtype=self.dtype_, name="beta2")
        self.differentiate_dual = differentiate_dual
        self.standardize_z = standardize_z
        self.linear = linear_kernel
        # OC-SVMs (alpha_sv, rho, z_sv, gamma) of the last M training iterations, for the decision function
        self.n_last_ocsvms = n_last_ocsvms
        self.last_ocsvms = collections.deque(maxlen=n_last_ocsvms)

        # /!\ every n is //2 to separate the z into two : z_sv used to compute support vector (the cvx optim problem) and z_loss used to enforce the ocsvm objective
        self.ocsvm_layer_train = build_ocsvm_layer(batch_size_train // 2, nu_ocsvm_coeff)
        self.ocsvm_layer_valid = build_ocsvm_layer(batch_size_valid // 2, nu_ocsvm_coeff)

    def set_betas(self, beta1, beta2):
        self.beta1.assign(beta1)
        self.beta2.assign(beta2)

    def standardize(self, z, z_ref):
        # Standardisation of z with the statistics of z_ref (1e-6 in case of a collapsed dimension):
        return (z - tf.reduce_mean(z_ref, axis=0)) / (tf.math.reduce_std(z_ref, axis=0) + 1e-6)

    def compute_gamma(self, z_sv):
        # Gamma computation if needed
        z_dim = tf.cast(tf.shape(z_sv)[1], self.dtype_)
        if str(self.gamma_rbf_coeff) == "scale":
            gamma_rbf_coeff = 1 / (z_dim * tf.stop_gradient(tf.math.reduce_variance(z_sv)))
            return tf.constant(1e32, self.dtype_) if tf.math.is_inf(gamma_rbf_coeff) else gamma_rbf_coeff  # in case of variance 0 (collapse) just to avoid nan
        if str(self.gamma_rbf_coeff) == "auto":
            return 1 / z_dim  # This corresponds to the "auto" param of the OneClassSVM (sklearn)
        return self.gamma_rbf_coeff

    def kernel(self, z_a, z_b, gamma_rbf_coeff):
        # Computation of the K (kernel) matrix between z_a and z_b
        if self.linear:
            return tf.tensordot(z_a, tf.transpose(z_b), axes=1)
        l2_dist_z_i_j = tf.reduce_sum((z_a[:, None, ...] - z_b[None, ...]) ** 2, axis=-1)  # sum is over z dimension (norm l2 dim z)
        return tf.exp(-gamma_rbf_coeff * l2_dist_z_i_j)  # Kernel matrix K with RBF kernel, i.e. K_i_j = <z_a_i,z_b_j> = exp( - gamma (z_a_i - z_b_j)**2)

    def solve_ocsvm_problem(self, latent, training):
        # Identification of n
        n_subjects = tf.cast(tf.shape(latent)[0], self.dtype_)
        # Split the batch in two : one for solving ocsvm, one for loss computation
        z_sv, _ = tf.split(latent, num_or_size_splits=2, axis=0)  # z_loss not used here !
        if self.standardize_z:
            z_sv = self.standardize(z_sv, z_sv)
        gamma_rbf_coeff = self.compute_gamma(z_sv)
        # Computation of the K (kernel) matrix and its square root, parameters of the optim problem
        k_z_sv = self.kernel(z_sv, z_sv, gamma_rbf_coeff)
        num_stability_coeff = 1e-8 / gamma_rbf_coeff if not self.linear else 1e-8
        k_z_sqrt_sv = tf.linalg.sqrtm(tf.cast(k_z_sv + num_stability_coeff * tf.eye(tf.shape(z_sv)[0], dtype=self.dtype_), tf.float64))
        # Computing support vectors from OC-SVM problem
        if training:
            alpha_sv, = self.ocsvm_layer_train(k_z_sqrt_sv)
        else:  # validation:
            alpha_sv, = self.ocsvm_layer_valid(k_z_sqrt_sv)
        alpha_sv = tf.cast(alpha_sv, self.dtype_)  # CVXPylayer outputs float64
        alpha_sv = alpha_sv / (self.nu * n_subjects / 2)  # important to return to the unscaled problem

        return alpha_sv, k_z_sv

    def compute_rho(self, alpha_sv, k_z_sv):
        n_subjects = 2 * tf.cast(tf.shape(alpha_sv)[0], self.dtype_)
        # rho is the mean over all SV (0 < alpha_i < 1/(nu x n/2)) for numerical stability, as in LIBSVM
        sv_all = (alpha_sv - 1 / (self.nu * n_subjects)) ** 2 < (1 / (self.nu * n_subjects) - 1e-6) ** 2  # the middle is 1/nu*n, low bound 0, high bound 2/nu*n, small tolerance eps 1e-6
        e_j_all = tf.cast(sv_all, self.dtype_)  # sum of e_j_all will be the number of SV, to obtain the mean
        # max(., 1) : with no SV in the band, 0/0 would give a nan that poisons the loss, even for a term weighted by 0
        rho_mean = 1 / tf.maximum(tf.reduce_sum(e_j_all), 1.) * alpha_sv[None,] @ k_z_sv @ e_j_all[..., None]  # need to insert dimensions for correct multiplication, equivalent to a.T @ K @ e_j
        return rho_mean

    def decision_functions(self, alpha_sv, rho, k_z_sv_z):
        n_subjects = 2 * tf.cast(tf.shape(alpha_sv)[0], self.dtype_)
        return (alpha_sv[None,] @ k_z_sv_z - rho) * self.nu * (n_subjects / 2)  # Another de-normalization is necessary because of the scaled problem

    def compute_ocsvm_objective(self, alpha_sv, latent, k_z_sv, store_ocsvm=False):
        # Split the batch in two : one for solving ocsvm, one for loss computation
        z_sv_raw, z_loss = tf.split(latent, num_or_size_splits=2, axis=0)
        z_sv = z_sv_raw
        if self.standardize_z:
            z_sv, z_loss = self.standardize(z_sv, z_sv), self.standardize(z_loss, z_loss)
        gamma_rbf_coeff = self.compute_gamma(z_sv)
        rho = self.compute_rho(alpha_sv, k_z_sv)
        if store_ocsvm:
            self.last_ocsvms.append({"alpha_sv": tf.stop_gradient(alpha_sv), "rho": tf.stop_gradient(rho),
                                     "z_sv": tf.stop_gradient(z_sv_raw), "gamma_rbf_coeff": tf.stop_gradient(gamma_rbf_coeff)})

        # Expander : gradient only through z_sv (and alpha, rho if differentiate_dual), z_loss is stopped
        k_z_sv_loss_expander = self.kernel(z_sv, tf.stop_gradient(z_loss), gamma_rbf_coeff)  # "Kernel" matrix K_sv_loss of distance z_sv to z_loss
        if self.differentiate_dual:
            decision_functions_expander = self.decision_functions(alpha_sv, rho, k_z_sv_loss_expander)
        else:
            decision_functions_expander = self.decision_functions(tf.stop_gradient(alpha_sv), tf.stop_gradient(rho), k_z_sv_loss_expander)
        # Compactor : gradient only through z_loss
        k_z_sv_loss_compactor = self.kernel(tf.stop_gradient(z_sv), z_loss, gamma_rbf_coeff)
        decision_functions_compactor = self.decision_functions(tf.stop_gradient(alpha_sv), tf.stop_gradient(rho), k_z_sv_loss_compactor)

        # minus sign because deci_func is neg for outliers, relu to penalize only outliers (applied on z_loss !),  (nu as the upper bound of outliers seems the natural normalizing coefficient)
        ocsvm_objective_expander = (1 / self.nu) * tf.nn.relu(-decision_functions_expander) @ (tf.ones(alpha_sv[..., None].shape))  # sum to n so need but sparse so no need to divide by n
        ocsvm_objective_compactor = (1 / self.nu) * tf.nn.relu(-decision_functions_compactor) @ (tf.ones(alpha_sv[..., None].shape))

        # Both terms have the same value, only their gradients differ : the value of the objective is kept and its
        # gradient is beta1 * grad_expander + beta2 * grad_compactor (multiply_no_nan : a term weighted by 0 never gives nan)
        ocsvm_objective = (tf.stop_gradient(ocsvm_objective_expander)
                           + tf.math.multiply_no_nan(ocsvm_objective_expander - tf.stop_gradient(ocsvm_objective_expander), self.beta1)
                           + tf.math.multiply_no_nan(ocsvm_objective_compactor - tf.stop_gradient(ocsvm_objective_compactor), self.beta2))

        return tf.squeeze(ocsvm_objective)

    def decision_function(self, inputs):
        """Anomaly score (negative outside the support), mean of the decision functions of the last M OC-SVMs"""
        if not self.last_ocsvms:
            raise RuntimeError("No stored OC-SVM : train with n_last_ocsvms > 0, or train a final OC-SVM on the latent representations")
        _, latent = self(inputs, training=False)
        decision_functions_all = []
        for ocsvm in self.last_ocsvms:
            z_sv, z = ocsvm["z_sv"], latent
            if self.standardize_z:  # new samples standardized with the statistics of z_sv
                z_sv, z = self.standardize(z_sv, ocsvm["z_sv"]), self.standardize(z, ocsvm["z_sv"])
            k_z_sv_z = self.kernel(z_sv, z, ocsvm["gamma_rbf_coeff"])
            decision_functions_all.append(tf.squeeze(self.decision_functions(ocsvm["alpha_sv"], ocsvm["rho"], k_z_sv_z), axis=0))
        return tf.reduce_mean(tf.stack(decision_functions_all), axis=0)

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
            ocsvm_objective = self.compute_ocsvm_objective(alpha_sv, latent, k_z_sv, store_ocsvm=self.n_last_ocsvms > 0)

            # Total loss is reconstruction loss + OC-SVM objective
            total_loss = mse_recons_loss + self.ocsvm_coeff * ocsvm_objective

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return self.metrics_dict(total_loss, mse_recons_loss, ocsvm_objective, latent)

    def test_step(self, inputs):
        # Forward pass
        decoded, latent, alpha_sv, k_z_sv = self(inputs, training=False, inference=False)

        # Compute losses
        mse_recons_loss = tf.reduce_mean(tf.square(inputs - decoded))

        # OC-SVM objective
        ocsvm_objective = self.compute_ocsvm_objective(alpha_sv, latent, k_z_sv)
        total_loss = mse_recons_loss + self.ocsvm_coeff * ocsvm_objective

        return self.metrics_dict(total_loss, mse_recons_loss, ocsvm_objective, latent)

    @staticmethod
    def metrics_dict(total_loss, mse_recons_loss, ocsvm_objective, latent):
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


class BetaSchedule(tf.keras.callbacks.Callback):
    """Two-phase (beta1, beta2) schedule, e.g. expander only then expander + compactor (paper, Sec. IV-B).

    Note: with EarlyStopping(restore_best_weights=True), the restored weights may come from the first phase.
    """

    def __init__(self, switch_epoch, phase1=(1.0, 0.0), phase2=(0.5, 0.5)):
        super().__init__()
        self.switch_epoch = switch_epoch
        self.phase1, self.phase2 = phase1, phase2

    def on_epoch_begin(self, epoch, logs=None):
        self.model.set_betas(*(self.phase1 if epoch < self.switch_epoch else self.phase2))
