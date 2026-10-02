"""OCSVM-guidance of autoencoder training (OgAE), TensorFlow implementation.

Notation follows the paper (Sec. III-B, eq. 7): each batch z is split into z_sv, used to solve the OC-SVM dual
problem (alpha*, rho*), and z_L (here `z_loss`), on which misclassified samples are penalized. The penalty is split into
  - the expander term (weight beta1): gradient flows through z_sv (and alpha*, rho*), z_loss is stopped,
  - the compactor term (weight beta2): gradient flows through z_loss only, z_sv, alpha* and rho* are stopped.
Both terms have the same forward value, so the loss value does not depend on (beta1, beta2): only the gradient does
(beta1 * g_expander + beta2 * g_compactor). (beta1, beta2) = (1, 1) is the full, unsplit gradient.

The OC-SVM layer (cvxpylayers) runs outside the TF graph: models must be compiled with `run_eagerly=True`.
"""
import collections

import cvxpy as cp
import tensorflow as tf
from cvxpylayers.tensorflow import CvxpyLayer

STD_EPS = 1e-6  # avoids division by zero when standardizing a collapsed latent dimension
FREE_SV_TOL = 1e-6  # tolerance on alpha bounds to identify free support vectors (used to compute rho)


def build_ocsvm_dual_layer(n, nu):
    """Differentiable solver of the scaled OC-SVM dual problem for n points (alpha_scaled = nu * n * alpha).

    sum_squares(K^1/2 @ alpha) == alpha.T @ K @ alpha, written this way so the problem is linear in its parameter (DPP).
    """
    alpha = cp.Variable(n)
    k_sqrt = cp.Parameter((n, n))
    constraints = [cp.sum(alpha) == nu * n, alpha >= 0, alpha <= 1]
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(k_sqrt @ alpha)), constraints)
    return CvxpyLayer(problem, parameters=[k_sqrt], variables=[alpha])


class OCSVMGuidedAutoencoderBase(tf.keras.Model):
    """Autoencoder trained with reconstruction + lambda * OCSVM-guidance loss. Subclasses define encoder() and decoder().

    Args:
        batch_size_train, batch_size_valid: batch sizes (must be even, one OC-SVM layer is built per batch size).
        ocsvm_coeff: lambda, weight of the OCSVM-guidance term.
        nu_ocsvm_coeff: nu of the OC-SVM.
        gamma_rbf_coeff: RBF gamma, a number, "scale" (1 / (dim * var(z_sv))) or "auto" (1 / dim), as in sklearn.
        beta1, beta2: weights of the expander and compactor gradients (see module docstring), can be changed during
            training with set_betas() or BetaSchedule.
        differentiate_dual: if False, the expander gradient does not flow through alpha* and rho*.
        standardize_z: standardize z_sv and z_loss (each with its own batch statistics) before the kernel.
        linear_kernel: use a linear kernel instead of the RBF kernel.
        n_last_ocsvms: number M of OC-SVMs of the last training iterations kept for decision_function() (0: none,
            a final OC-SVM must then be trained on encode() outputs).
    """

    def __init__(self, batch_size_train, batch_size_valid, ocsvm_coeff=1e-2, nu_ocsvm_coeff=0.03, gamma_rbf_coeff=1e-2,
                 beta1=1.0, beta2=0.0, differentiate_dual=True, standardize_z=True, linear_kernel=False,
                 n_last_ocsvms=0, **kwargs):
        super().__init__(**kwargs)
        self.dtype_ = tf.keras.mixed_precision.global_policy().name.replace("mixed_", "")
        self.ocsvm_coeff = ocsvm_coeff
        self.nu = nu_ocsvm_coeff
        if gamma_rbf_coeff not in ("scale", "auto") and not isinstance(gamma_rbf_coeff, (int, float)):
            raise ValueError(f"gamma_rbf_coeff must be a number, 'scale' or 'auto', got {gamma_rbf_coeff!r}")
        self.gamma_rbf_coeff = gamma_rbf_coeff
        self.beta1 = tf.Variable(beta1, trainable=False, dtype=self.dtype_, name="beta1")
        self.beta2 = tf.Variable(beta2, trainable=False, dtype=self.dtype_, name="beta2")
        self.differentiate_dual = differentiate_dual
        self.standardize_z = standardize_z
        self.linear_kernel = linear_kernel
        self.n_last_ocsvms = n_last_ocsvms
        self.ocsvm_buffer = collections.deque(maxlen=n_last_ocsvms)

        # Half of each batch is used to solve the OC-SVM problem, the other half for the loss
        if batch_size_train % 2 or batch_size_valid % 2:
            raise ValueError("Batch sizes must be even (each batch is split in z_sv and z_loss)")
        self.ocsvm_layers = {str(n // 2): build_ocsvm_dual_layer(n // 2, nu_ocsvm_coeff)
                             for n in {batch_size_train, batch_size_valid}}

    def set_betas(self, beta1, beta2):
        self.beta1.assign(beta1)
        self.beta2.assign(beta2)

    def call(self, inputs, training=False):
        latent = self.encoder(inputs)
        return self.decoder(latent), latent

    def encode(self, inputs):
        return self(inputs, training=False)[1]

    def _gamma(self, z_sv):
        if self.gamma_rbf_coeff == "scale":
            gamma = 1 / (tf.cast(tf.shape(z_sv)[1], self.dtype_) * tf.stop_gradient(tf.math.reduce_variance(z_sv)))
            return tf.where(tf.math.is_inf(gamma), tf.constant(1e32, self.dtype_), gamma)  # variance 0 (collapse)
        if self.gamma_rbf_coeff == "auto":
            return 1 / tf.cast(tf.shape(z_sv)[1], self.dtype_)
        return tf.constant(self.gamma_rbf_coeff, self.dtype_)

    def _kernel(self, a, b, gamma):
        """Kernel matrix K_ij = k(a_i, b_j)."""
        if self.linear_kernel:
            return tf.matmul(a, b, transpose_b=True)
        sq_dists = tf.reduce_sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return tf.exp(-gamma * sq_dists)

    def solve_ocsvm_problem(self, z_sv, gamma):
        """Returns the (unscaled) dual solution alpha* and the kernel matrix of z_sv."""
        n_half = z_sv.shape[0]
        k_sv = self._kernel(z_sv, z_sv, gamma)
        num_stability_coeff = 1e-8 if self.linear_kernel else 1e-8 / gamma
        k_sqrt = tf.linalg.sqrtm(tf.cast(k_sv + num_stability_coeff * tf.eye(n_half, dtype=self.dtype_), tf.float64))
        alpha_scaled, = self.ocsvm_layers[str(n_half)](k_sqrt)
        return tf.cast(alpha_scaled, self.dtype_) / (self.nu * n_half), k_sv

    def _rho(self, alpha, k_sv):
        """rho* averaged over free support vectors (0 < alpha_j < 1 / (nu * n)) for stability, as in LIBSVM."""
        upper_bound = 1 / (self.nu * alpha.shape[0])
        free_sv = tf.cast((alpha > FREE_SV_TOL) & (alpha < upper_bound - FREE_SV_TOL), self.dtype_)
        # Guard: with no free SV, 0/0 would give a NaN that poisons the loss even for a zero-weighted term
        return tf.reduce_sum(tf.linalg.matvec(k_sv, alpha) * free_sv) / tf.maximum(tf.reduce_sum(free_sv), 1.)

    def _decision_function(self, alpha, rho, k):
        """Decision function of the OC-SVM on the columns of k, de-normalized from the scaled problem."""
        return (tf.linalg.matvec(k, alpha, transpose_a=True) - rho) * self.nu * alpha.shape[0]

    def ocsvm_guidance_loss(self, latent, store_ocsvm=False):
        z_sv, z_loss = tf.split(latent, num_or_size_splits=2, axis=0)
        sv_mean, sv_std = tf.reduce_mean(z_sv, axis=0), tf.math.reduce_std(z_sv, axis=0) + STD_EPS
        if self.standardize_z:
            z_sv = (z_sv - sv_mean) / sv_std
            z_loss = (z_loss - tf.reduce_mean(z_loss, axis=0)) / (tf.math.reduce_std(z_loss, axis=0) + STD_EPS)
        gamma = self._gamma(z_sv)
        alpha, k_sv = self.solve_ocsvm_problem(z_sv, gamma)
        rho = self._rho(alpha, k_sv)
        sg = tf.stop_gradient

        if store_ocsvm:
            self.ocsvm_buffer.append(dict(alpha=sg(alpha), rho=sg(rho), z_sv=sg(z_sv), gamma=sg(gamma),
                                          mean=sg(sv_mean), std=sg(sv_std)))

        alpha_exp, rho_exp = (alpha, rho) if self.differentiate_dual else (sg(alpha), sg(rho))
        decision_exp = self._decision_function(alpha_exp, rho_exp, self._kernel(z_sv, sg(z_loss), gamma))
        decision_comp = self._decision_function(sg(alpha), sg(rho), self._kernel(sg(z_sv), z_loss, gamma))
        # Penalize only misclassified z_loss (negative decision function), nu as upper bound of outliers fraction
        loss_exp = tf.reduce_sum(tf.nn.relu(-decision_exp)) / self.nu
        loss_comp = tf.reduce_sum(tf.nn.relu(-decision_comp)) / self.nu

        # Straight-through weighting: forward value is the unsplit loss, gradient is beta1 * g_exp + beta2 * g_comp.
        # multiply_no_nan so that a zero-weighted term never propagates a NaN.
        return (sg(loss_exp)
                + tf.math.multiply_no_nan(loss_exp - sg(loss_exp), self.beta1)
                + tf.math.multiply_no_nan(loss_comp - sg(loss_comp), self.beta2))

    def decision_function(self, inputs):
        """Anomaly score (negative outside the support) as the mean of the last M stored OC-SVMs' decision functions.

        New samples are standardized with the statistics of each stored z_sv.
        """
        if not self.ocsvm_buffer:
            raise RuntimeError("No stored OC-SVM: train with n_last_ocsvms > 0, or fit a final OC-SVM on encode()")
        latent = self.encode(inputs)
        scores = []
        for ocsvm in self.ocsvm_buffer:
            z = (latent - ocsvm["mean"]) / ocsvm["std"] if self.standardize_z else latent
            k = self._kernel(ocsvm["z_sv"], z, ocsvm["gamma"])
            scores.append(self._decision_function(ocsvm["alpha"], ocsvm["rho"], k))
        return tf.reduce_mean(tf.stack(scores), axis=0)

    def _compute_losses(self, inputs, training):
        decoded, latent = self(inputs, training=training)
        mse_recons_loss = tf.reduce_mean(tf.square(inputs - decoded))
        ocsvm_objective = self.ocsvm_guidance_loss(latent, store_ocsvm=training and self.n_last_ocsvms > 0)
        total_loss = mse_recons_loss + self.ocsvm_coeff * ocsvm_objective
        return total_loss, mse_recons_loss, ocsvm_objective, latent

    @staticmethod
    def _metrics_logs(total_loss, mse_recons_loss, ocsvm_objective, latent):
        pairwise_distances = tf.norm(latent[:, None, :] - latent[None, :, :], axis=-1)  # latent spread monitoring
        return {"total_loss": total_loss, "mse_recons_loss": mse_recons_loss, "ocsvm_objective": ocsvm_objective,
                "mean_pairwise_distance": tf.reduce_mean(pairwise_distances), "std_z": tf.math.reduce_std(latent)}

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            total_loss, mse_recons_loss, ocsvm_objective, latent = self._compute_losses(inputs, training=True)
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return self._metrics_logs(total_loss, mse_recons_loss, ocsvm_objective, latent)

    def test_step(self, inputs):
        return self._metrics_logs(*self._compute_losses(inputs, training=False))


class BetaSchedule(tf.keras.callbacks.Callback):
    """Two-phase (beta1, beta2) schedule, e.g. expander only then expander + compactor (paper, Sec. IV-B).

    The default reproduces the paper's best setting "(1, 0) -> (0.5, 0.5)", which was run with the full unsplit
    gradient, i.e. (1, 1) here: (0.5, 0.5) gives the same direction with half the OCSVM-guidance gradient magnitude.
    Note: with EarlyStopping(restore_best_weights=True), the restored weights may come from the first phase.
    """

    def __init__(self, switch_epoch, phase1=(1.0, 0.0), phase2=(1.0, 1.0)):
        super().__init__()
        self.switch_epoch = switch_epoch
        self.phase1, self.phase2 = phase1, phase2

    def on_epoch_begin(self, epoch, logs=None):
        self.model.set_betas(*(self.phase1 if epoch < self.switch_epoch else self.phase2))
