# uad_ocsvm_guided_repr_learning 
This repository hosts the code for the paper titled "OCSVM-Guided Representation Learning for Unsupervised Anomaly Detection"

The implementation of models (including OgAE) benchmarked in the paper for xp1 (MNIST-C) and xp2 (brain MRI) in both tensorflow and pytorch are located in models_xp1/ and models_xp2/.

Bellow is the pseudo-code of our proposed OgAE model :
```
# Input: Batch of data x_batch [n, ...]
# Predefined: nu, lambda, gamma_rbf, standardize_z, M (number of last OC-SVMs stored) and (beta1, beta2) (Expander / compactor weights )

# --- 1. Forward Pass ---
z = encoder(x_batch)                # Latent space [n, latent_dim]
x_recon = decoder(z)                # Reconstructed input
reconstruction_loss = MSE(x_batch, x_recon)

# --- 2. OC-SVM Setup ---
z_sv, z_loss = split(z, 2)          # Split batch: n/2 for SV, n/2 for loss
z_sv = normalize(z_sv)              # Standardize support vectors
if standardize_z:   
    z_sv, z_loss = standardize(z_sv), standardize(z_loss)

# --- 3. RBF Kernel Matrix ---
pairwise_dists = squared_distances(z_sv, z_sv)
K_sv = exp(-gamma_rbf * pairwise_dists)  # Gram matrix of the z used for solving OCSVM problem
stability_coeff = 1e-8/gamma_rbf    # Gamma-adjusted stability term
K_sv_sqrt = matrix_sqrt(K_sv + stability_coeff*eye(n/2))  # SQRT of gram matrix for the cvx problem to be linear in parameter

# --- 4. Solve Scaled OC-SVM ---
# Scaled problem: min_alpha 0.5||K_sv_sqrt@alpha||^2 s.t. sum(alpha)=nu*n/2, 0≤alpha_i≤1  # for numerical stability
alpha_scaled = solve_dual_ocsvm(K_sv_sqrt, nu, n/2)
alpha_sv = alpha_scaled / (nu * n/2)  # Descale solution
# Bias term using support vectors (0 < alpha < 1/(nu*n))
is_sv = (alpha_sv > 1e-6) & (alpha_sv < 1/(nu*n/2))
rho = (transpose(alpha_sv[is_sv]) @ K_sv[is_sv] @ alpha_sv[is_sv]) / sum(is_sv)  # mean for stability as in LIBSVM

# --- 5. OC-SVM guidance loss ---
# Expander: gradient only through z_sv (and alpha_sv, rho), z_loss is stopped
deci_func_exp = (transpose(alpha_sv) @ exp(-gamma_rbf * squared_distances(z_sv, stop_gradient(z_loss))) - rho) * nu*n/2
# Compactor: gradient only through z_loss, z_sv, alpha_sv and rho are stopped
deci_func_comp = (transpose(stop_gradient(alpha_sv)) @ exp(-gamma_rbf * squared_distances(stop_gradient(z_sv), z_loss))
          - stop_gradient(rho)) * nu*n/2
# Relu of minus the decision values penalize only misclassified z_loss (nu*n/2 de-normalizes the scaled problem)
loss_exp, loss_comp = sum(relu(-deci_func_exp)) / nu, sum(relu(-deci_func_comp)) / nu
# Both terms have the same value: the loss value is kept, its gradient is beta1*grad(loss_exp) + beta2*grad(loss_comp)
ocsvm_g_loss = stop_gradient(loss_exp) + beta1 * (loss_exp - stop_gradient(loss_exp)) + beta2 * (loss_comp - stop_gradient(loss_comp))

# --- 6. Update ---
total_loss = reconstruction_loss + lambda * ocsvm_g_loss
update_weights(total_loss)  # by SGD
if iteration in last M iterations:
    store(alpha_sv, rho, z_sv)      # anomaly score = mean of the M decision functions (otherwise: final OC-SVM on all z)
```

## Repository structure
- `ocsvm_guidance_tf.py`, `ocsvm_guidance_torch.py`: core of OgAE (OCSVM-guidance loss, expander/compactor weighting), shared by both experiments.
- `models_xp1/`: Experiment 1 (MNIST-C). `models_coupled_*`: OgAE and Deep SVDD variants, `models_basic_*`: AE, VAE and siamese AE (used with reconstruction error or a decoupled OC-SVM).
- `models_xp2/`: Experiment 2 (brain MRI). `ogae_patch_*`: OgAE with localized OC-SVM, `ae_unet_baur_*` and `vq_vae_transformer_pinaya_*`: compared methods.

Modules are imported from the repository root (e.g. `from models_xp1.models_coupled_tf import OCSVMguidedAutoencoder`).

## Requirements
`cvxpy` and `cvxpylayers`, plus `tensorflow` or `torch`. The TensorFlow backend of cvxpylayers was removed in version 1.0: TensorFlow models require `cvxpylayers<1` (tested with `cvxpy<1.6`), and must be compiled with `run_eagerly=True`.

## Usage
```python
from models_xp1.models_coupled_tf import OCSVMguidedAutoencoder
from ocsvm_guidance_tf import BetaSchedule

model = OCSVMguidedAutoencoder(batch_size_train=100, batch_size_valid=100)  # lambda=1e-2, nu=0.03, gamma=1e-2
model.compile(optimizer="adam", run_eagerly=True)
# Expander only for the first half of the epochs, then expander + compactor
model.fit(train_ds, validation_data=valid_ds, epochs=20, callbacks=[BetaSchedule(switch_epoch=10)])
scores = model.decision_function(x_test)  # mean of the last M=10 OC-SVMs, negative = anomalous
```


