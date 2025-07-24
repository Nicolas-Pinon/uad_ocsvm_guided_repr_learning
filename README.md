# uad_ocsvm_guided_repr_learning 
This repository host the code for the paper titled "OCSVM-Guided Representation Learning for Unsupervised Anomaly Detection"

The implementation of models (including OgAE) benchmarked in the paper for xp1 (MNIST-C) and xp2 (brain MRI) in both tensorflow and pytorch are located in models_xp1/ and models_xp2/.

Bellow is the pseudo-code of our proposed OgAE model :
```
# Input: Batch of data x_batch [n, ...]
# Predefined: nu, lambda, gamma_rbf, jz_mode
# Modes: 'StopGradSV' (compactor), 'StopGradLoss' (expander), 'FullGrad' (both)

# --- 1. Forward Pass ---
z = encoder(x_batch)                # Latent space [n, latent_dim]
x_recon = decoder(z)                # Reconstructed input
reconstruction_loss = MSE(x_batch, x_recon)

# --- 2. OC-SVM Setup ---
z_sv, z_loss = split(z, 2)          # Split batch: n/2 for SV, n/2 for loss
z_sv = normalize(z_sv)              # Standardize support vectors
z_loss = normalize(z_loss)          # Standardize loss set

# --- 3. RBF Kernel Matrix ---
pairwise_dists = squared_distances(z_sv, z_sv)
K_sv = exp(-gamma_rbf * pairwise_dists)  # Gram matrix of the z used for solving OCSVM problem
stability_coeff = 1e-8/gamma_rbf    # Gamma-adjusted stability term
K_sv_sqrt = matrix_sqrt(K_sv + stability_coeff*eye(n/2))  # SQRT of gram matrix for the cvx problem to be linear in parameter

# --- 4. Solve Scaled OC-SVM ---
# Scaled problem: min_alpha 0.5||K_sv_sqrt@alpha||^2 s.t. sum(alpha)=nu*n/2, 0≤alpha_i≤1  # for numerical stability
alpha_scaled = solve_dual_ocsvm(K_sv_sqrt, nu, n/2)
alpha_sv = alpha_scaled / (nu * n/2)  # Descale solution

# --- 5. OC-SVM guidance loss ---
# Gradient control:
if jz_mode == "StopGradSV":       # COMPACTOR (only z_loss gets gradients)
    z_sv_compute = stop_gradient(z_sv)
    z_loss_compute = z_loss
elif jz_mode == "StopGradLoss":   # EXPANDER (only z_sv gets gradients)
    z_sv_compute = z_sv
    z_loss_compute = stop_gradient(z_loss)
else:                             # FullGrad (both get gradients)
    z_sv_compute = z_sv
    z_loss_compute = z_loss
# Kernel between SVs and loss set
dists_sv_loss = squared_distances(z_sv_compute, z_loss_compute)
K_sv_loss = exp(-gamma_rbf * dists_sv_loss)  # Gram matrix of z_sv and z_loss
# Bias term using support vectors (0 < alpha < 1/(nu*n))
is_sv = (alpha_sv > 1e-6) & (alpha_sv < 1/(nu*n/2))
rho = (transpose(alpha_sv[is_sv]) @ K_sv[is_sv] @ alpha_sv[is_sv]) / sum(is_sv)  # mean for stability as in LIBSVM
# OC-SVM loss (penalize outliers)
decision_values = transpose(alpha_sv) @ K_sv_loss - rho
ocsvm_g_loss = mean(relu(-decision_values))  # Relu of minus the decision values penalize only misclassified samples

# --- 6. Update ---
total_loss = reconstruction_loss + lambda * ocsvm_g_loss
update_weights(total_loss)  # by SGD
```
