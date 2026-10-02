import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D, Conv2DTranspose, Flatten, Reshape

from ocsvm_guidance_tf import OCSVMGuidedAutoencoderBase


class OCSVMguidedPatchAutoencoder(OCSVMGuidedAutoencoderBase):
    """OgAE with localized OCSVM for Experiment 2 (brain MRI), architecture of supplementary material S-F2.

    Takes 15x15 patches; each training batch must contain co-localized patches (same voxel location across subjects),
    so that one OC-SVM per location is estimated. The latent representation (5x5x16 feature map, flattened) is associated
    to the central voxel of the patch. After training, a final OC-SVM is fitted per location on encode() outputs.

    Defaults are the ones used for the paper: no standardization of z and a fixed gamma = 1e-2. With an unconstrained
    latent scale, expander-heavy settings can saturate the RBF kernel (k ~ 0 between subjects) and collapse the decision
    function to -rho (supplementary S-D); gamma_rbf_coeff="scale" or standardize_z=True avoid this.
    Must be compiled with run_eagerly=True.
    """

    def __init__(self, batch_size_train, batch_size_valid, nb_channels=1, standardize_z=False, **ogae_kwargs):
        super().__init__(batch_size_train, batch_size_valid, standardize_z=standardize_z, **ogae_kwargs)
        # 'valid' convolutions: 15 -> 11 -> 9 -> 7 -> 5
        self.encoder_blocks = [tf.keras.Sequential([Conv2D(n_filters, kernel_size, padding='valid'),
                                                    BatchNormalization(), Activation('gelu')])
                               for n_filters, kernel_size in ((3, 5), (4, 3), (12, 3), (16, 3))]
        self.flatten = Flatten()
        self.decoder_reshape = Reshape((5, 5, 16))
        self.decoder_blocks = [tf.keras.Sequential([Conv2DTranspose(n_filters, 3, padding='valid'),
                                                    BatchNormalization(), Activation('gelu')])
                               for n_filters in (12, 4, 3)]
        self.output_layer = Conv2DTranspose(nb_channels, 5, padding='valid', activation='sigmoid')

    def encoder(self, inputs):
        x = inputs
        for block in self.encoder_blocks:
            x = block(x)
        return self.flatten(x)

    def decoder(self, latent):
        x = self.decoder_reshape(latent)
        for block in self.decoder_blocks:
            x = block(x)
        return self.output_layer(x)
