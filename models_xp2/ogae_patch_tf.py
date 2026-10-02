from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Flatten, Reshape

from ocsvm_guidance_tf import OCSVMGuidedAutoencoderBase


class OCSVMguidedPatchAutoencoder(OCSVMGuidedAutoencoderBase):
    """OgAE with localized OCSVM for Experiment 2 (brain MRI).

    Takes 15x15 patches, encoded into a latent vector of dimension 16 associated to the central voxel of the patch. Each
    training batch must contain co-localized patches (same voxel location across subjects), so that one OC-SVM per
    location is estimated. After training, a final OC-SVM is trained per location on the latent representations.

    Defaults are the ones used for the paper: no standardization of z and a fixed gamma = 1e-2. With an unconstrained
    latent scale, expander-heavy settings can saturate the RBF kernel (k ~ 0 between subjects) and collapse the decision
    function to -rho (supplementary S-D); gamma_rbf_coeff="scale" avoids this.
    Must be compiled with run_eagerly=True.
    """

    def __init__(self, batch_size_train, batch_size_valid, nb_channels=1, **ogae_kwargs):
        super().__init__(batch_size_train, batch_size_valid, **ogae_kwargs)
        # Encoder : 15x15 -> 11x11 -> 9x9 -> 3x3 -> 1x1
        self.conv1 = Conv2D(filters=3, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation="gelu")
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="gelu")
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(filters=12, kernel_size=(3, 3), strides=(3, 3), padding='valid', activation="gelu")
        self.bn3 = BatchNormalization()
        self.conv4 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="gelu")
        self.bn4 = BatchNormalization()
        self.flatten = Flatten()  # 1x1x16 -> 16

        # Decoder : 1x1 -> 3x3 -> 9x9 -> 11x11 -> 15x15
        self.reshape = Reshape((1, 1, 16))
        self.tconv4 = Conv2DTranspose(filters=12, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="gelu")
        self.tbn4 = BatchNormalization()
        self.tconv3 = Conv2DTranspose(filters=4, kernel_size=(3, 3), strides=(3, 3), padding='valid', activation="gelu")
        self.tbn3 = BatchNormalization()
        self.tconv2 = Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="gelu")
        self.tbn2 = BatchNormalization()
        self.tconv1 = Conv2DTranspose(filters=nb_channels, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation="sigmoid")

    def encoder(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        z = self.conv4(x)
        z = self.bn4(z)
        return self.flatten(z)

    def decoder(self, z):
        x = self.reshape(z)
        x = self.tconv4(x)
        x = self.tbn4(x)
        x = self.tconv3(x)
        x = self.tbn3(x)
        x = self.tconv2(x)
        x = self.tbn2(x)
        x_hat = self.tconv1(x)
        return x_hat
