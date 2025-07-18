import os
import shutil
import warnings

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Layer, Conv2D, Conv2DTranspose, Dropout, Dense
from hilbertcurve.hilbertcurve import HilbertCurve
from sklearn.cluster import k_means
from keras_nlp.layers import TokenAndPositionEmbedding, TransformerDecoder
from keras_nlp.metrics import Perplexity
from cv2 import resize, INTER_NEAREST
from scipy.ndimage import gaussian_filter



class PinayaTransformerModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, seq_length, dropout):
        super().__init__()
        self.embed = TokenAndPositionEmbedding(vocabulary_size=vocab_size, sequence_length=seq_length, embedding_dim=d_model, mask_zero=False)
        self.embed_drop = Dropout(dropout)
        self.decoders = [TransformerDecoder(intermediate_dim=dff, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        self.dropout = Dropout(dropout)
        self.final_layer = Dense(vocab_size)
        self.perplexity = Perplexity(from_logits=True, name="perplexity")
        self.losses_name = ['loss', 'accuracy', 'top 5 accuracy', "unique true", "unique predictions", "perplexity"]

    def call(self, inputs, training=False):
        x = self.embed(inputs)
        for k in range(len(self.decoders)):
            x = self.decoders[k](x)
        x = self.dropout(x)
        output = self.final_layer(x)
        return output

    def train_step(self, batch):
        # print('BATCH',len(batch))
        x_train, y_train = batch
        with tf.GradientTape() as tape:
            y_pred = self(x_train, training=True)
            loss = tf.math.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_train, y_pred, from_logits=True))
            acc = tf.math.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_train, y_pred))
            top_k_acc = tf.math.reduce_mean(tf.keras.metrics.sparse_top_k_categorical_accuracy(y_train, y_pred, k=5))
            perpl = self.perplexity(y_train, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Keep track of the losses by returning them :
        return {"loss": loss, "accuracy": acc, "top 5 accuracy": top_k_acc, "perplexity": perpl}

    def test_step(self, batch):
        x_train, y_train = batch
        y_pred = self(x_train, training=False)
        # print('TEST STEP', y_pred.shape)
        loss = tf.math.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_train, y_pred, from_logits=True))
        acc = tf.math.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_train, y_pred))
        top_k_acc = tf.math.reduce_mean(tf.keras.metrics.sparse_top_k_categorical_accuracy(y_train, y_pred, k=5))
        perpl = self.perplexity(y_train, y_pred)

        return {"loss": loss, "accuracy": acc, "top 5 accuracy": top_k_acc, "perplexity": perpl}


class ImageRestorer:
    def __init__(self, transformer, vqvae, sequence_generator, thr):
        self.transformer = transformer
        self.vqvae = vqvae
        self.sequence_generator = sequence_generator
        self.seq_len = self.sequence_generator.seq_len
        self.latent_dims = self.sequence_generator.latent_dims
        self.thr = thr
        print('THRESHOLD', self.thr)

    def __call__(self, x, batch_size=32, return_reconstructions=True):
        '''
        Restore a batch of 2D images/slices (shape (B,X,Y,C)) or 3D volumes (shape (B,X,Y,Z,C)).
        Please note that 4D Tensor will be automatically interpreted as 2D images and 5D Tensor as 3D volumes
        (i.e. Batch and Channel dimensions are requiered).
        '''
        warnings.warn("entered __call__ from image restorer")
        restored_images, up_resampling_maks = self.restore_images(x, batch_size=batch_size)
        warnings.warn("finished restorating and resampling masks")
        if return_reconstructions:
            recons = self.vqvae.predict(x, batch_size=batch_size)
            warnings.warn("finished recons")
            return restored_images, up_resampling_maks, recons
        else:
            return restored_images, up_resampling_maks

    def restore_images(self, data, batch_size=32):
        discrete_latent_indexes = self._get_latent_indexes(data, batch_size=batch_size)
        restored_sequences, resampling_masks = self.restore_sequences(discrete_latent_indexes, batch_size=batch_size)
        resampling_masks = self.sequence_generator.reverse_process_features(resampling_masks)
        restored_sequences = self.sequence_generator.reverse_process_features(restored_sequences)
        up_resampling_masks = np.zeros(data.shape[:3])  # (B,X,Y)
        for k in range(resampling_masks.shape[0]):
            up_resampling_masks[k] = gaussian_filter(resize(resampling_masks[k], data.shape[1:3], interpolation=INTER_NEAREST), sigma=5)  # resize to original image shape
        restored_images = self.vqvae.decoder.predict(self.vqvae.get_encodings_from_indices(restored_sequences), batch_size=batch_size)

        return restored_images, up_resampling_masks

    def restore_sequences(self, seq, batch_size=32):
        '''
        Reads features with specific order and outputs with shape (B, seq_len)
        Input 'batch' shape is (B,h,w) (2d indexes of codebook vectors)
        '''
        flat_seq = np.hstack((np.zeros((seq.shape[0], 1)), self.sequence_generator.process_features(seq)))
        preds_inds = flat_seq[:, 1:]
        probas = self._softmax(self.transformer.predict(flat_seq[:, :-1], batch_size=batch_size), axis=-1)
        resampling_masks = np.zeros((seq.shape[0], self.seq_len))
        restored_sequences = np.copy(flat_seq[:, 1:])
        for i in range(resampling_masks.shape[0]):
            for j in range(resampling_masks.shape[1]):
                if probas[i, j, int(preds_inds[i, j])] < self.thr:
                    resampling_masks[i, j] = 1.
                    new = 0
                    while new == 0:
                        new = np.random.choice(np.arange(self.vqvae.num_embeddings + 1), p=probas[i, j])
                        restored_sequences[i, j] = new
        return restored_sequences - 1, resampling_masks

    def _get_latent_indexes(self, images, batch_size=32):
        latent_indexes = np.zeros((images.shape[0], *self.latent_dims))
        for k in range(int(np.ceil(images.shape[0] / batch_size))):
            latent_indexes[k * batch_size:(k + 1) * batch_size, ] = self.vqvae.get_embeds_indices_from_images(
                images[k * batch_size:(k + 1) * batch_size, ], batch_size=batch_size) + 1  # (B,w,h) indices of encoding vectors used for the discrete latent representation
        return latent_indexes

    def _get_restored_images_from_seq(self, restored_sequences, resampling_masks, image_shape):
        '''
        Take as input (flat) restored sequences and associated resampling masks (output of retore_sequences)
        and return the full 2D restoration error map (of shape (B,H,W))
        '''
        resampling_masks = self.sequence_generator.reverse_process_features(resampling_masks)
        restored_sequences = self.sequence_generator.reverse_process_features(restored_sequences)

        up_resampling_masks = np.zeros(image_shape)  # (B,X,Y)
        for k in range(resampling_masks.shape[0]):
            up_resampling_masks[k] = resize(resampling_masks, self.data.shape[1:3], interpolation=INTER_NEAREST)  # resize to original image shape
        restored_images = self.vqvae.decoder(restored_sequences)

        return restored_images, up_resampling_masks

    def _softmax(self, x, axis=None):
        return (np.exp(x - np.max(x, axis=axis, keepdims=True)) / np.exp(x - np.max(x, axis=axis, keepdims=True)).sum(axis=axis, keepdims=True))



class EMAWarmUpCallback(tf.keras.callbacks.Callback):
    def __init__(self, decay_start, decay_end=None, steps=None):
        super().__init__()
        self.decay_start = decay_start
        if decay_start is not None and steps is not None:
            self.decay_step = (decay_end - decay_start)/steps
        self.n_steps = steps if steps is not None else 0
        self.current_decay = decay_start

    def on_train_begin(self, logs=None):
        self.model.vq.vector_quantizer._set_decay(self.decay_start)

    def on_epoch_end(self, epoch, logs=None):
        if epoch<self.n_steps:
            self.current_decay += self.decay_step
            self.model.vq.vector_quantizer._set_decay(self.current_decay)


## Dict used as reference for the 32 possible sequence orderings
## Less explicit than sspecifying directly to SequenceGenerator the parameters
## but easier to follow which experiments had already been launched

## Flip indicates if the features tensor must be flip vertically(1), horizontally(2), both([1,2]) or neither (0)
sequences_orderings={
    0:{'ordering': 'raster', 'flip': 0, 'rotate':False},
    1:{'ordering': 'raster', 'flip': 1, 'rotate':False},
    2:{'ordering': 'raster', 'flip': 2, 'rotate':False},
    3:{'ordering': 'raster', 'flip': [1,2], 'rotate':False},
    4:{'ordering': 'raster', 'flip': 0, 'rotate':True},
    5:{'ordering': 'raster', 'flip': 1, 'rotate':True},
    6:{'ordering': 'raster', 'flip': 2, 'rotate':True},
    7:{'ordering': 'raster', 'flip': [1,2], 'rotate':True},
    8:{'ordering': 's_curve', 'flip': 0, 'rotate':False},
    9:{'ordering': 's_curve', 'flip': 1, 'rotate':False},
    10:{'ordering': 's_curve', 'flip': 2, 'rotate':False},
    11:{'ordering': 's_curve', 'flip': [1,2], 'rotate':False},
    12:{'ordering': 's_curve', 'flip': 0, 'rotate':True},
    13:{'ordering': 's_curve', 'flip': 1, 'rotate':True},
    14:{'ordering': 's_curve', 'flip': 2, 'rotate':True},
    15:{'ordering': 's_curve', 'flip': [1,2], 'rotate':True},
    16:{'ordering': 'hilbert_curve', 'flip': 0, 'rotate':False},
    17:{'ordering': 'hilbert_curve', 'flip': 1, 'rotate':False},
    18:{'ordering': 'hilbert_curve', 'flip': 2, 'rotate':False},
    19:{'ordering': 'hilbert_curve', 'flip': [1,2], 'rotate':False},
    20:{'ordering': 'hilbert_curve', 'flip': 0, 'rotate':True},
    21:{'ordering': 'hilbert_curve', 'flip': 1, 'rotate':True},
    22:{'ordering': 'hilbert_curve', 'flip': 2, 'rotate':True},
    23:{'ordering': 'hilbert_curve', 'flip': [1,2], 'rotate':True},
    24:{'ordering': 'random', 'flip': 0, 'rotate':False} # random ordering doesnt need the 8 variations, only to do 8 experiments with 'random' config
}


class Encoder2DPinaya(Model):
    def __init__(self, filters=256, kernel_size=(4, 4), stride=(2, 2), activation='relu', n_residuals=3, *args, **kwargs):
        super().__init__()
        self.convs = Sequential([Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding="same", activation=activation) for _ in range(3)])
        self.residuals = Sequential([ResidualBlock(filters=filters, ) for _ in range(n_residuals)])

    def call(self, inputs, training=None, mask=None):
        return self.residuals(self.convs(inputs))


class Decoder2DPinaya(Model):
    def __init__(self, nb_channels, filters=256, kernel_size=(4, 4), stride=(2, 2), activation='relu', n_residuals=3, *args, **kwargs):
        super().__init__()
        self.residuals = Sequential([ResidualBlock(filters=filters, ) for _ in range(n_residuals)])
        self.convs = Sequential([Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=stride, padding="same", activation=activation) for _ in range(3)]
                                + [Conv2D(filters=nb_channels, kernel_size=(1, 1), strides=(1, 1), activation='relu'), ])

    def call(self, inputs, training=None, mask=None):
        return self.convs(self.residuals(inputs))


class VQVAEGeneric(tf.keras.models.Model):
    """"VQVAE class in its generic form, can be used to instanciate differente convolutional siamese networks.
        Generic form like SiameseGeneric, classes specific to different AutoEncoder architectures are defined below """

    def __init__(self, nb_channels, nb_filters=(8, 12, 16), num_embeddings=512, patch_size=(63, 63), type_="zara", name="vqvae",
                 latent_dim=None, dropout=0, GRN_kernel=(3, 3), apply_batch_norm=True,
                 codebook_learning="loss", reservoir_size=1024, reestimate_step_iter=100, epoch_start=2, num_epoch_train=2, beta=0.25, compression="high",
                 preprocessing_layer=True, **kwargs):
        super(VQVAEGeneric, self).__init__()
        self.nb_channels = nb_channels
        if (type_ == "gatedresnet" and latent_dim is not None):
            self.latent_dim = latent_dim
        elif isinstance(nb_filters, int):
            self.latent_dim = nb_filters
        elif isinstance(nb_filters, list):
            self.latent_dim = nb_filters[-1]
        self.num_embeddings = num_embeddings
        self.codebook_learning = codebook_learning
        self.reservoir_size = reservoir_size
        self.reestimate_step_iter = reestimate_step_iter
        self.epoch_start = epoch_start
        self.num_epoch_train = num_epoch_train
        self.beta = beta
        self.perplexity = tf.keras.metrics.Mean(name="perplexity")
        self.preprocessing = preprocessing_layer
        if self.preprocessing:
            self.preprocess_layer = tf.keras.Sequential([tf.keras.layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=[-0.1, 0.1], fill_mode="constant"),
                                                         RandomBrightness(factor=(-0.1, 0.1), value_range=(0., 1.)),
                                                         tf.keras.layers.RandomContrast(factor=0.1)])
        # tf.keras.layers.RandomBrightness(factor= (-0.1,0.1), value_range=(0., 1.)),
        self.encoder = Encoder2DPinaya(filters=nb_filters, kernel_size=(4, 4), stride=(2, 2), activation='relu', n_residuals=3)
        self.decoder = Decoder2DPinaya(nb_channels=nb_channels, filters=nb_filters, kernel_size=(4, 4), stride=(2, 2), activation='relu', n_residuals=3)
        self.vq = BottleneckVQ(self.num_embeddings, self.latent_dim, codebook_learning=self.codebook_learning, reservoir_size=self.reservoir_size,
                               reestimate_step_iter=self.reestimate_step_iter, epoch_start=self.epoch_start,
                               num_epoch_train=self.num_epoch_train)

        self.loss_names = ["total_loss", "reconstruction_loss", "commitment_loss"]
        if self.codebook_learning == "loss":
            self.loss_names.append("codebook_loss")

    def call(self, x_batch, training=False):
        if training and self.preprocessing:
            x_batch = self.preprocess_layer(x_batch)
        out_enc = self.encoder(x_batch)
        encodings, out_vq = self.vq(out_enc, training=training)
        out_dec = self.decoder(out_vq)

        # Calculate the losses.
        reconstruction_loss = (
            tf.reduce_mean((x_batch - out_dec) ** 2)
        )

        self.add_loss(reconstruction_loss)

        avg_probs = tf.reduce_mean(encodings, axis=0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))
        self.perplexity.update_state(perplexity)

        return out_dec

    def get_vq_usage(self, x_batch):
        out_enc = self.encoder(x_batch)
        encodings = self.vq(out_enc)[0]
        avg_probs = tf.reduce_mean(encodings, axis=0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))
        return {
            'encodings': encodings,
            'perplexity': perplexity
        }

    def get_embeds_indices_from_images(self, x_batch, apply_preprocessing=True, batch_size=32):
        final_output_indices = []
        for k, _ in enumerate(range(0, x_batch.shape[0], batch_size)):
            images = x_batch[k * batch_size:(k + 1) * batch_size]
            if apply_preprocessing:
                images = self.preprocess_layer(images, training=False)
            out_enc = self.encoder(images, training=False)
            flattened = tf.reshape(out_enc, [-1, self.latent_dim])
            indices = self.vq.vector_quantizer.get_code_indices(flattened)
            final_output_indices.append(np.reshape(indices, out_enc.shape[:-1]))
        return np.concatenate(final_output_indices, axis=0)  # (B, h_latent, w_latent)

    def get_embeds_indices_from_images_bis(self, x_batch, apply_preprocessing=True):
        if apply_preprocessing:
            images = self.preprocess_layer(x_batch, training=False)
        out_enc = self.encoder(images, training=False)
        flattened = tf.reshape(out_enc, [-1, self.latent_dim])
        indices = self.vq.vector_quantizer.get_code_indices(flattened)
        return indices

    def get_encodings_from_indices(self, encoding_indices):
        encodings = tf.one_hot(encoding_indices, self.vq.vector_quantizer.num_embeddings)
        quantized = tf.matmul(encodings, self.vq.vector_quantizer.embeddings, transpose_b=True)
        return quantized

    def train_step(self, x_batch):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            self(x_batch, training=True)
            codebook_loss = self.losses[-2]
            commitment_loss = self.losses[-1]
            reconstruction_loss = self.losses[0]
            perplexity = self.metrics[0].result()

            total_loss = reconstruction_loss + self.beta * codebook_loss + commitment_loss

        # Backpropagation.
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        res = {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "commitment_loss": commitment_loss
        }

        if self.codebook_learning == "loss":
            res["codebook_loss"] = codebook_loss
        res["perplexity"] = perplexity

        # Log results.
        return res

    def test_step(self, x_batch):
        # Same as train step but without storing gradients
        self(x_batch, training=False)
        codebook_loss = self.losses[-2]
        commitment_loss = self.losses[-1]
        reconstruction_loss = self.losses[0]

        perplexity = self.perplexity.result()

        total_loss = reconstruction_loss + codebook_loss + commitment_loss

        res = {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "commitment_loss": commitment_loss
        }

        if self.codebook_learning == "loss":
            res["codebook_loss"] = codebook_loss

        res["perplexity"] = perplexity

        return res


class VQVAEPinaya(VQVAEGeneric):
    def __init__(self, nb_channels, nb_filters=256, num_embeddings=32, type_="pinaya", name="vqvae", codebook_learning="ema", **kwargs):
        super().__init__(nb_channels, nb_filters=nb_filters, num_embeddings=num_embeddings, type_=type_, name=name, codebook_learning=codebook_learning, **kwargs)


######
## Sequence generator for Transformer (it may be better to rename this file generators.py ?)
######

class SequenceGenerator(tf.keras.utils.Sequence):
    def __init__(self, data=None, model=None, ordering=None, preprocessing=True, batch_size=16, seed=15):
        super().__init__()
        ''''
        Generator of sequences used to train Pinaya's Transformer from sequences of encodings vectors' indices
        given by a VQVAE.

        If model is None, data is assumed to be sequences of features already extracted from VQVAE (B,h,w),
        else data is assumed to be full 2D-images/slices (B,H,W,C) and the SequenceGenerator will pass patchs
        though the model (VQVAE) to get the sequences of codebook vectors
        In that case, it is expected to have a method called 'get_embeds_indices_from_image'
        that takes input images (B,H,W,C) and returns a Tensor (B,h,w) containing the indices of the encoding vectors
        using to quantize the latent space of the batch images.
        N.B.: capital letter refers to shape in the images' domain while lowercase letters are used for latent space dimensions

        data is expected to be numpy array.
        ordering can be an int (see config for for reference) or a dict containing the parameters of list (useful for random ordering).
        '''
        self.model=model
        self.number = data.shape[0]
        self.preprocessing = preprocessing
        if model is None:
            self.sequence_data = data
            self.latent_dims = data.shape[1:3]
        else:
            self.data = data
            self.sequence_data = self._get_sequences_data_from_images()
            self.latent_dims = self.sequence_data.shape[1:3]
        self.batch_size = batch_size
        self.nb_samples = data.shape[0]
        if isinstance(seed, int):
            self.seed = seed
        else:
            self.seed = np.random.randint(0,1000000)
        # self.batch_sizes = [self.batch_size,] * self.nb_samples//self.batch_size if self.nb_samples%self.batch_size==0 \
                            # else [self.batch_size] * self.nb_samples//self.batch_size + [self.nb_samples%self.batch_size,]
        self.splits = [self.batch_size*k for k in range(1,self.number//self.batch_size)]
        self.seq_len = self.latent_dims[0] * self.latent_dims[1]
        self._get_ordering_config(ordering)

        self._get_ordering_path(self.order)
        self._generate_indexes()

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, index):
        sequences = self.sequence_data[self.indexes[index],:]
        # if self.model is not None:
        #     images = self.data[self.indexes[index],:]
        #     sequences = self.model.get_embeds_indices_from_images(images, apply_preprocessing=self.preprocessing) # (B,H,W,C)
        # else:
        #     sequences = self.data[self.indexes[index],:]
        sequences = self.process_features(sequences) # apply transformations and read features sequence in specific order

        # returns x_train, y_train both with shape (B, self.seq_len)
        # A column of zeros is added at the beginning of x_train
        return np.concatenate((np.zeros((self.batch_size,1)),sequences[:,:-1]), axis=1), sequences

    def on_epoch_end(self):
        self.seed +=12345
        self._generate_indexes()
        if self.preprocessing:
            self.sequence_data = self._get_sequences_data_from_images()

    def _get_sequences_data_from_images(self):
        return self.model.get_embeds_indices_from_images(self.data, apply_preprocessing=self.preprocessing, batch_size=128) + 1

    def _get_ordering_config(self, ordering):
        if isinstance(ordering,int):
            order_config = sequences_orderings[ordering]
        elif isinstance(ordering, dict):
            order_config = ordering
        elif isinstance(ordering, list):
            self.random_path = ordering
            self.order = "random"
            self.flip = 0
            self.rotation = 0
            return
        else:
            raise Exception('Ordering must be a dict, an int (reference to predefined orderings) or a list (for random ordering)')

        self.order = order_config['ordering']
        if order_config['flip'] in (None, 0):
            self.flip = None
        else:
            self.flip = order_config['flip']
        self.rotation = 1 if int(order_config['rotate'])==90 else order_config['rotate'] # rotation can be 0, 1 or 90 (degrees)

    def _generate_indexes(self):
        p = np.random.permutation(self.nb_samples)[:self.batch_size*(self.number//self.batch_size)]
        self.indexes = np.split(p, self.splits)

    def _get_ordering_path(self, ordering):
        if ordering=='raster':
            self.sequence_path = slice(None, None, None)
            self.reverse_path = np.arange(self.seq_len)
        elif ordering=='s_curve':
            self._get_s_curve_path()
            self.reverse_path = np.argsort(self.sequence_path)
        elif ordering=='hilbert_curve':
            self._get_hilbert_path()
            self.reverse_path = np.argsort(self.sequence_path)
        elif ordering=='random' and not hasattr(self, 'sequence_path'):
            self.sequence_path = np.random.permutation(np.arange(self.seq_len))
            self.reverse_path = np.argsort(self.sequence_path)

    def _get_s_curve_path(self):
        order = np.reshape(np.arange(self.seq_len),self.latent_dims)
        order[1::2] = np.flip(order[1::2], axis=-1)
        self.sequence_path = order.flatten()

    def _get_hilbert_path(self): # give the hilbert path for a reference sequence (2d)
        for index, n in enumerate(self.latent_dims):
            if not((n & (n-1) == 0) and n != 0):
                print(f'Dimension {index} of latent space in not a power of 2. Sequences reading will not be an exact Hilbert Curve')
        dim_h_curve = int(np.ceil(max(np.log2(self.latent_dims))))
        hilbert_curve = HilbertCurve(n=2, p=dim_h_curve)
        full_path = hilbert_curve.points_from_distances(np.arange(2**(2*dim_h_curve)))
        crop_path = [] # is cropped only if dimension is not a power of 2
        for k in range(len(full_path)):
            if full_path[k][0]<self.latent_dims[0] and full_path[k][1]<self.latent_dims[1]:
                crop_path.append(full_path[k])
        self.sequence_path = np.array([np.reshape(np.arange(self.seq_len),(self.latent_dims))[x,y] for x,y in crop_path]).transpose()

    def process_features(self, s): #
        '''
        Transform a tensor of features indexes (B,H,W,C) to sequences (B,L) with a specific order
        '''
        s = np.rot90(s, self.rotation, axes=(1,2,))
        s = np.flip(s, self.flip) if self.flip is not None else s
        s = np.reshape(s,(-1, self.seq_len))
        return s[:,self.sequence_path]

    def reverse_process_features(self, s):
        '''
        Reverse the process_features function, from 1d seqquences (B,L) to tensor representation (B,H,W,C).
        Used to feed decoder after restoration.
        '''
        s = s[:,self.reverse_path]
        s = np.reshape(s, (-1, *self.latent_dims))
        s = np.flip(s, self.flip) if self.flip is not None else s
        return np.rot90(s, self.rotation, axes=(2,1,))


class LossesAndMetricsSavingCallbackVQVAE(tf.keras.callbacks.Callback):
    def __init__(self, saving_dir, losses_name, model_name, loss_folder_name="losses"):
        super().__init__()
        self.saving_dir = saving_dir  # root for saving
        self.model_name = model_name
        self.loss_saving_dir = os.path.join(saving_dir, loss_folder_name)  # where graphes and arrays of losses will be stored
        self.nb_batch = 0
        self.losses_name = losses_name
        self.train_losses_batch_mean = {loss_name: [] for loss_name in losses_name}  # dictionary of lists of training losses (batch mean)
        self.val_losses_epoch_mean = {loss_name: [0] for loss_name in losses_name}  # dictionary of lists of testing losses (epoch mean)

    def on_train_batch_end(self, batch, logs=None):
        # Retrieve the training loss after each batch :
        for loss_name in self.losses_name:
            self.train_losses_batch_mean[loss_name].append(logs[loss_name])

        for metric in self.model.metrics:
            self.train_metrics_batch_mean[metric.name].append(metric.result())

        if hasattr(self.model, 'epoch_start'):
            if self.model.vq.current_epoch < self.model.vq.last_epoch_train:
                self.model.vq.current_iter += 1

    def on_test_batch_end(self, batch, logs=None):
        # Retrieve the testing loss after each batch :
        for loss_name in self.losses_name:
            self.val_losses_epoch_mean[loss_name][-1] += logs[loss_name]
        for metric in self.model.metrics:
            self.val_metrics_epoch_mean[metric.name][-1] += metric.result()
        self.nb_batch += 1

    def on_epoch_begin(self, epoch, logs=None):
        # For the following epoch to aggregate valid losses :
        for loss_name in self.losses_name:
            self.val_losses_epoch_mean[loss_name].append(0)
        for metric in self.model.metrics:
            self.val_metrics_epoch_mean[metric.name].append(0)
        self.nb_batch = 0
        if hasattr(self.model.vq, "current_epoch"):
            self.model.vq.current_epoch = epoch + 1

    def on_epoch_end(self, epoch, logs=None):

        for loss_name in self.losses_name:
            print(f'\nThe training {loss_name} for epoch {epoch + 1} was {"{:.4e}".format(self.train_losses_batch_mean[loss_name][0])} at the beginning'
                  f' and {"{:.4e}".format(self.train_losses_batch_mean[loss_name][-1])}  at the end'
                  f' with mean\u00B1std of {"{:.4e}".format(np.mean(self.train_losses_batch_mean[loss_name]))}\u00B1{"{:.4e}".format(np.std(self.train_losses_batch_mean[loss_name]))}')
            print(f'The validation {loss_name} for epoch {epoch + 1} is {"{:.4e}".format(self.val_losses_epoch_mean[loss_name][-1])}')

        # Saving model :
        os.makedirs(os.path.join(self.saving_dir, self.model_name + "_epoch_" + str(epoch + 1)), exist_ok=True)
        self.model.save_weights(os.path.join(self.saving_dir, self.model_name + "_epoch_" + str(epoch + 1), "weights"))

    def on_test_begin(self, logs=None):
        print(f'\nEvaluation begins\n')
        if not hasattr(self, "val_metrics_epoch_mean"):
            self.train_metrics_batch_mean = {metric.name: [] for metric in self.model.metrics}  # dictionary of lists of training metrics (batch mean)
            self.val_metrics_epoch_mean = {metric.name: [0] for metric in self.model.metrics}  # dictionary of lists of testing metrics (epoch mean)

    def on_test_end(self, logs=None):
        print(f'\nEvaluation finished\n')
        for loss_name in self.losses_name:
            self.val_losses_epoch_mean[loss_name][-1] /= self.nb_batch  # val_loss is averaged over batches (size b), we want averaged over samples (size n), n = m * b with m = nb_batches
        for metric in self.model.metrics:
            self.val_metrics_epoch_mean[metric.name][-1] = metric.result()

    def on_train_begin(self, logs=None):
        print(f'\nTraining begins\n')
        if not hasattr(self, "val_metrics_epoch_mean"):
            self.train_metrics_batch_mean = {metric.name: [] for metric in self.model.metrics}  # dictionary of lists of training metrics (batch mean)
            self.val_metrics_epoch_mean = {metric.name: [0] for metric in self.model.metrics}  # dictionary of lists of testing metrics (epoch mean)

    def on_train_end(self, logs=None):
        print(f'\nTraining finished, saving losses in {self.loss_saving_dir}\n')
        os.makedirs(self.loss_saving_dir, exist_ok=True)
        # Saving npy arrays :
        for loss_name in self.losses_name:
            np.save(os.path.join(self.loss_saving_dir, f"{loss_name}_batch_mean_over_batches"), self.train_losses_batch_mean[loss_name])
            np.save(os.path.join(self.loss_saving_dir, f"val_{loss_name}_batch_mean_over_epochs"), self.val_losses_epoch_mean[loss_name])

        # determining best epoch according to the validation of total loss :  #TODO this is already implemented in keras
        epoch_min_total_loss = np.argmin(self.val_losses_epoch_mean["total_loss"][1:]) + 1  # 0 will be the first eval loss
        print("Epoch of minimal total validation loss was epoch " + str(epoch_min_total_loss))

        matplotlib.use('Agg')  # non-interactive backend
        losses_tuple = [(loss_name, self.train_losses_batch_mean[loss_name], self.val_losses_epoch_mean[loss_name]) for loss_name in self.losses_name]

        # Plotting each loss individually :
        for loss_tuple in losses_tuple:
            name_loss = loss_tuple[0]
            train_loss = loss_tuple[1]
            valid_loss = loss_tuple[2]
            iters_batch = np.linspace(0, len(train_loss), len(train_loss))  # start, stop, nb samples
            iters_epoch = np.linspace(0, len(train_loss), len(valid_loss))
            nb_iters_per_epoch = len(iters_batch) / (len(iters_epoch) - 1)  # -1 for the initial validation

            def iter_to_epoch(i):
                e = (i - 1) // np.round(nb_iters_per_epoch)
                return e + 0.81  # weird constant to align the second axis

            def epoch_to_iter(e):
                i = (e - 1) * np.round(nb_iters_per_epoch)
                return i + 1

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8), dpi=200)
            ax.plot(iters_batch, train_loss, "-b", label="Training")
            ax.plot(iters_epoch, valid_loss, "og", label="Validation")
            ax.axvline(x=iters_epoch[epoch_min_total_loss], color="red", label="min total loss")
            ax.legend()
            ax.set_xlabel("iterations (over batches of data)")
            secax = ax.secondary_xaxis('top', functions=(iter_to_epoch, epoch_to_iter))  # secondary x axis
            secax.set_xlabel('epochs')
            fig.suptitle(name_loss + " over batch iterations and epochs")
            plt.savefig(os.path.join(self.loss_saving_dir, f"{name_loss}.png"))
            plt.close(fig=fig)

        # Plotting every loss on the same graph :
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8), dpi=200)
        for loss_tuple in losses_tuple:
            ax.plot(iters_batch, loss_tuple[1], "-", label=loss_tuple[0] + " train")
            ax.plot(iters_epoch, loss_tuple[2], "o", label=loss_tuple[0] + " valid")
        ax.set_xlabel("iterations (over batches of data)")
        secax = ax.secondary_xaxis('top', functions=(iter_to_epoch, epoch_to_iter))
        secax.set_xlabel('epochs')
        ax.legend()
        fig.suptitle("Losses")
        plt.savefig(os.path.join(self.loss_saving_dir, "losses.png"))
        plt.close()

        # Plotting each metric individually (only perplexity for now)
        for metric in self.model.metrics:
            train_metric = self.train_metrics_batch_mean[metric.name]
            valid_metric = self.val_metrics_epoch_mean[metric.name]
            iters_batch = np.linspace(0, len(train_metric), len(train_metric))  # start, stop, nb samples
            iters_epoch = np.linspace(0, len(train_metric), len(valid_metric))
            nb_iters_per_epoch = len(iters_batch) / (len(iters_epoch) - 1)  # -1 for the initial validation

            def iter_to_epoch(i):
                e = (i - 1) // np.round(nb_iters_per_epoch)
                return e + 0.81  # weird constant to align the second axis

            def epoch_to_iter(e):
                i = (e - 1) * np.round(nb_iters_per_epoch)
                return i + 1

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8), dpi=200)
            ax.plot(iters_batch, train_metric, "-b", label="Training")
            ax.plot(iters_epoch, valid_metric, "og", label="Validation")
            ax.axvline(x=iters_epoch[epoch_min_total_loss], color="red", label="min total metric")
            ax.legend()
            ax.set_xlabel("iterations (over batches of data)")
            secax = ax.secondary_xaxis('top', functions=(iter_to_epoch, epoch_to_iter))  # secondary x axis
            secax.set_xlabel('epochs')
            fig.suptitle(metric.name + " over batch iterations and epochs")
            plt.savefig(os.path.join(self.loss_saving_dir, f"{metric.name}.png"))
            plt.close(fig=fig)

        # Saving best models :
        shutil.rmtree(os.path.join(self.saving_dir, self.model_name + "_best"), ignore_errors=True)  # in case best already exists
        shutil.copytree(os.path.join(self.saving_dir, self.model_name + "_epoch_" + str(epoch_min_total_loss)),
                        os.path.join(self.saving_dir, self.model_name + "_best"))


class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, training_mode="loss", decay=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.training_mode = training_mode
        self.decay = decay
        self.epsilon = 1e-5

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=self.training_mode == "loss",
            name="embeddings",
        )

        if training_mode == "ema":
            self.ema_cluster_size = ExponentialMovingAverage(decay=self.decay)
            self.ema_cluster_size.initialize(tf.zeros([self.num_embeddings], dtype=tf.float32))
            self.ema_dw = ExponentialMovingAverage(decay=self.decay)
            self.ema_dw.initialize(self.embeddings)

    def call(self, x, training=False):
        # Quantization.
        encoding_indices = self.get_code_indices(x)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Calculate vector quantization loss and add that to the layer.
        if self.training_mode == "loss":
            codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
            self.add_loss(codebook_loss)
        elif self.training_mode == "ema" and training:
            updated_ema_cluster_size = self.ema_cluster_size(tf.reduce_sum(encodings, axis=0))
            dw = tf.matmul(x, encodings, transpose_a=True)
            updated_ema_dw = self.ema_dw(dw)
            n = tf.reduce_sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n)
            normalised_updated_ema_w = (updated_ema_dw / tf.reshape(updated_ema_cluster_size, [1, -1]))

            self.embeddings.assign(normalised_updated_ema_w)
            self.add_loss(0.)  # for compatibility

        else:
            self.add_loss(0.)  # for compatibility
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)

        self.add_loss(commitment_loss)
        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)

        return encodings, quantized

    def update_embeddings(self, new_embeddings):
        self.embeddings.assign(new_embeddings)

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
                tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
                + tf.reduce_sum(self.embeddings ** 2, axis=0)
                - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

    def _set_decay(self, decay):
        self.decay = decay
        if self.training_mode == "ema":
            self.ema_cluster_size.update_decay(decay)
            self.ema_dw.update_decay(decay)


class Reservoir(tf.keras.layers.Layer):
    def __init__(self, max__samples=1024):
        super(Reservoir, self).__init__()
        self.trainable = False
        self.n = max__samples
        self.i = 0
        self.buffer = None

    def reset(self):
        self.i = 0
        self.buffer = None

    def add(self, samples):
        if self.buffer is None:
            self.buffer = np.empty((self.n, samples.shape[-1]))

        if self.i < self.n:
            slots = self.n - self.i
            add_samples = samples[:slots]
            self.buffer[self.i:self.i + len(add_samples)] = add_samples
            self.i += len(add_samples)
            samples = samples[slots:]
            if len(samples) == 0:
                return

        if np.random.random() < 0.01:
            print('\nUpdating reservoir...')
            for s in samples:
                ind = np.random.randint(0, self.i)
                self.i += 1

                if ind < len(self.buffer):
                    self.buffer[ind] = s

    def content(self):
        return self.buffer[:self.i]


class BottleneckVQ(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, codebook_learning="ema", reservoir_size=1024, reestimate_step_iter=1024,
                 epoch_start=2, num_epoch_train=5, name="vq"):
        super(BottleneckVQ, self).__init__(name=name)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.reestrimate_step_tier = reestimate_step_iter
        self.epoch_start = epoch_start
        self.last_epoch_train = epoch_start + num_epoch_train
        self.reservoir_size = reservoir_size

        self.current_iter = 0
        self.current_epoch = 0

        assert codebook_learning in ("kmeans", "loss", "ema")
        if codebook_learning == "kmeans":
            self.reservoir = Reservoir(max__samples=self.reservoir_size)
            self.vector_quantizer = VectorQuantizer(self.num_embeddings, self.embedding_dim, training_mode="kmeans")
        elif codebook_learning == "loss":
            self.reservoir = None
            self.vector_quantizer = VectorQuantizer(self.num_embeddings, self.embedding_dim, training_mode="loss")
        elif codebook_learning == "ema":
            self.reservoir = None
            self.vector_quantizer = VectorQuantizer(self.num_embeddings, self.embedding_dim, training_mode="ema")

    def reestimate(self):
        if self.reservoir is None or self.current_epoch < self.epoch_start:
            return

        if self.current_epoch >= self.last_epoch_train:
            self.reservoir = None
            return

        samples_reservoir = self.reservoir.content()
        if samples_reservoir is None:
            return
        else:
            if samples_reservoir.shape[0] < self.num_embeddings or self.current_iter < self.reestrimate_step_tier:
                return
            encodings, *_ = k_means(samples_reservoir, n_clusters=self.num_embeddings)
            print('\nUpdating codebook vectors...')
            self.vector_quantizer.embeddings.assign(np.transpose(encodings))
            self.reservoir.reset()
            self.current_iter = 0

    def call(self, batch, training=False):
        self.reestimate()

        input_shape = tf.shape(batch)
        flattened = tf.reshape(batch, [-1, self.embedding_dim])

        if self.reservoir is not None:
            if training:
                self.reservoir.add(flattened)

        if training and self.reservoir is not None and self.current_epoch < self.epoch_start:
            encodings = np.zeros((input_shape))
            quantized = batch

        encodings, quantized = self.vector_quantizer(flattened, training=training)

        out_vq = tf.reshape(quantized, input_shape)
        return encodings, out_vq

    def _get_latent_indexes_volume(self, batch):
        '''
        Get indexes of closest encodings vectors in the codebook with the shape (B,h,w).
        Batch is a batch of latent representation
        '''
        input_shape = tf.shape(batch)
        flattened = tf.reshape(batch, [-1, self.embedding_dim])
        encodings_indexes = self.vector_quantizer.get_code_indices(flattened)
        print(encodings_indexes.shape)
        print(tf.reshape(encodings_indexes, input_shape[:-1]))
        return tf.reshape(encodings_indexes, input_shape[:-1])



class ResidualBlock(Layer):
    def __init__(self, filters=256, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=filters, kernel_size=(1, 1), activation='relu')

    def call(self, inputs, *args, **kwargs):
        return inputs + self.conv2(self.conv1(inputs))


class RandomBrightness(tf.keras.layers.Layer):
    def __init__(self, factor, value_range):
        super().__init__()
        self.trainable = False
        self.factor = sorted(factor)
        self.value_range = sorted(value_range)

    def call(self, batch, training=False):
        if training:
            br = tf.random.uniform(shape=[batch.shape[0], ] + [1, ] * len(batch.shape[1:]), minval=self.factor[0], maxval=self.factor[1], dtype=batch.dtype)
            batch += tf.tile(br, (1,) + batch.shape[1:])
            return tf.clip_by_value(batch, self.value_range[0], self.value_range[1])
        else:
            return batch


class ExponentialMovingAverage:
    def __init__(self, decay) -> None:
        self._decay = decay
        self._counter = tf.Variable(0, trainable=False, dtype=tf.int32, name="counter")
        self._hidden = None
        self.average = None
        self.ema = None

    def __call__(self, value):
        self.update(value)
        return self.value

    def update(self, value):
        # self._counter.assign_add(1)
        value = tf.convert_to_tensor(value)
        # counter = tf.cast(self._counter, value.dtype)
        self.ema.assign_sub((self.ema - value) * (1 - self._decay))
        # self._hidden.assign_sub((self._hidden - value) * (1 - self._decay))
        # self.average.assign((self._hidden / (1 - tf.pow(self._decay, counter))))

    @property
    def value(self):
        return self.ema.read_value()

    def reset(self):
        self.ema.assign(tf.zeros_like(self.ema))
        # self._counter.assign(tf.zeros_like(self._counter))
        # self._hidden.assign(tf.zeros_like(self._hidden))
        # self.average.assign(tf.zeros_like(self.average))

    def initialize(self, value):
        self.ema = tf.Variable(tf.zeros_like(value), trainable=False, name="ema")
        # self._hidden = tf.Variable(tf.zeros_like(value), trainable=False, name="hidden")
        # self.average = tf.Variable(tf.zeros_like(value), trainable=False, name="average")

    def update_decay(self, value):
        self._decay = value