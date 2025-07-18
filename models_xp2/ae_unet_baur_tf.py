import tensorflow as tf


class UnetBaur(tf.keras.models.Model):
    def __init__(self, out_chan=2, filters=(64,128,256,512,64), preprocessing=True):
        super().__init__()
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.preprocessing=preprocessing
        if preprocessing:
            self.preprocess_layer = tf.keras.Sequential([tf.keras.layers.RandomTranslation(height_factor=(-0.1,0.1), width_factor=[-0.1,0.1], fill_mode="constant"),
                                                        RandomBrightness(factor= (-0.1,0.1), value_range=(0., 1.)),
                                                        tf.keras.layers.RandomContrast(factor=0.1)])

        self.encoder_blocks = []
        for i, n_filter in enumerate(filters):
            self.encoder_blocks.append(tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=n_filter, kernel_size=(5,5), strides=(2,2) if i!=0 else (1,1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ]))

        self.decoder_blocks = []
        self.decoder_blocks.append(tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=filters[-2], kernel_size=(5,5), strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ]))
        self.decoder_blocks.append(tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=filters[-3], kernel_size=(5,5), strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ]))
        self.concat1 = tf.keras.layers.Concatenate()
        self.decoder_blocks.append(tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=filters[-4], kernel_size=(5,5), strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ]))
        self.concat2 = tf.keras.layers.Concatenate()
        self.decoder_blocks.append(tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=filters[-5], kernel_size=(5,5), strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ]))
        self.final_conv = tf.keras.layers.Conv2D(filters=out_chan, kernel_size=(1,1), strides=(1,1), activation='tanh')


    def call(self, x):
        x = self.encoder_blocks[0](x)
        s1 = self.encoder_blocks[1](x)
        s2 = self.encoder_blocks[2](s1)
        x = self.encoder_blocks[3](s2)
        x = self.encoder_blocks[4](x)
        x = self.decoder_blocks[0](x)
        x = self.concat1([self.decoder_blocks[1](x), s2])
        x = self.concat2([self.decoder_blocks[2](x), s1])
        x = self.decoder_blocks[3](x)
        x = self.final_conv(x)
        return x

    def train_step(self, batch):
        if self.preprocessing:
            batch = self.preprocess_layer(batch)
        with tf.GradientTape() as tape:
            pred = self(batch)
            loss = self.mse_loss(batch,pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {'loss':loss}

    def test_step(self, batch):
        pred = self(batch)
        loss = self.mse_loss(batch,pred)
        return {'loss':loss}


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