from keras.layers import Dense, Input, LeakyReLU, Conv2DTranspose, BatchNormalization, Activation, Reshape, Conv2D
from keras.models import Model

from base.base_model import BaseModel


class Generator(BaseModel):
    def define_model(self, model_name):
        z = Input(shape=(self.config.data.latent_dim+self.config.data.label_len,))
        # z = Input(shape=(self.config.data.latent_dim,))
        x = Dense((7 * 7 * 128))(z)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Reshape((7, 7, 128))(x)
        x = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same")(x)
        x = Conv2D(filters=1, kernel_size=1, strides=1, padding="same")(x)
        img = Activation("tanh")(x)
        model = Model(z, img, name=model_name)
        return model

    def build_model(self, **kargs):
        raise NotImplementedError
