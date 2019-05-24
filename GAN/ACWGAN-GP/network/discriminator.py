from keras.layers import Dense, Input, LeakyReLU, BatchNormalization, Activation, Conv2D, \
    Flatten
from keras.models import Model

from base.base_model import BaseModel


class Discriminator(BaseModel):
    def define_model(self, model_name):
        image = Input(shape=self.config.data.img_shape)
        x = Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(image)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1)(x)
        model = Model(inputs=image, outputs=x, name=model_name)
        return model

    def build_model(self, model_name):
        raise NotImplementedError
