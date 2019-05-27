from keras.layers import Activation
from keras.layers import Dense, Input, LeakyReLU, BatchNormalization, Conv2D, \
    Flatten
from keras.models import Model

from base.base_model import BaseModel


class Discriminator(BaseModel):
    def define_model(self, model_name):
        image = Input(shape=self.config.data.img_shape)
        x = Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(image)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)
        x = Flatten()(x)

        x_adv = Dense(128)(x)
        x_adv = LeakyReLU(alpha=0.2)(x_adv)
        output_adv = Dense(1)(x_adv)

        x_aux = Dense(128)(x)
        x_aux = LeakyReLU(alpha=0.2)(x_aux)
        x_aux = Dense(64)(x_aux)
        x_aux = LeakyReLU(alpha=0.2)(x_aux)
        x_aux = Dense(self.config.data.label_len + 1)(x_aux)
        output_aux = Activation('softmax')(x_aux)

        model = Model(inputs=image, outputs=[output_adv, output_aux], name=model_name)
        return model

    def build_model(self, model_name):
        raise NotImplementedError
