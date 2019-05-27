from keras.layers import Dense, Input, LeakyReLU, Conv2D, \
    Flatten, RepeatVector, Reshape, Concatenate
from keras.models import Model

from base.base_model import BaseModel


class Discriminator(BaseModel):
    def define_model(self, model_name):
        image = Input(shape=self.config.data.img_shape)
        label = Input(shape=(self.config.data.label_len,))
        enlarged_label = RepeatVector(self.config.data.img_size ** 2)(label)
        enlarged_label = Reshape((self.config.data.img_size, self.config.data.img_size, -1))(enlarged_label)
        x = Concatenate()([image, enlarged_label])
        x = Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        # x = BatchNormalization()(x)
        x = Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        # x = BatchNormalization()(x)
        x = Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(0.2)(x)
        # x = BatchNormalization()(x)
        x = Conv2D(filters=128, kernel_size=1, strides=1, padding="same")(x)
        x = LeakyReLU(0.2)(x)
        # x = BatchNormalization()(x)
        x = Flatten()(x)

        x_adv = Dense(128)(x)
        x_adv = LeakyReLU(alpha=0.2)(x_adv)
        output_adv = Dense(1)(x_adv)

        model = Model(inputs=[image, label], outputs=output_adv, name=model_name)
        return model

    def build_model(self, model_name):
        raise NotImplementedError
