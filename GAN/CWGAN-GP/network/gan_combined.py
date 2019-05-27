from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from base.base_model import BaseModel
from utils.losses import wgan_loss


class GanCombined(BaseModel):
    def define_model(self, generator, discriminator, model_name):
        z = Input(shape=(self.config.data.latent_dim+self.config.data.label_len,))
        label = Input(shape=(self.config.data.label_len,))
        # z = Input(shape=(self.config.data.latent_dim,))
        img_gen = generator(z)
        val = discriminator([img_gen, label])

        return Model(inputs=[z, label], outputs=val, name=model_name)

    def build_model(self, generator, discriminator, model_name):
        discriminator.trainable = False
        combined = self.define_model(generator, discriminator, model_name)
        parallel_combined = self.multi_gpu_model(combined)

        optimizer = Adam(
            self.config.model.generator.lr,
            self.config.model.generator.beta1,
            self.config.model.generator.beta2,
            clipvalue=self.config.model.generator.clipvalue,
            clipnorm=self.config.model.generator.clipnorm)

        parallel_combined.compile(
            optimizer=optimizer,
            loss=wgan_loss
        )

        return combined, parallel_combined
