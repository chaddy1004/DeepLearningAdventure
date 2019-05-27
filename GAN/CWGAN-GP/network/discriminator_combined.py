from functools import partial

from keras import backend as K
from keras.layers import Input
from keras.layers.merge import _Merge
from keras.models import Model
from keras.optimizers import Adam

from base.base_model import BaseModel
from utils.losses import _gp_loss, wgan_loss


class BlendRealandFake(_Merge):

    def build(self, input_shape):
        return super().build(input_shape)

    def _merge_function(self, inputs):
        weights = K.random_uniform_variable((1, 1, 1), low=0, high=1)
        return (weights * inputs[0]) + ((1 - weights) * inputs[-1])

    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(input_shape)


class DiscriminatorCombined(BaseModel):
    def define_model(self, discriminator, model_name):
        input_fakes = Input(shape=self.config.data.img_shape)
        input_reals = Input(shape=self.config.data.img_shape)
        label = Input(shape=(self.config.data.label_len,))

        fake_vals = discriminator([input_fakes, label])
        real_vals = discriminator([input_reals, label])

        dummy_vals = real_vals
        return Model(inputs=[input_fakes, input_reals, label], outputs=[fake_vals, real_vals, dummy_vals],
                     name=model_name)

    def gp_loss(self, blended_sample=None, blended_sample_pred=None):
        gp_loss_func = partial(_gp_loss, blended_sample=blended_sample, blended_sample_pred=blended_sample_pred)
        gp_loss_func.__name__ = "gradient_penalty"
        return gp_loss_func

    def build_model(self, discriminator, model_name):
        model = self.define_model(discriminator=discriminator, model_name=model_name)
        parallel_model = self.multi_gpu_model(model)
        input_fakes = model.inputs[0]
        input_reals = model.inputs[1]
        labels = model.inputs[2]
        input_blends = BlendRealandFake()([input_reals, input_fakes])
        gp_loss = self.gp_loss(blended_sample=input_blends, blended_sample_pred=discriminator([input_blends, labels]))

        optim = Adam(
            lr=self.config.model.discriminator.lr,
            beta_1=self.config.model.discriminator.beta1,
            beta_2=self.config.model.discriminator.beta2,
            clipvalue=self.config.model.discriminator.clipvalue,
            clipnorm=self.config.model.discriminator.clipnorm)

        parallel_model.compile(optimizer=optim,
                               loss=[wgan_loss,
                                     wgan_loss,
                                     gp_loss],
                               loss_weights=[self.config.model.discriminator.adv_weight,
                                             self.config.model.discriminator.adv_weight,
                                             self.config.model.discriminator.gradient_penalty])
        return model, parallel_model
