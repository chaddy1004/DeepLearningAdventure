import numpy as np
import tensorflow as tf
from keras import Model, losses
from keras import backend as K
from keras.applications import VGG19
from keras.applications import keras_modules_injection
from keras_applications.vgg19 import preprocess_input


@keras_modules_injection
def preprocess_vgg_input(image_tensor, **kwargs):
    """Gets an image tensor valued between -1 and 1 and outputs an vgg preprocessed tensor
    :param image_tensor: value should be between [-1, 1]:
    :return: tensor valued between [-VGG_BGR_MEAN, 255-VGG_BGR_MEAN]
    """
    if len(K.int_shape(image_tensor)) < 4:  # (batch, H, W):
        image_tensor = K.expand_dims(image_tensor, axis=-1)
    denormed_y = image_tensor * 127.5 + 127.5
    # 1ch to 3ch
    processed_y = tf.image.grayscale_to_rgb(denormed_y)
    # preprocess
    processed_y = preprocess_input(processed_y, **kwargs)

    return processed_y


def perceptual_loss(y_true, y_pred):
    """
    Perceptual loss with VGG19 network; single image version. Prime example is from SRGAN(https://arxiv.org/pdf/1609.04802.pdf)
    """
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    loss_model = Model(inputs=vgg.inputs, outputs=vgg.get_layer('block5_conv4').output)
    loss_model.trainable = False
    preprocessed_y_pred = preprocess_vgg_input(y_pred)
    preprocessed_y_true = preprocess_vgg_input(y_true)
    return losses.mse(loss_model(preprocessed_y_true), loss_model(preprocessed_y_pred))


def image_gradient_loss(y_true, y_pred):
    """
    Loss based on image gradient
    """
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    return losses.mse(dy_true, dy_pred) + losses.mae(dx_true, dx_pred)


def wgan_loss(y_true, y_pred):
    """
    y_true should be [-1,1] and y_pred is the value that discriminator outputs after a linear activation
    The [-1, 1] is there to reproduce the WGAN value function using Kantorovich-Rubinstein duality
    where it is E(D(x_real)) - E(D(x_fake))

    :param y_true: Correct value that discriminator should have. -1 for fake, 1 for real
    :param y_pred: Value predicted by the discriminator
    :return: The expectation of discriminator outputs
    """
    return K.mean(y_true * y_pred)


def _gp_loss(y_true, y_pred, blended_sample, blended_sample_pred):
    """
    y_true and y_pred are ignored since GP is not calculated based on prediction and ground truth, but rather the
    gradient of D(x_blended)=blended_sample_pred w.r.t. blended_sample
    :param y_true: ignored. Dummy variable used in trainer
    :param y_pred: ignored.
    :param blended_sample: Blended image tensor from ground truth and predicted
    :param blended_sample_pred: The result of blended_sample fed into the discriminator
    :return:
    """
    grad = K.gradients(blended_sample_pred, blended_sample)[0]
    grad_squared = K.square(grad)
    grad_squared_summed = K.sum(grad_squared, axis=np.arange(1, len(K.int_shape(blended_sample))))
    L2_grad = K.sqrt(grad_squared_summed + 1e-7)  # epsilon added to prevent value inside sqrt being negative
    penalty = K.square((L2_grad - 1))
    return K.mean(penalty)
