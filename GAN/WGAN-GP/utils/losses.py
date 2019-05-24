import numpy as np
from keras import backend as K

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
    L2_grad = K.sqrt(grad_squared_summed)
    penalty = K.square((L2_grad - 1))
    return K.mean(penalty)
