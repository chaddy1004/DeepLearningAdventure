import keras.backend as K
import tensorflow as tf
from keras import initializers
from keras.engine import Layer
from keras.layers import (Activation, Reshape)
from keras.layers import Conv2D
from keras.layers.merge import _Merge


class BlendRealandFake(_Merge):
    """
    Layer to blend real and fake sample when applying gradient penalty method introduced in WGAN-GP paper
    https://arxiv.org/pdf/1704.00028.pdf
    """
    def build(self, input_shape):
        return super().build(input_shape)

    def _merge_function(self, inputs):
        weights = K.random_uniform_variable((1, 1, 1), low=0, high=1)
        return ((weights * inputs[0]) + ((1 - weights) * inputs[-1]))

    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(input_shape)


class LayerNorm(Layer):
    def call(self, x, **kwargs):
        return tf.contrib.layers.layer_norm(x)


class PixelShuffler(Layer):
    """
    PixelShuffler from SRGAN
    https://arxiv.org/pdf/1707.02937.pdf
    """

    def __init__(self, **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.scale = 2

    def build(self, input_shape):
        super(PixelShuffler, self).build(input_shape)

    def call(self, x, **kwargs):
        shape = K.int_shape(x)
        f = int(int(shape[3]) / (self.scale ** 2))

        bsize, a, b, c = shape
        bsize = tf.shape(x)[0]

        x_s = tf.split(x, self.scale, 3)
        x_r = tf.concat(x_s, 2)
        return tf.reshape(x_r, (bsize, self.scale * a, self.scale * b, f))

    def compute_output_shape(self, input_shape):
        dims = [input_shape[0],
                input_shape[1] * self.scale,
                input_shape[2] * self.scale,
                int(input_shape[3] / (
                        self.scale ** 2))]
        output_shape = tuple(dims)
        return output_shape


class HWFlatten(Layer):
    """
    Flattening layer used for attention mechanism
    Flattens the height and width dimension but keeps the channel
    """

    def call(self, x, **kwargs):
        shape = K.int_shape(x)
        b_size = tf.shape(x)[0]
        # print("shape", shape)
        return tf.reshape(x, (b_size, shape[1] * shape[2], shape[-1]))

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[1]
                        * input_shape[2], input_shape[3])
        return output_shape


class MatMul(Layer):
    """
    Matrix multiplication layer used for attention mechanism.
    :param outer: Boolean to indicate whether to perform outer product or not
    """

    def __init__(self, outer=True, **kwargs):
        super(MatMul, self).__init__(**kwargs)
        self.outer = outer

    def compute_output_shape(self, input_shape):
        if self.outer:
            return (input_shape[0][0], input_shape[0][1], input_shape[1][1])
        else:
            return (input_shape[0][0], input_shape[0][1], input_shape[1][2])

    def call(self, tensor_inputs, **kwargs):
        matA, matB = tensor_inputs
        if self.outer:
            return tf.matmul(matA, matB, transpose_b=True)
        else:
            return tf.matmul(matA, matB)


def attention_block(kv, q):
    shape = K.int_shape(kv)
    q = ConvSN2D(filters=shape[3] // 8, kernel_size=1, strides=1, padding="same")(q)
    q = HWFlatten()(q)
    k = ConvSN2D(filters=shape[3] // 8, kernel_size=1,
                 strides=1, padding="same")(kv)
    k = HWFlatten()(k)
    v = ConvSN2D(filters=shape[3], kernel_size=1,
                 strides=1, padding="same")(kv)
    v = HWFlatten()(v)
    qkt = MatMul(outer=True)([q, k])
    qkt = Activation("softmax")(qkt)
    output = MatMul(outer=False)([qkt, v])
    output = Reshape((shape[1], shape[2], shape[3]))(output)
    return output


class ConvSN2D(Conv2D):
    """
    Credit goes to IShengang for the original implementation that I based this off of.
    https://github.com/IShengFang/SpectralNormalizationKeras
    """

    def build(self, input_shape):
        super(ConvSN2D, self).build(input_shape)
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)

    def call(self, inputs, training=None):
        def _l2norm(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)

        def power_iter(W, u):
            # performing the power iteration once as per paper
            _u = u
            _v = _l2norm(K.dot(_u, K.transpose(W)))
            _u = _l2norm(K.dot(_v, W))
            return _u, _v

        # Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        # Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iter(W_reshaped, self.u)
        # Calculate Sigma (Singular value)
        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        # Performing spectral normalization
        W_bar = W_reshaped / sigma
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)

        outputs = K.conv2d(
            inputs,
            W_bar,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
