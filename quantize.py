import tensorflow as tf

from larq import utils
from larq.layers_base import (
    QuantizerBase
)

@utils.register_keras_custom_object
class QuantLSTM(QuantizerBase, tf.keras.layers.LSTM):
     def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.)/2., 0, 1)
    
def make_clips(a, n):
    temp = []
    for i in range(n):
        m = 2*i + 1
        clip = (m*a) / (2*n - 1)
        temp.append(clip)
    return temp

def make_quantize(x: tf.Tensor, clip_value: list):
    mask = tf.less(tf.math.abs(x), (clip_value[0] + clip_value[1]) / 2)
    temp = tf.where(mask, tf.math.sign(x)*clip_value[0], x)
    for i in range(1, len(clip_value[:-1])):
        upper = tf.greater_equal(tf.math.abs(temp), (clip_value[i] + clip_value[i-1]) / 2)
        lower = tf.less(tf.math.abs(temp), (clip_value[i] + clip_value[i+1]) / 2)
        mask = tf.math.logical_and(upper, lower)
        temp = tf.where(mask, tf.math.sign(temp)*clip_value[i], temp)
    mask = tf.greater_equal(tf.math.abs(temp), (clip_value[-2] + clip_value[-1]) / 2)
    temp = tf.where(mask, tf.math.sign(temp)*clip_value[-1], temp)
    return temp
    
def ste_sign(x: tf.Tensor, clip_value: list = [1]) -> tf.Tensor:
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            return _clipped_gradient(x, dy, clip_value[-1])

        if len(clip_value) == 1:
            return math.sign(x)*clip_value[0], grad

        return make_quantize(x, clip_value), grad

    return _call(x)