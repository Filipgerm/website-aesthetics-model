# The architecture that will be used for the CNN is CaffeNet. First, we define the LRN (Local Response Normalization) layer.
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer
from keras import backend as K


class LRN(Layer):

    def __init__(self, n=5, alpha=0.0001, beta=0.75, k=2, **kwargs):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LRN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(LRN, self).build(input_shape)

    def call(self, x, mask=None):
        if K.image_data_format() == "channels_first":
            _, f, r, c = self.shape
        else:
            _, r, c, f = self.shape
        half_n = self.n // 2
        squared = tf.square(x)
        pooled = tf.nn.pool(squared, window_shape=(half_n, half_n), pooling_type='AVG', strides=(1, 1),
                         padding="SAME")
        if K.image_data_format() == "channels_first":
            summed = tf.reduce_sum(pooled, axis=1, keepdims=True)
            averaged = (self.alpha / self.n) * tf.repeat(summed, f, axis=1)
        else:
            summed = tf.reduce_sum(pooled, axis=3, keepdims=True)
            averaged = (self.alpha / self.n) * tf.repeat(summed, f, axis=3)
        denom = tf.pow(self.k + averaged, self.beta)
        return x / denom

    def get_output_shape_for(self, input_shape):
        return input_shape
