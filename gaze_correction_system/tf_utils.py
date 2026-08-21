import tensorflow as tf
from tensorflow import keras


class BatchNorm(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.bn = keras.layers.BatchNormalization(
            momentum=0.9,
            epsilon=1e-5,
            center=True,
            scale=True
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        return self.bn(inputs, training=training)


class CNNBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            activation=None,
            use_bias=False
        )
        self.bn = BatchNorm()
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = keras.activations.relu(x)
        x = self.bn(x, training=training)
        return x


class DNNBlock(keras.layers.Layer):
    def __init__(self, nodes, **kwargs):
        super().__init__(**kwargs)
        self.nodes = nodes

    def build(self, input_shape):
        self.dense = keras.layers.Dense(
            units=self.nodes,
            activation=None
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        x = keras.activations.relu(x)
        return x
