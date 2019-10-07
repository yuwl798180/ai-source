import os
import tensorflow as tf
from tensorflow.keras.layers import Layer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 1 way
class Linear(Layer):
    """y = w * x + b"""
    def __init__(self, units: int, input_dim: int):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype='float32'), trainable=True)
        self.b = tf.Variable(initial_value=b_init(shape=(units, ), dtype='float32'), trainable=True)
        # self.w = self.add_weight(shape=(input_dim, units),initializer='random_normal')

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# Instantiate our layer.
linear_layer = Linear(units=4, input_dim=2)

y = linear_layer(tf.ones((2, 2)))
print(y.shape)


# 2 way
class Linear(Layer):
    """y = w.x + b"""
    def __init__(self, units: int):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units, ), initializer='random_normal', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# Instantiate our lazy layer.
linear_layer = Linear(4)

# This will also call `build(input_shape)` and create the weights.
y = linear_layer(tf.ones((2, 2)))
print(y.shape)


class Dropout(Layer):
    def __init__(self, rate):
        super(Dropout, self).__init__()
        self.rate = rate

    def call(self, inputs, training: bool = None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs


class MLP(Layer):
    """Simple stack of Linear layers."""
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = Linear(32)
        self.fc2 = Linear(20)
        self.fc3 = Linear(10)
        self.dropout = Dropout(0.5)

    def call(self, inputs, training: bool = None):
        x = tf.nn.relu(self.fc1(inputs))
        x = self.dropout(x, training=training)
        x = tf.nn.relu(self.fc2(x))
        x = self.dropout(x, training=training)
        x = self.fc3(x)
        return x


mlp = MLP()
y_train = mlp(tf.ones(shape=(3, 64)), training=True)
print(y.shape)
for args in mlp.weights:
    print(args.shape)
