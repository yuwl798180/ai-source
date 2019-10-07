import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## GradientTape
a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))
a = tf.Variable(a)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as tape:
        # tape.watch(a)  # 等价于 a = tf.Variable(a)
        c = tf.sqrt(tf.square(a) + tf.square(b))
        dc_da = tape.gradient(c, a)  # == a/c
    d2c_da2 = outer_tape.gradient(dc_da, a)
    # print(d2c_da2)  # == b**2/c**3

## model
input_dim = 2
output_dim = 1
learning_rate = 0.01

# This is our weight matrix and bias vector
w = tf.Variable(tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(tf.zeros(shape=(output_dim, )))


def compute_predictions(features):
    return tf.matmul(features, w) + b


def compute_loss(labels, predictions):
    return tf.reduce_mean(tf.square(labels - predictions))


# 添加这个修饰符，变成静态图，速度可以明显提升很多
@tf.function
def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        predictions = compute_predictions(x)
        loss = compute_loss(y, predictions)
        dloss_dw, dloss_db = tape.gradient(loss, [w, b])
    w.assign_sub(learning_rate * dloss_dw)
    b.assign_sub(learning_rate * dloss_db)
    return loss


# Prepare a dataset.
num_samples = 10000
negative_samples = np.random.multivariate_normal(mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples)
positive_samples = np.random.multivariate_normal(mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples)
features = np.vstack((negative_samples, positive_samples)).astype(np.float32)
labels = np.vstack((np.zeros((num_samples, 1), dtype='float32'), np.ones((num_samples, 1), dtype='float32')))
plt.scatter(features[:, 0], features[:, 1], c=labels[:, 0])
plt.show()

# Shuffle the data.
random.Random(1337).shuffle(features)
random.Random(1337).shuffle(labels)

# Create a tf.data.Dataset object for easy batched iteration
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
dataset = dataset.shuffle(buffer_size=1024).batch(256)

# training
t0 = time.time()
for epoch in range(20):
    for step, (x, y) in enumerate(dataset):
        loss = train_on_batch(x, y)
    print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))
t_end = time.time() - t0
print('Time per epoch: %.3f s' % (t_end / 20, ))

# view predictions performance
predictions = compute_predictions(features)
plt.scatter(features[:, 0], features[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
