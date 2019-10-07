import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 1

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def show_image(num):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[num + i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[num + i]])
    plt.show()


# show_image(random.randint(0,1000))

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=EPOCHS)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)


def plt_image(predictions_array, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predict_label = np.argmax(predictions_array)
    if predict_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel('{} {:2.2f}% ({})'.format(class_names[predict_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label],
                                         color=color))


def plt_value_array(predictions_array, true_label):
    plt.grid(False)
    plt.xticks(range(10), class_names, rotation=270)
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def show_one_prediction(num):

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt_image(predictions[num], test_labels[num], test_images[num])

    plt.subplot(1, 2, 2)
    plt_value_array(predictions[num], test_labels[num])
    plt.show()


def show_predictions(row, col):
    num_images = row * col
    sample = random.sample(range(10000), num_images)

    plt.figure(figsize=(4 * col, 2 * row))
    for i in range(num_images):
        plt.subplot(row, 2 * col, 2 * i + 1)
        plt_image(predictions[sample[i]], test_labels[sample[i]], test_images[sample[i]])
        plt.subplot(row, 2 * col, 2 * i + 2)
        plt_value_array(predictions[sample[i]], test_labels[sample[i]])
    plt.tight_layout()
    plt.show()


while 1:
    inputs = input('\ninput number for show prediction result:...')
    row, col = inputs.strip().split(',')

    # show_one_prediction(idx)
    show_predictions(int(row), int(col))
