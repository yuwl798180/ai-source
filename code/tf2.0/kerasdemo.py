import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

model = Sequential()
model.add(Dense(50, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
print(model.summary())

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

x_train = np.random.randn(1000, 100)
y_train = np.random.randint(10, size=(1000, 1))
y_train = keras.utils.to_categorical(y_train)
x_test = np.random.randn(200, 100)
y_test = np.random.randint(10, size=(200, 1))
y_test = keras.utils.to_categorical(y_test)

history = model.fit(
    x=x_train,
    y=y_train,
    validation_split=0.2,
    batch_size=4,
    epochs=10,
    verbose=1,
)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.evaluate(
    x_test,
    y_test,
    batch_size=32,
    verbose=1,
)

classes = model.predict(x_test, batch_size=32, verbose=1)
print(classes.shape)
