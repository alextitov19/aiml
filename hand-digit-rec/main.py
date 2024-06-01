import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3)

# model.save('hand_digit_rec.model.keras')

model = tf.keras.models.load_model('hand-digit-rec/hand_digit_rec.model.keras')

# loss, acc = model.evaluate(x_test, y_test)

# print(f'Loss: {loss}, Accuracy: {acc}')

image = cv2.imread('hand-digit-rec/dig_3.png')[:,:,0]
image = np.invert(np.array([image]))
preditction = model.predict(image)
print(f'Prediction: {np.argmax(preditction)}')