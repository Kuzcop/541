import tensorflow as tf
import time

from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from copy import deepcopy

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
#   try:
#     tf.config.set_visible_devices(gpus[0], 'GPU')
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#   except RuntimeError as e:
#     # Visible devices must be set before GPUs have been initialized
#     print(e)


class CNN:
    def __init__(self, hyperparameters, show_summary=False):
        model = models.Sequential()
        activation, kernel_size, padding, filters = hyperparameters['conv_1'].values()
        model.add(layers.Conv2D(filters = filters, kernel_size = kernel_size, activation=activation, padding = padding, input_shape=(32, 32, 3)))

        pool_size, strides, padding = hyperparameters['pool_1'].values()
        model.add(layers.MaxPooling2D(pool_size, strides, padding))

        activation, kernel_size, padding, filters = hyperparameters['conv_2'].values()
        model.add(layers.Conv2D(filters = filters, kernel_size = kernel_size, activation=activation, padding = padding))

        pool_size, strides, padding = hyperparameters['pool_2'].values()
        model.add(layers.MaxPooling2D(pool_size, strides, padding))

        activation, kernel_size, padding, filters = hyperparameters['conv_3'].values()
        model.add(layers.Conv2D(filters = filters, kernel_size = kernel_size, activation=activation, padding = padding))
        model.add(layers.Flatten())
        model.add(layers.Dense(filters, activation='relu'))
        model.add(layers.Dense(10))

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        if show_summary:
            model.summary()

        self.model = model
        # ------------------------Example-------------------------
        # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        # model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # model.add(layers.Flatten())
        # model.add(layers.Dense(64, activation='relu'))
        # model.add(layers.Dense(10))

    def train(self, train_images, train_labels, test_images, test_labels):

        history = self.model.fit(train_images, train_labels, epochs=10, batch_size = 32,
                            validation_data=(test_images, test_labels))

        start_time = time.time()
        test_loss, test_acc = self.model.evaluate(test_images,  test_labels)
        end_time = time.time()
        test_time = end_time - start_time
        return test_acc, test_time
