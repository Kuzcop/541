import tensorflow as tf
import time

from tensorflow.keras import datasets, layers, models
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

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


def train(hyperparameters, show_summary = False):
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

    # ------------------------Example-------------------------
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(10))

    if show_summary:
        model.summary()

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10, batch_size = 32, 
                        validation_data=(test_images, test_labels))

    start_time = time.time()
    test_loss, test_acc = model.evaluate(test_images,  test_labels)
    end_time = time.time()
    test_time = end_time - start_time
    return test_acc, test_time


##########################COPY ABOVE#####################################

########################## TESTING ######################################

if __name__ == '__main__':
    conv = {
        'activation' : '', # https://www.tensorflow.org/api_docs/python/tf/keras/activations
        'kernel_size':  0, # any int
        'padding'    : '', # 'valid' or 'same'
        'strides'    :  0, # int
        'filters'    :  0  # int
    }

    pool = {
        'pool_size': 0 , # int
        'strides'  : 0 , # int
        'padding'  : '', # 'valid' or 'same'
    }

    # When editing hyperparameters in metaheurisitcs, we know how to change each hyperparameter in each layer via the key name (activation, pool_size, etc.)
    hyperparameters = {
        'conv_1': 0,
        'pool_1': 0,
        'conv_2': 0,
        'pool_2': 0,
        'conv_3': 0,
    }

    hyperparameters['conv_1'] = {
        'activation' : 'relu', # https://www.tensorflow.org/api_docs/python/tf/keras/activations
        'kernel_size':  3, # any int
        'padding'    : 'valid', # 'valid' or 'same'
        'filters'    :  32  # int
    }

    hyperparameters['conv_2'] = {
        'activation' : 'relu', # https://www.tensorflow.org/api_docs/python/tf/keras/activations
        'kernel_size':  3, # any int
        'padding'    : 'valid', # 'valid' or 'same'
        'filters'    :  64  # int
    }

    hyperparameters['conv_3'] = {
        'activation' : 'relu', # https://www.tensorflow.org/api_docs/python/tf/keras/activations
        'kernel_size':  3, # any int
        'padding'    : 'valid', # 'valid' or 'same'
        'filters'    :  64  # int
    }

    hyperparameters['pool_1'] = {
        'pool_size': 2 , # int
        'strides'  : 2 , # int
        'padding'  : 'valid', # 'valid' or 'same'
    }

    hyperparameters['pool_2'] = {
        'pool_size': 2 , # int
        'strides'  : 2 , # int
        'padding'  : 'valid', # 'valid' or 'same'
    }

    train(hyperparameters, True)