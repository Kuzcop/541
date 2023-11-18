from tensorflow.keras import layers, models
import time

import matplotlib.pyplot as plt
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

class Predictor:
    def __init__(self, n_layers, keras_loss_fun, hyperparameters, show_summary=False):
        # Hyperparameters:
        # List of dictionaries (one for each layer)
        model = models.Sequential()
        # Create input and hidden layers
        for i in range(n_layers):
            model.add(layers.Dense(hyperparameters[i]['size'], activation=hyperparameters[i]['activation']))

        # Create output layer (accuracy, test_latency)
        model.add(layers.Dense(2))  # No activation function, since it is a regression problem

        model.compile(optimizer='adam',
                      loss=keras_loss_fun,
                      metrics=['accuracy'])

        if show_summary:
            model.summary()

        self.model = model

    def train(self, train_data, train_labels, test_data, test_labels):

        history = self.model.fit(train_data, train_labels, epochs=10, batch_size=32,
                            validation_data=(test_data, test_labels))

        start_time = time.time()
        test_loss, test_acc = self.model.evaluate(test_data, test_labels)
        end_time = time.time()
        test_time = end_time - start_time
        return test_acc, test_time

        return test_acc, test_time
