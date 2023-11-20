import ast
import tensorflow as tf
from predictor import Predictor
from copy import deepcopy
import random

# TRAIN AND TEST DATA
def _get_predictor_data(filename):
    hyperparameters, outputs = _parse_predictor_data_file(filename)

    dataset_size = len(hyperparameters)
    train_percentage = 0.7
    train_size = round(dataset_size * train_percentage)

    dataset_indexes = range(0, dataset_size)
    train_sample_indexes = random.sample(dataset_indexes, train_size)
    test_sample_indexes = [index for index in dataset_indexes if index not in train_sample_indexes]

    train_data, train_labels = [(hyperparameters[i], outputs[i]) for i in train_sample_indexes]
    test_data, test_labels = [(hyperparameters[i], outputs[i]) for i in test_sample_indexes]

    return train_data, train_labels, test_data, test_labels


def _parse_predictor_data_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    hyperparameters = []
    outputs = []  # Output  = (accuracy, latency)
    for line in lines:
        line = line.strip()
        result = ast.literal_eval(line)
        hyperparameters.append(result['HP'])
        outputs.append((result['Accuracy'], result['Latency']))

    return hyperparameters, outputs


predictor_data_filename = "INVALID FILENAME: SPECIFY BEFORE TRAINING PREDICTOR"
training_data, training_labels, testing_data, testing_labels = _get_predictor_data(predictor_data_filename)

predictor_loss_set = [
    'mean_squared_error',
    'mean_absolute_error',
    tf.keras.losses.Huber(),
    tf.keras.losses.LogCosh()
]

predictor_n_layer_set = range(1, 10)

predictor_activation_set = [
    'relu',
    'selu',
    'tanh',
    'linear',
    'swish',
]

predictor_layer_hp_set = {
    'activation': predictor_activation_set,
    'size': list(range(10, 100)) #TODO: find good limits
}

predictor_layer_default_hyperparameters = {
    'activation': predictor_activation_set[0],
    'size': 50
}


def get_default_predictor(n_layers, loss):
    return {
        "keras_loss_fun": loss,
        "n_layers": n_layers,
        "hyperparameters": [deepcopy(predictor_layer_default_hyperparameters) for _ in range(0, n_layers)]
    }

def predictor_objective(params):
    n_layers = params["n_layers"]
    keras_loss_fun = params["keras_loss_fun"]
    hyperparameters = params["hyperparameters"]
    predictor = Predictor(n_layers, keras_loss_fun, hyperparameters)

    test_acc, test_time = predictor.train(training_data, training_labels, test_data, test_labels)

    obj_value = test_acc / test_time

    print("\n", "-" * 8, "model: {} #### test_accuracy: {} #### test_latency: {} #### objective: {}".format(params, test_acc, test_time, obj_value), "-" * 8)

    return obj_value

def predictor_get_random_neighbouring_solution(old_solution, rd):
    solution = deepcopy(old_solution)
    n_layers = solution["n_layers"]

    # Change the size of two layers
    change_size_layers = rd.sample(range(0, n_layers), 2)
    for change_size_layer in change_size_layers:
        new_size = rd.sample(predictor_layer_hp_set["size"], 1)[0]
        solution["hyperparameters"][change_size_layer]["size"] = new_size
    # Change activation of one layer
    change_activation_layer = rd.sample(range(0, n_layers), 1)[0]
    new_activation = rd.sample(predictor_layer_hp_set["activation"], 1)[0]
    solution["hyperparameters"][change_activation_layer]["activation"] = new_activation

    return solution
