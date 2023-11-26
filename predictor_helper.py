import ast
import tensorflow as tf
from predictor import Predictor
from copy import deepcopy
import random



show = False
predictor_data_filename = "INVALID FILENAME: SPECIFY BEFORE TRAINING PREDICTOR"


# TRAIN AND TEST DATA
def _get_predictor_data(filename, show):
    hyperparameters, outputs = _parse_predictor_data_file(filename)
    hyperparameters, outputs = _preprocess_predictor_data(hyperparameters, outputs, show)

    dataset_size = len(hyperparameters)
    train_percentage = 0.7
    train_size = round(dataset_size * train_percentage)

    dataset_indexes = range(0, dataset_size)
    train_sample_indexes = random.sample(dataset_indexes, train_size)
    test_sample_indexes = [index for index in dataset_indexes if index not in train_sample_indexes]

    train_data, train_labels = [(hyperparameters[i], outputs[i]) for i in train_sample_indexes]
    test_data, test_labels = [(hyperparameters[i], outputs[i]) for i in test_sample_indexes]

    return train_data, train_labels, test_data, test_labels


activation_mappings = {
    'linear': 0,
    'relu': 1,
    'sigmoid': 5,
    'tanh': 6,
    'softmax': 8,
    'exponential': 10,
}

padding_mappings = {
    'valid': 0,
    'same': 1,
}


def _preprocess_predictor_data(inputs, outputs, show):
    processed_inputs = []
    for model in inputs:
        processed_model = []
        for (layer_type, params) in model:  # Iterate over all layers
            if 'conv' in layer_type:
                processed_model.append(activation_mappings[params['activation']])
                processed_model.append(padding_mappings[params['padding']])
                processed_model.append(params['kernel_size'])
                processed_model.append(params['strides'])
                processed_model.append(params['filters'] / 10)  #  Make it around the same size as the others
                processed_model.append(params['pool_size'])
            else:
                processed_model.append(padding_mappings[params['padding']])
                processed_model.append(params['strides'])
                processed_model.append(params['pool_size'])
        if show:
            print("\nModel = {}".format(model))
            print("\n--> Processed model = {}\n".format(processed_model))
        processed_inputs.append(processed_model)
    return processed_inputs, outputs


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


training_data, training_labels, testing_data, testing_labels = _get_predictor_data(predictor_data_filename, show_data)

predictor_loss_set = [
    'mean_squared_error',
    'mean_absolute_error',
    tf.keras.losses.Huber(),
    tf.keras.losses.LogCosh()
]

predictor_n_layer_set = range(1, 10)

predictor_layer_hp_set = {
    'activation': [
        'relu',
        'selu',
        'tanh',
        'linear',
        'swish',
    ],
    'size': [20, 40, 60, 80, 100]
}


def get_predictor_layer_default_hyperparameters():
    return {
        'activation': random.sample(predictor_layer_hp_set['activation'], 1)[0],
        'size': random.sample(predictor_layer_hp_set['size'], 1)[0]
    }


def get_default_predictor(n_layers, loss):
    return {
        "keras_loss_fun": loss,
        "n_layers": n_layers,
        "hyperparameters": [get_predictor_layer_default_hyperparameters() for _ in range(0, n_layers)]
    }


def predictor_objective(params):
    n_layers = params["n_layers"]
    keras_loss_fun = params["keras_loss_fun"]
    hyperparameters = params["hyperparameters"]
    predictor = Predictor(n_layers, keras_loss_fun, hyperparameters)

    test_acc, test_time = predictor.train(training_data, training_labels, testing_data, testing_labels)

    obj_value = test_acc**2 / test_time
    if show:
        print("\n", "-" * 8, "test_accuracy: {} #### test_latency: {} #### objective: {}"
                    .format(test_acc,test_time, obj_value), "-" * 8)

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
