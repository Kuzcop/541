import ast
import tensorflow as tf
from predictor import Predictor
from copy import deepcopy

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

training_data, training_labels, test_data, test_labels = get_predictor_data()


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


def get_predictor_data(filename):
    # TODO
    with open(filename, 'r') as f:
        lines = f.readlines()

    hps = []
    accuracies =[]
    for line in lines:
        line = line.strip()
        result = ast.literal_eval(line)
        hps.append(result['HP'])
        accuracies.append(result['Accuracy'])