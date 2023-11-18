import ast
import tensorflow as tf
from predictor import Predictor

predictor_loss_set = [
    'mean_squared_error',
    'mean_absolute_error',
    tf.keras.losses.Huber(),
    tf.keras.losses.LogCosh()
]

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

predictor_layer_hyperparameters = {
    'activation': predictor_activation_set[0],
    'size': 50
}


def predictor_objective(n_layers, keras_loss_fun, hyperparameters):
    predictor = Predictor(n_layers, keras_loss_fun, hyperparameters)

    training_data, training_labels, test_data, test_labels = get_predictor_data()
    test_acc, test_time = predictor.train(training_data, training_labels, test_data, test_labels)

    obj_value = test_acc / test_time

    params = {
        "n_layers": n_layers,
        "loss": keras_loss_fun,
        "hyperparameters": hyperparameters
    }
    print("\n", "-" * 8, "model: {} #### test_accuracy: {} #### test_latency: {} #### objective: {}".format(params, test_acc, test_time, obj_value), "-" * 8)

    return obj_value


def predictor_get_random_neighbouring_solution():
    #TODO
    return 0


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