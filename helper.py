import matplotlib.pyplot as plt
from copy import deepcopy
import random
from datetime import datetime
# from cnn import train
import re
import ast

activation_set = [
    'relu',
    'sigmoid',
    'tanh',
    'linear',
    'exponential',
    'softmax'
]

padding_set = [
    'valid',
    'same'
]

kernel_size_set = list(range(2,6))

filters_set = list(range(10, 101))

strides_set = list(range(1,4))

pool_size_set = list(range(1,5))

hp_set = {
    'activation' : activation_set,
    'kernel_size': kernel_size_set,
    'padding'    : padding_set,
    'strides'    : strides_set,
    'filters'    : filters_set,
    'pool_size'  : pool_size_set
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


def objective(params, show=False):
    try:
        test_acc, test_time = train(params, False)
        obj_value = test_acc * test_time
        if show:
            print("\n", "#" * 8, "The Objective function value for {} is: {}".format(params, obj_value), "#" * 8)
        
        with open(log_dir, 'a+') as f:
            result = {'HP': params, 'Accuracy': obj_value}
            print(result, file=f)
    except:
        print("Bad Params: {}".format(params))
        obj_value = -1
    return obj_value


def get_random_neighbouring_solution(solution, rd):
    neighbour = deepcopy(solution)
    layers_to_change = rd.sample(list(neighbour.keys()), k=3)

    for layer in layers_to_change:
        if 'conv' in layer:
            hps = rd.sample(list(neighbour[layer].keys()), k=2)
            for hp in hps:
                neighbour[layer][hp] = rd.sample(hp_set[hp], k=1)[0]
        else:
            hps = rd.sample(list(neighbour[layer].keys()), k=1)
            for hp in hps:
                neighbour[layer][hp] = rd.sample(hp_set[hp], k=1)[0]
    return neighbour


def get_file_name():
    # get current date and time
    file_name = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    return file_name

log_dir = "meta_results/" + get_file_name() + '.txt'


def get_predictor_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    hps = []
    accuracies =[]
    for line in lines:
        line = line.strip()
        result = ast.literal_eval(line)
        hps.append(result['HP'])
        accuracies.append(result['Accuracy'])
    print(hps)
    print(accuracies)
    return hps, accuracies


if __name__ == '__main__':
    file_name = 'meta_results/2023-11-16_16-40-57.txt'

    get_predictor_data(file_name)
##########################COPY ABOVE#####################################

# random.seed(0)

# layers_to_change = random.sample(list(hyperparameters.keys()), k = 3)

# for layer in layers_to_change:
#     if 'conv' in layer:
#         hps = random.sample(list(hyperparameters[layer].keys()), k = 2)
#         for hp in hps:
#             hyperparameters[layer][hp] = random.sample(hp_set[hp], k = 1)[0]
#     else:
#         hps = random.sample(list(hyperparameters[layer].keys()), k = 1)
#         for hp in hps:
#             hyperparameters[layer][hp] = random.sample(hp_set[hp], k = 1)[0]
