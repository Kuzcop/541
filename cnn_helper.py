from copy import deepcopy
from datetime import datetime
from cnn import CNN
from tensorflow.keras import datasets

cnn_activation_set = [
    'relu',
    'sigmoid',
    'tanh',
    'linear',
    'exponential',
    'softmax'
]

cnn_hp_set = {
    'activation' : cnn_activation_set,
    'kernel_size': list(range(2,6)),
    'padding'    : ['valid', 'same'],
    'strides'    : list(range(1,4)),
    'filters'    : [20, 40, 60, 80, 100],
    'pool_size'  : list(range(1,5))
}

# When editing hyperparameters in metaheurisitcs, we know how to change each hyperparameter in each layer via the key name (activation, pool_size, etc.)
cnn_default_hyperparameters = {
    'conv_1': 0,
    'pool_1': 0,
    'conv_2': 0,
    'pool_2': 0,
    'conv_3': 0,
}

cnn_default_hyperparameters['conv_1'] = {
    'activation' : 'relu', # https://www.tensorflow.org/api_docs/python/tf/keras/activations
    'kernel_size':  3, # any int
    'padding'    : 'valid', # 'valid' or 'same'
    'filters'    :  32  # int
}

cnn_default_hyperparameters['conv_2'] = {
    'activation' : 'relu', # https://www.tensorflow.org/api_docs/python/tf/keras/activations
    'kernel_size':  3, # any int
    'padding'    : 'valid', # 'valid' or 'same'
    'filters'    :  64  # int
}

cnn_default_hyperparameters['conv_3'] = {
    'activation' : 'relu', # https://www.tensorflow.org/api_docs/python/tf/keras/activations
    'kernel_size':  3, # any int
    'padding'    : 'valid', # 'valid' or 'same'
    'filters'    :  64  # int
}

cnn_default_hyperparameters['pool_1'] = {
    'pool_size': 2 , # int
    'strides'  : 2 , # int
    'padding'  : 'valid', # 'valid' or 'same'
}

cnn_default_hyperparameters['pool_2'] = {
    'pool_size': 2 , # int
    'strides'  : 2 , # int
    'padding'  : 'valid', # 'valid' or 'same'
}


(cnn_train_images, cnn_train_labels), (cnn_test_images, cnn_test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
cnn_train_images, cnn_test_images = cnn_train_images / 255.0, cnn_test_images / 255.0


def cnn_objective(params, show=False):
    try:
        cnn = CNN(params)
        test_acc, test_time = cnn.train(cnn_train_images, cnn_train_labels, cnn_test_images, cnn_test_labels)
        obj_value = test_acc*test_acc / test_time
        if show:
            print("\n", "-" * 8, "model = {} #### accuracy = {} #### test_latency = {} #### objective = {}".format(params, test_acc, test_time, obj_value), "-" * 8)

    except Exception as e:
        print(e)
        print("Bad Params: {}".format(params))
        obj_value = -1
        test_acc = -1
        test_time = -1
    return [obj_value, test_acc, test_time]


def cnn_get_random_neighbouring_solution(solution, rd, is_TS = False):
    neighbour = deepcopy(solution)
    layers_to_change = rd.sample(list(neighbour.keys()), k=3)
    diff = {}

    for layer in layers_to_change:
        if 'conv' in layer:
            hps = rd.sample(list(neighbour[layer].keys()), k=2)
            diff[layer] = hps
            for hp in hps:
                neighbour[layer][hp] = rd.sample(cnn_hp_set[hp], k=1)[0]
        else:
            hps = rd.sample(list(neighbour[layer].keys()), k=1)
            diff[layer] = hps
            for hp in hps:
                neighbour[layer][hp] = rd.sample(cnn_hp_set[hp], k=1)[0]
    if is_TS:
        return neighbour, diff
    return neighbour

def cnn_preturb(solution, rd):
    neighbour = deepcopy(solution)
    layers_to_change = rd.sample(list(neighbour.keys()), k=len(solution))

    for layer in layers_to_change:
        if 'conv' in layer:
            hps = rd.sample(list(neighbour[layer].keys()), k=3)
            for hp in hps:
                neighbour[layer][hp] = rd.sample(cnn_hp_set[hp], k=1)[0]
        else:
            hps = rd.sample(list(neighbour[layer].keys()), k=2)
            for hp in hps:
                neighbour[layer][hp] = rd.sample(cnn_hp_set[hp], k=1)[0]
    return neighbour


def _get_log_file_path():
    # get current date and time
    file_name = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    return "meta_results/" + file_name + '.txt'

file_name = _get_log_file_path()


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
