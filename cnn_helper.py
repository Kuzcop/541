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
    'filters'    : list(range(10, 101)),
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
        obj_value = test_acc / test_time
        if show:
            print("\n", "-" * 8, "model = {} #### accuracy = {} #### test_latency = {} #### objective = {}".format(params, test_acc, test_time, obj_value), "-" * 8)

        with open(_get_log_file_path(), 'a+') as f:
            result = {'HP': params, 'Accuracy': obj_value}
            print(result, file=f)
    except ValueError:
        print("Bad Params: {}".format(params))
        obj_value = -1
    return obj_value


def cnn_get_random_neighbouring_solution(solution, rd):
    neighbour = deepcopy(solution)
    layers_to_change = rd.sample(list(neighbour.keys()), k=3)

    for layer in layers_to_change:
        if 'conv' in layer:
            hps = rd.sample(list(neighbour[layer].keys()), k=2)
            for hp in hps:
                neighbour[layer][hp] = rd.sample(cnn_hp_set[hp], k=1)[0]
        else:
            hps = rd.sample(list(neighbour[layer].keys()), k=1)
            for hp in hps:
                neighbour[layer][hp] = rd.sample(cnn_hp_set[hp], k=1)[0]
    return neighbour


def _get_log_file_path():
    # get current date and time
    file_name = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    return "meta_results/" + file_name + '.txt'




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
