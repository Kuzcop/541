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

    x = {'HP': {'conv_1': {'activation': 'tanh', 'kernel_size': 3, 'padding': 'valid', 'filters': 15},
                'pool_1': {'pool_size': 1, 'strides': 1, 'padding': 'same'},
                'conv_2': {'activation': 'relu', 'kernel_size': 4, 'padding': 'same', 'filters': 56},
                'pool_2': {'pool_size': 2, 'strides': 3, 'padding': 'valid'},
                'conv_3': {'activation': 'softmax', 'kernel_size': 4, 'padding': 'valid', 'filters': 40}},
         'Accuracy': 0.6883999705314636, 'Latency: ': 1.8673241138458252}
