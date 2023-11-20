import predictor_helper
from predictor_helper import *
from simulated_annealing import simulated_annealing


def optimize_predictor(optimize_fun):
    predictor_helper.predictor_data_filename = "meta_results/2023-11-18_23-43-37.txt"  #TODO: Specify
    best_results = {}

    for n_layers in predictor_n_layer_set:
        for loss_fun in predictor_loss_set:
            model, eval = optimize_fun(n_layers, loss_fun)
            print("#### LAYERS: {}, LOSS: {}: MODEL: {}, EVAL: {} ####\n".format(n_layers, loss_fun, model, eval))
            best_results[n_layers, loss_fun] = {
                "model": model,
                "eval": eval,
            }


def _predictor_sa_optimization(n_layers, loss_fun):
    c = 1
    iterations_per_conf = 50
    seed = -1
    return simulated_annealing(get_default_predictor(n_layers, loss_fun),
                               predictor_get_random_neighbouring_solution,
                               predictor_objective,
                               iterations_per_conf,
                               c,
                               seed)


# Run Simulated Annealing Optimization for predictor
optimize_predictor(_predictor_sa_optimization)
