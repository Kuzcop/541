import random as rd
from numpy import exp
from helper import cnn_hyperparameters, cnn_get_random_neighbouring_solution, cnn_objective


def simulated_annealing(initial_solution, neighbour_gen_fun, objective_fun, iterations, c, seed=-1, show=False):
    # Initialization phase
    if (seed >= 0):
        rd.seed = seed
    current_solution = initial_solution
    current_eval = objective_fun(current_solution, show)
    # Optimization phase
    for i in range(iterations):
        next_solution = neighbour_gen_fun(current_solution, rd)
        next_eval = objective_fun(next_solution, show)
        if next_eval > current_eval or _accept_worse_solution(c, i+1, current_eval, next_eval):
            current_solution = next_solution
            current_eval = next_eval
            if show:
                print("\n", "#", "Accepted new solution")
    return current_solution, current_eval


def _accept_worse_solution(c, interval, curr_val, next_val):
    return rd.random() < exp(interval * (next_val - curr_val)/ (c * curr_val))


res, res_eval = simulated_annealing(cnn_hyperparameters, cnn_get_random_neighbouring_solution, cnn_objective, 100, 1, show=True)
print('#' * 50, "Performed iterations: {}".format(100),
      "Best found Solution: {} , Objvalue: {}".format(res, res_eval), sep="\n")

