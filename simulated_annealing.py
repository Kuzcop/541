import random as rd
from numpy import exp
from cnn_helper import cnn_default_hyperparameters, cnn_get_random_neighbouring_solution, cnn_objective, file_name
import time

def simulated_annealing(initial_solution, neighbour_gen_fun, objective_fun, iterations, c, seed=-1, show=False):
    # Initialization phase
    if seed >= 0:
        rd.seed = seed
    current_solution = initial_solution
    current_eval = objective_fun(current_solution, show)
    count = 0
    # Optimization phase
    i = 0
    while i < iterations:
        next_solution = neighbour_gen_fun(current_solution, rd)
        next_eval = objective_fun(next_solution)
        if next_eval > current_eval or current_eval == 0 or _accept_worse_solution(c, i+1, current_eval, next_eval):
            current_solution = next_solution
            current_eval = next_eval
            count = 0
            i += 1
            if show:
                print("\n", "#", "Accepted new solution")
        elif (next_eval < current_eval) and next_eval != -1:
            count +=1
            i += 1
        
        if count == -1:
            break
    return current_solution, current_eval, i


def _accept_worse_solution(c, interval, curr_val, next_val):
    return rd.random() < exp(interval * (next_val - curr_val) / (c * curr_val))

start_time = time.time()
res, res_eval, iterations_completed = simulated_annealing(cnn_default_hyperparameters, cnn_get_random_neighbouring_solution, cnn_objective, 500, 1, show=True)
end_time = time.time()
with open(file_name, 'a+') as f:
    print('-' * 50, file=f)
    print(end_time-start_time, file=f)
    print(res, file=f)
    print(res_eval, file=f)
print('#' * 50, "Performed iterations: {}".format(iterations_completed),
      "Best found Solution: {} , Objvalue: {}, Search Time: {}".format(res, res_eval, end_time-start_time), sep="\n")

