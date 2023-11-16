import random as rd
from numpy import exp

def simulated_annealing(iterations, c, seed=2012, show=False):
    # Initialization phase
    rd.seed = seed
    current_solution = hyperparameters
    current_eval = objective(current_solution, show)
    # Optimization phase
    for i in range(iterations):
        next_solution = get_random_neighbouring_solution(current_solution, rd)
        next_eval = objective(next_solution, show)
        if next_eval > current_eval or _accept_worse_solution(c, i+1, current_eval, next_eval):
            current_solution = next_solution
            current_eval = next_eval
            if show:
                print("\n", "#", "Accepted new solution")
    return current_solution, current_eval

def _accept_worse_solution(c, interval, curr_val, next_val):
    return rd.random() < exp(interval * (next_val - curr_val)/ (c * curr_val))


res, res_eval = simulated_annealing(100, 1, show=True)
print('#' * 50, "Performed iterations: {}".format(100),
      "Best found Solution: {} , Objvalue: {}".format(res, res_eval), sep="\n")

