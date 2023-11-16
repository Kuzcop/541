import pandas as pd
import random as rd
from itertools import combinations
import math
from copy import deepcopy
from cnn import train
from helper import hyperparameters, hp_set

class TS():
    def __init__(self, seed, tabu_length):
        self.seed = seed
        self.tabu_list = []
        self.num_neighbours = 5
        self.tabu_length = tabu_length
        self.Initial_solution = self.get_InitialSolution()
        self.Best_solution, self.Best_objvalue = self.TSearch()

    def get_InitialSolution(self, show=False):
        initial_solution = hyperparameters
        rd.seed(self.seed)
        
        if show == True:
            print("Initial Random Solution: {}".format(initial_solution))
        return initial_solution

    def Objfun(self, solution, show = False):
        objfun_value = train(solution, False)
        if show == True:
            print("\n","#"*8, "The Objective function value for {} is: {}".format(solution ,objfun_value),"#"*8)
        return objfun_value

    def get_neighbours_and_evaluate(self, solution):     

        neighbours = []
        for i in range(self.num_neighbours):
            while True:
                is_tabu = False
                neighbour = deepcopy(solution)
                layers_to_change = rd.sample(list(neighbour.keys()), k = 3)

                for layer in layers_to_change:
                    if 'conv' in layer:
                        hps = rd.sample(list(neighbour[layer].keys()), k = 2)
                        for hp in hps:
                            neighbour[layer][hp] = rd.sample(hp_set[hp], k = 1)[0]
                    else:
                        hps = rd.sample(list(neighbour[layer].keys()), k = 1)
                        for hp in hps:
                            neighbour[layer][hp] = rd.sample(hp_set[hp], k = 1)[0]
                
                for sol in self.tabu_list:
                    if neighbour == sol:
                        is_tabu = True
                        break
                if not is_tabu:
                    break
            neighbours.append(neighbour)

        assert len(neighbours) == self.num_neighbours
        best_neighbour = {}
        best_accuracy = 0
        for neighbour in neighbours:
            val = self.Objfun(neighbour)
            if val > best_accuracy:
                best_accuracy = val
                best_neighbour = deepcopy(neighbour)

        return best_neighbour, best_accuracy, neighbours

    def TSearch(self):
        '''The implementation Tabu search algorithm with short-term memory and pair_swap as Tabu attribute.
        '''
        # Parameters:
        tenure =self.tabu_length
        best_solution = self.Initial_solution
        best_objvalue = self.Objfun(best_solution)
        current_solution = {}
        current_objvalue = 0

        print("#"*30, "Short-term memory TS with Tabu Tenure: {}\nInitial Solution: {}, Initial Objvalue: {}".format(tenure, best_solution, best_objvalue), "#"*30, sep='\n\n')
        Terminate = 0
        while Terminate < 100:
            print('\n\n### iter {}###  Current_Objvalue: {}, Best_Objvalue: {}'.format(iter, current_objvalue,
                                                                                    best_objvalue))
            # Searching the whole neighborhood of the current solution:
            current_solution, current_objvalue, neighbours = self.get_neighbours_and_evaluate(best_solution)
            if len(self.tabu_list) == self.tabu_length:
                del self.tabu_list[:5]
            self.tabu_list = self.tabu_list + neighbours

            if current_objvalue > best_objvalue:
                best_objvalue = current_objvalue
                best_solution = current_solution

            Terminate += 1
            
            
        print('#'*50 , "Performed iterations: {}".format(Terminate), "Best found Solution: {} , Objvalue: {}".format(best_solution,best_objvalue), sep="\n")
        return best_solution, best_objvalue


test = TS(seed = 2012, tabu_length = 30)
