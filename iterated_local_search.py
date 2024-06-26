import random as rd
from copy import deepcopy
from cnn_helper import cnn_default_hyperparameters, cnn_get_random_neighbouring_solution, cnn_objective, cnn_preturb, file_name
from numpy import exp
import time

class ILS:
    def __init__(self, initial_solution, neighbour_gen_fun, objective_fun, preturb_fun, ILS_length, c, seed=-1):
        if (seed >= 0):
            rd.seed(seed)

        
        self.ILS_list = []
        self.c = c
        self.ILS_conditions = []
        self.diversification = False
        self.num_neighbours = 5
        self.ILS_length = ILS_length
        self.Initial_solution = initial_solution
        self.neighbour_gen_fun = neighbour_gen_fun
        self.preturb_fun = preturb_fun
        self.objective_fun = objective_fun

        self.Best_solution, self.Best_objvalue, self.iter = self.ILSearch()

    def _accept_worse_solution(self, interval, curr_val, next_val):
        return rd.random() < exp(interval * (next_val - curr_val) / (self.c * curr_val))

    def get_neighbours_and_evaluate(self, solution):
        neighbours = []
        diffs = []
        for i in range(self.num_neighbours):
            while True:
                neighbour, diff = self.neighbour_gen_fun(solution, rd, True)
                while neighbour in neighbours:
                    neighbour, diff = self.neighbour_gen_fun(solution, rd, True)
                
                if neighbour not in self.ILS_list:
                    if (diff in self.ILS_conditions) and (not self.diversification):
                        print('Tabu Solution')
                        continue
                    else:
                        break
            neighbours.append(neighbour)
            diffs.append(diff)

        assert len(neighbours) == self.num_neighbours
        best_neighbour = {}
        best_accuracy = [-10]
        for neighbour, diff in zip(neighbours, diffs):
            val = self.objective_fun(neighbour)
            if val[0] > best_accuracy[0]:
                best_accuracy = val
                best_neighbour = deepcopy(neighbour)
            elif val[0] < 0.1:
                self.ILS_conditions.append(diff)

        return best_neighbour, best_accuracy, neighbours

    def ILSearch(self):
        '''The implementation ILS search algorithm with short-term memory
        '''
        # Parameters:
        tenure =self.ILS_length
        best_solution = self.Initial_solution
        best_objvalue = self.objective_fun(best_solution)[0]
        current_objvalue = 0
        starting_point = self.Initial_solution

        print("#"*30, "Short-term memory ILS with Short Term Memory: {}\nInitial Solution: {}, Initial Objvalue: {}".format(tenure, best_solution, best_objvalue), "#"*30, sep='\n\n')
        Terminate = 0
        count = 0
        while Terminate < 100:
            print('\n\n### Iteration: {}###  Memory {}, Diversification {}, Current_Objvalue: {}, Best_Objvalue: {}'.format(Terminate, len(self.ILS_list), self.diversification, current_objvalue, best_objvalue))
            current_solution, ex, neighbours = self.get_neighbours_and_evaluate(starting_point)

            current_objvalue = ex[0]
            acc = ex[1]
            lat = ex[2]


            self.ILS_list = self.ILS_list + neighbours
            while len(self.ILS_list) > self.ILS_length:
                del self.ILS_list[0]
            
            while len(self.ILS_conditions) > 5:
                del self.ILS_conditions[0]

            if (current_objvalue > best_objvalue) or (self._accept_worse_solution(Terminate+1, best_objvalue, current_objvalue)):
                best_objvalue = current_objvalue
                best_solution = current_solution
                count = 0
                self.diversification = False
                self.ILS_length = tenure
                Terminate += 1
                print("\n", "#", "Accepted new solution")
                with open(file_name, 'a+') as f:
                    result = {'HP': best_solution, 'Accuracy': acc, 'Latency': lat, 'Obj': best_objvalue, 'iter': Terminate}
                    print(result, file=f)
            elif (current_objvalue < best_objvalue) and (current_objvalue != -1):
                count += 1
                Terminate += 1
            if count == 10:
                self.diversification = True
                self.ILS_length = tenure*2
            if count == 20:
                break

            starting_point = self.preturb_fun(starting_point, rd)

        print('#'*50, "Performed iterations: {}".format(Terminate), "Best found Solution: {} , Objvalue: {}".format(best_solution,best_objvalue), sep="\n")
        return best_solution, best_objvalue, Terminate

seed = 0
start_time = time.time()
test = ILS(cnn_default_hyperparameters, cnn_get_random_neighbouring_solution, cnn_objective, cnn_preturb, ILS_length = 100, seed=seed, c = 1)
end_time = time.time()

with open(file_name, 'a+') as f:
    print('-' * 50, file=f)
    print(end_time-start_time, file=f)
    print(test.Best_solution, file=f)
    print(test.Best_objvalue, file=f)
    print(test.iter, file=f)
    print(seed, file=f)

print(test.Best_objvalue, test.Best_solution, end_time-start_time)
