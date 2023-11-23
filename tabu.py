import random as rd
from copy import deepcopy
from cnn_helper import cnn_default_hyperparameters, cnn_get_random_neighbouring_solution, cnn_objective


class TS:
    def __init__(self, initial_solution, neighbour_gen_fun, objective_fun, tabu_length, seed=-1):
        if (seed >= 0):
            rd.seed(seed)

        self.tabu_list = []
        self.tabu_conditions = []
        self.diversification = False
        self.num_neighbours = 5
        self.tabu_length = tabu_length
        self.Initial_solution = initial_solution
        self.neighbour_gen_fun = neighbour_gen_fun
        self.objective_fun = objective_fun

        self.Best_solution, self.Best_objvalue = self.TSearch()

    def get_neighbours_and_evaluate(self, solution):
        neighbours = []
        diffs = []
        for i in range(self.num_neighbours):
            while True:
                # is_tabu = False
                neighbour, diff = self.neighbour_gen_fun(solution, rd, True)
                while neighbour in neighbours:
                    neighbour, diff = self.neighbour_gen_fun(solution, rd, True)

                if neighbour not in self.tabu_list:
                    if (diff in self.tabu_conditions) and (not self.diversification):
                        print('Tabu Solution')
                        continue
                    else:
                        break

            neighbours.append(neighbour)
            diffs.append(diff)

        assert len(neighbours) == self.num_neighbours
        best_neighbour = {}
        best_accuracy = 0
        for neighbour, diff in zip(neighbours, diffs):
            val = self.objective_fun(neighbour)
            if val > best_accuracy:
                best_accuracy = val
                best_neighbour = deepcopy(neighbour)
            elif val < 0.1:
                self.tabu_conditions.append(diff)

        return best_neighbour, best_accuracy, neighbours

    def TSearch(self):
        '''The implementation Tabu search algorithm with short-term memory and pair_swap as Tabu attribute.
        '''
        # Parameters:
        tenure =self.tabu_length
        best_solution = self.Initial_solution
        best_objvalue = self.objective_fun(best_solution)
        current_objvalue = 0

        print("#"*30, "Short-term memory TS with Tabu Tenure: {}\nInitial Solution: {}, Initial Objvalue: {}".format(tenure, best_solution, best_objvalue), "#"*30, sep='\n\n')
        Terminate = 0
        count = 0
        while Terminate < 100:
            print('\n\n### Iteration: {}###  Tabu List {}, Diversification {}, Current_Objvalue: {}, Best_Objvalue: {}'.format(Terminate, len(self.tabu_list), self.diversification, current_objvalue, best_objvalue))
            current_solution, current_objvalue, neighbours = self.get_neighbours_and_evaluate(best_solution)

            self.tabu_list = self.tabu_list + neighbours
            while len(self.tabu_list) > self.tabu_length:
                del self.tabu_list[0]
            
            while len(self.tabu_conditions) > 5:
                del self.tabu_conditions[0]

            if current_objvalue > best_objvalue:
                best_objvalue = current_objvalue
                best_solution = current_solution
                count = 0
                self.diversification = False
                self.tabu_length = tenure
                Terminate += 1
            elif (current_objvalue < best_objvalue) and (current_objvalue != -1):
                count += 1
                Terminate += 1
            if count == 10:
                self.diversification = True
                self.tabu_length = tenure*2
            if count == 20:
                break

        print('#'*50, "Performed iterations: {}".format(Terminate), "Best found Solution: {} , Objvalue: {}".format(best_solution,best_objvalue), sep="\n")
        return best_solution, best_objvalue


test = TS(cnn_default_hyperparameters, cnn_get_random_neighbouring_solution, cnn_objective, tabu_length = 100)

print(test.Best_objvalue, test.Best_solution)
