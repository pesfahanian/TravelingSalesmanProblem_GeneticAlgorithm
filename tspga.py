# [[0, 10, 15, 20],
#  [10, 0, 35, 25],  
#  [15, 35, 0, 30],
#  [20, 25, 30, 0]]

# [[0, 3, 4, 2, 7],
#  [3, 0, 4, 6, 3],  
#  [4, 4, 0, 5, 8],
#  [2, 6, 5, 0, 6],
#  [7, 3, 8, 6, 0]]

# [[0, 14, 15, 4, 9],
#  [14, 0, 18, 5, 13],  
#  [15, 18, 0, 19, 10],
#  [4, 5, 19, 0, 12],
#  [9, 13, 10, 12, 0]]

# [[0, 12, 29, 22, 13, 24],
#  [12, 0, 19, 3, 25, 6],  
#  [29, 19, 0, 21, 21, 28],
#  [22, 3, 21, 0, 4, 5],
#  [13, 25, 23, 4, 0, 16],
#  [24, 6, 28, 5, 16, 0]]

# [[0,1,9,9,9,9,9,9,9,9],
#  [1,0,1,9,9,9,9,9,9,9],
#  [9,1,0,1,9,9,9,9,9,9],
#  [9,9,1,0,1,9,9,9,9,9],
#  [9,9,9,1,0,1,9,9,9,9],
#  [9,9,9,9,1,0,1,9,9,9],
#  [9,9,9,9,9,1,0,1,9,9],
#  [9,9,9,9,9,9,1,0,1,9],
#  [9,9,9,9,9,9,9,1,0,1],
#  [9,9,9,9,9,9,9,9,1,0]]

import random
import numpy as np
from sklearn.utils import shuffle
import operator

pop_size = 20
generations = 1000
mutation_ratio = 0.02
mutation_probability = 0.25

def sort(T):
  T.sort(key = operator.itemgetter(1))
  return T

adj_matrix = np.array( [[0,1,9,9,9,9,9,9,9,9],
                        [1,0,1,9,9,9,9,9,9,9],
                        [9,1,0,1,9,9,9,9,9,9],
                        [9,9,1,0,1,9,9,9,9,9],
                        [9,9,9,1,0,1,9,9,9,9],
                        [9,9,9,9,1,0,1,9,9,9],
                        [9,9,9,9,9,1,0,1,9,9],
                        [9,9,9,9,9,9,1,0,1,9],
                        [9,9,9,9,9,9,9,1,0,1],
                        [9,9,9,9,9,9,9,9,1,0]])
num_nodes = len(adj_matrix)
print(adj_matrix)
print(adj_matrix.shape)

def initialize():  
  initial_population = []
  while (len(initial_population)!=pop_size): 
    chromosome = []
    nodes = []
    for i in range(0, num_nodes):
      nodes.append(i)
    start_end_node = nodes[random.randint(0, num_nodes-1)]
    chromosome.append(start_end_node)
    nodes.remove(start_end_node)
    nodes = shuffle(nodes)
    for n in nodes:
      chromosome.append(n)
    chromosome.append(start_end_node)  
    count = 0
    for k in range(0, len(chromosome)-1):
      if(adj_matrix[chromosome[k]][chromosome[k+1]]==0):
        count =+ 1     
    if(count==0):    
      initial_population.append(chromosome)
  return initial_population

def evaluate(chromosome):
  distance = 0
  for i in range(0, len(chromosome)-1):
    distance = distance + adj_matrix[chromosome[i]][chromosome[i+1]]
  return distance

def fitness(chromosome):
    return round(1/evaluate(chromosome), 4)

# Works correctly - ordered crossover
def crossover(parent1, parent2):
  del parent1[len(parent1)-1]
  del parent2[len(parent2)-1]
  parents = [parent1, parent2]
  random.shuffle(parents)
  child = []
  indices = random.sample(range(0, len(parent1)), 2)
  indices.sort()
  from_parent1 = parents[0][indices[0]:indices[1]+1]
  for value in from_parent1:
    parents[1].remove(value)
  child = from_parent1 + parents[1] + [from_parent1[0]]
  return child

# Works correctly - swap mutation
def mutate(child):
  mutated_child = child.copy()
  n = len(mutated_child)
  num_mutated_genes = int(mutation_probability * len(child))
  for i in range(num_mutated_genes):
    indices = random.sample(range(1, len(mutated_child)-1), 2)
    mutated_child[indices[0]], mutated_child[indices[1]] = mutated_child[indices[1]], mutated_child[indices[0]]
  return mutated_child

population = initialize()
for g in range(0, generations):
  print("Generation", g+1)
  elites = []
  evaluations = []
  fitnesses = []
  selection_probabilities = []
  for member in population:
    evaluations.append(evaluate(member))
    fitnesses.append(fitness(member))
  # print("Population =", population)
  # print("Evaluations =", evaluations)  
  # print("Fitnesses =", fitnesses)
  sum_fitnesses = 0
  for f in fitnesses:
    sum_fitnesses += f
  for member in population:
    sp = round((fitness(member)/sum_fitnesses), 4)
    selection_probabilities.append(sp)
  # print("Selection Probabilities =", selection_probabilities)
  table = []
  for i in range(0, len(population)):
    table.append((population[i], evaluations[i], fitnesses[i], selection_probabilities[i]))
  table.sort(key=lambda tup: tup[2] , reverse=True)
  # print(table)
  # print('--------------------------------------')
  print("Best Member =", table[0][0])
  print("Distance =", table[0][1])
  # print("Fitness =", table[0][2])
  # print("Selection Probability =", table[0][3])
  # print('--------------------------------------')
  elites.append(table[0][0])
  elites.append(table[1][0])
  # print("Elites =", elites)
  del table[0:2]
  # print(table)
  mating_pool = []
  while(len(mating_pool)!=len(table)):
    for t in table:
      rnd = random.sample(range(0, 1), 1)
      r = round(rnd[0], 4)
      if t[3] >= r:
        mating_pool.append(t[0])
  # print("Mating Pool =", mating_pool)
  intermediate_generation = []
  while(len(intermediate_generation)!=len(mating_pool)):
    new_mating_pool = mating_pool.copy()
    parents_indices = random.sample(range(0, len(new_mating_pool)), 2)
    # print("*******")
    # print("Parents Indices = ", parents_indices)
    parent1 = new_mating_pool[parents_indices[0]].copy()
    parent2 = new_mating_pool[parents_indices[1]].copy()
    # print("parent1 =", parent1)
    # print("parent2 =", parent2)
    child = crossover(parent1, parent2)
    # print("child =", child)
    intermediate_generation.append(child)
    # print("*******")
  # print("intermediate_generation = ", intermediate_generation)
  num_mutated_members = int(len(intermediate_generation) * mutation_ratio)
  # print("num_mutated_members", num_mutated_members)
  mutated_indices = random.sample(range(0, len(intermediate_generation)), num_mutated_members)
  # print("mutated_indices =", mutated_indices)
  for index in mutated_indices:
    mutated_member = mutate(intermediate_generation[index])
    intermediate_generation[index] = mutated_member
  # print("intermediate_generation = ", intermediate_generation)
  next_generation = elites + intermediate_generation
  # print("next_generation =", next_generation)
  population = next_generation.copy()
  print("////////////////////////////////////////////////////////////////")
