import numpy #permite generar aleatorios

#Entradas de la ecuación
equation_inputs = [4,-2,3.5,5,-11,-4.7]
 
#Número de pesos que buscamos optimizar.
num_weights = len(equation_inputs)  # 6
 
sol_per_pop = 8
num_parents_mating = 4
 
# Definimos el tamaño de la población：(8,6).
pop_size = (sol_per_pop,num_weights)

# Se crea la población inical random
new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
 

 # salida matriz de población generada aleatoriamente (8,6)
print("Población inicial:")
print(new_population)


# Calcular fitness
def cal_pop_fitness(equation_inputs, pop):
    fitness = numpy.sum (pop * equation_inputs, axis = 1) # Cada fila de 6 bits corresponde a multiplicar y sumar para obtener la suma, se calcula un total de 8 sumas
    return fitness
 
 
# Genere los valores de aptitud calculados de las 8 soluciones: (8,6) * (6,1) = (8,1)
a = cal_pop_fitness(equation_inputs,new_population)
print('\nFitness inicial:\n', a)
 
 
 
# Seleccionar padres
def select_mating_pool(pop, fitness, num_parents):
    parents = numpy.empty ((num_parents, pop.shape [1])) # padres se utilizan para almacenar el padre seleccionado, forma: (4,6)
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        # print (max_fitness_idx) guarda la matriz ([3],), por lo que debe eliminar el valor específico 3 en el siguiente paso
        max_fitness_idx = max_fitness_idx[0][0]
        # print(max_fitness_idx)
        parents [parent_num,:] = pop [max_fitness_idx,:] # ponga la solución con el valor de aptitud más grande en el padre
        fitness [max_fitness_idx] = -99999999999 # Asigne el valor de aptitud de la solución seleccionada para que sea muy pequeño, para no volver a ser seleccionado
    return parents
 
 
b = select_mating_pool(new_population,a,num_parents_mating)
print('\nPadres:\n',b)



# Crossover de un solo punto
def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    crossover_point = numpy.uint8(offspring_size[1] / 2)  # 6/2=3
    print('\ncrossover point: ',crossover_point)
 
    for k in range(offspring_size[0]):
        # indice del primer padre que se unió
        parent1_idx = k % parents.shape[0]
        # indice del segundo padre que se unió
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring
 
 
offspring_size = (pop_size[0]-b.shape[0],num_weights)  # 8-4=4
c = crossover(b,offspring_size)
print('Unión offspring:\n',c)


# Mutación aleatoria
def mutation(offspring_crossover):
    for idx in range(offspring_crossover.shape[0]):
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    return offspring_crossover
 
 
d = mutation(c)
print('\nMutación offspring:\n',d)

# Nueva población: 4 padres + 4 crías
new_population[0:b.shape[0], :] = b
new_population[b.shape[0]:, :] = d
 
print("\nNueva Población:\n",new_population)



# Calcule la aptitud de la nueva población
e = cal_pop_fitness(equation_inputs,new_population)
print("\nNueva Población fitness:\n",e)