import numpy as np #"permite generar aleatorio"
import copy

poblacion = []
tamPoblacion = 100
numGeneracion = 100
tamGenoma = 10
pobIni = []

# Seleccione
def topNSelectFull(topn=10, maximo=True):
    global poblacion, tamPoblacion
    poblacion.sort(key=lambda x: x.fitness, reverse=maximo)
    selected = copy.deepcopy(poblacion[:topn])
    c = 0
    for i in range(tamPoblacion):
        poblacion[i] = copy.deepcopy(selected[c])
        c += 1
        if c == topn:
            c = 0
            
# Mutaciones
def mutate(wGenoma, rotacionMutacion=0.1):
	for i in range(len(wGenoma)):
		if np.random.random() < rotacionMutacion:
   			wGenoma[i] = np.random.randint(2)



class individual:
    def __init__(self):
        self.fitness = 0 # Fitness individual
        self.Genoma = [] # Individual es una lista
    def setGenoma(self, newGenoma):
        self.Genoma = newGenoma
    def getGenoma(self):
        return self.Genoma



if __name__ == "__main__":
    
    for i in range(tamPoblacion):
        indiv = individual()
        indiv.setGenoma( np.zeros(tamGenoma, dtype=int) )
        poblacion.append( indiv )
        
    print("Población incial y Fitness Inicial:\n")    
    for generation in range(numGeneracion):
        for p in poblacion:
            #Calcula la aptitud de cada individuo
            p.fitness = sum(p.getGenoma())
        
        if generation % 1 == 0:
            print("Genoma ",generation+1," : ", poblacion[generation].getGenoma(), "-> Fitness : ", np.max( [p.fitness for p in poblacion] ) )

        topNSelectFull(topn=5, maximo=True)
        # Seleccione las cinco personas con la mejor forma física

        for i in range(tamPoblacion):
        # Muta el 10% de las personas.
            newGenoma = mutate(poblacion[i].getGenoma(), rotacionMutacion=0.1)


print("\n\n\n\n")
print("Nueva Población y Nuevo Fitness :\n")   
count = 0
for p in poblacion:
    count = count +1
    print("Genoma ",count," :", p.getGenoma(), "-> Fitness : ", np.max( [p.fitness for p in poblacion] ) )

