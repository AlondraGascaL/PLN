import math

def disEuc (x, y):
    #Listas auxiliares
    a = []
    b = []
    #Obteer los valores de las listas
    x = x[:]        
    y = y[:]  
    #Variables para las listas
    suma_de_notas = 0
    
    #resta
    for i,o in zip(x,y):
        a.append(i-o)
    
    #Potencia de la resta
    for p in a:
        b.append(pow(p,2))
    
    #sumar las listas de potencias
    for nota1 in b:
        suma_de_notas  += nota1
    
    #calcular la raiz
    raiz1 = math.sqrt(suma_de_notas)
    
    return raiz1
    
    
def simCoseno(x, y):
    #Listas auxiliares
    a = []
    b = []
    #Obteer los valores de las listas
    x = x[:]        
    y = y[:]  
    #Variables para las listas
    suma_de_notas = 0
    suma_de_notas2 = 0
    
    #Multiplicación
    res0 = x[0]*y[0]
    res1 = x[1]*y[1]
    res2 = x[2]*y[2]
    res3 = x[3]*y[3]
    
    #Obtener las potencias de las listas y guardar en listas auxiliares
    for i,o in zip(x,y):
        a.append(pow(i,2))
        b.append(pow(o,2))
    
    #sumar las listas de potencias
    for nota1, nota2 in zip(a,b):
        suma_de_notas  += nota1
        suma_de_notas2  += nota2
    
    #calcular la raiz
    raiz1 = math.sqrt(suma_de_notas)
    raiz2 = math.sqrt(suma_de_notas2)
    
    #Calcular similitud coseno
    sim = ((res0)+(res1)+(res2)+(res3))/((raiz1)*(raiz2))
    return sim


def norm(x, y):
    #Listas auxiliares
    a = []
    b = []
    #Obteer los valores de las listas
    x = x[:]        
    y = y[:]  
    #Variables para las listas
    suma_de_notas = 0
    
    #resta
    for i,o in zip(x,y):
        a.append(i-o)
    
    #Potencia de la resta
    for p in a:
        b.append(pow(p,2))
    
    #sumar las listas de potencias
    for nota1 in b:
        suma_de_notas  += nota1
    
    #calcular la raiz
    raiz1 = math.sqrt(suma_de_notas/4)
    
    return raiz1
    
    
#Main
CIC = [9,9000,25,3]
ESCOM = [9.2,4600,3,2]
torreEspacio = [224.2,56250,15,56]

#Distancia Eucladiana
print("Distancia Eucladiana de las características")
print("Euc(CIC,ESCOM)= ",disEuc(CIC,ESCOM))
print("Euc(CIC,TORRE ESPACIO)= ",disEuc(CIC,torreEspacio))
print("Euc(ESCOM,TORRE ESPACIO)= ",disEuc(ESCOM,torreEspacio))

#Similitud de coseno
print("\n\nSimilitud coseno de las características")
print("Sim(CIC,ESCOM)= ",simCoseno(CIC,ESCOM))
print("Sim(CIC,TORRE ESPACIO)= ",simCoseno(CIC,torreEspacio))
print("Sim(ESCOM,TORRE ESPACIO)= ",simCoseno(ESCOM,torreEspacio))

#Distancia Eucladiana Normalizada
print("\n\nDistancia Eucladiana Normalizada de las características")
print("Euc(CIC,ESCOM)= ",norm(CIC,ESCOM))
print("Euc(CIC,TORRE ESPACIO)= ",norm(CIC,torreEspacio))
print("Euc(ESCOM,TORRE ESPACIO)= ",norm(ESCOM,torreEspacio))