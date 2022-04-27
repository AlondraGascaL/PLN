import math

def coCO (x, y):
    #Listas auxiliares
    a = []
    b = []
    #Obteer los valores de las listas
    x = x[:]        
    y = y[:]  
    #Variables para las listas
    suma_de_notas = 0
    suma_de_notas2 = 0
    sum1 = 0
    sum2 = 0
    sum3 = 0
    
    #sumar las listas 
    for nota1, nota2 in zip(x,y):
        suma_de_notas  += nota1
        suma_de_notas2  += nota2
    
    list1 = suma_de_notas/len(x)
    list2 = suma_de_notas2/len(y)
    
    
    #resta de la lista original - listas
    for i,p in zip(x,y):
        a.append(i-list1)
        b.append(p-list2)
    
    #multiplicación de lista a y b y suma
    for o,q in zip(a,b):
        sum1 += (o*q)
        
    #potencia de lista a y b, suma de la potencia  
    for r,s in zip(a,b):
        sum2 += (pow(r,2))
        sum3 += (pow(s,2))
       
    #multiplicación y raiz
    raiz1 = math.sqrt(sum2*sum3)
    
    return sum1/raiz1
    
    
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
    sum1 = 0
    sum2 = 0
    sum3 = 0
    
    #Vector unitario
    for r,s in zip(x,y):
        sum1 += (pow(r,2))
        sum2 += (pow(s,2))
    raiz1 = math.sqrt(sum1)
    raiz2 = math.sqrt(sum2)
    
    #normalización por lista
    for i,j in zip(x,y):
        a.append(i/raiz1)
        b.append(j/raiz2)
        
    #Distancia Eucladiana (suma de la resta de potencias
    for k,l in zip(a,b):
        sum3 += (pow((k-l),2))
        
    raiz3 = math.sqrt(sum3)
    return raiz3
    
    
#Main
CIC = [9,9000,25,3]
ESCOM = [9.2,4600,3,2]
torreEspacio = [224.2,56250,15,56]

#Coeficiente de Coorelación
print("Coeficiente de Coorelación de las características")
print("r(CIC,ESCOM)= ",coCO(CIC,ESCOM))
print("r(CIC,TORRE ESPACIO)= ",coCO(CIC,torreEspacio))
print("r(ESCOM,TORRE ESPACIO)= ",coCO(ESCOM,torreEspacio))

#Similitud de coseno
print("\n\nSimilitud coseno de las características")
print("Sim(CIC,ESCOM)= ",simCoseno(CIC,ESCOM))
print("Sim(CIC,TORRE ESPACIO)= ",simCoseno(CIC,torreEspacio))
print("Sim(ESCOM,TORRE ESPACIO)= ",simCoseno(ESCOM,torreEspacio))

#Normalización
print("\n\nNormalización de las características")
print("Euc(CIC,ESCOM)= ",norm(CIC,ESCOM))
print("Euc(CIC,TORRE ESPACIO)= ",norm(CIC,torreEspacio))
print("Euc(ESCOM,TORRE ESPACIO)= ",norm(ESCOM,torreEspacio))