#!/usr/bin/env python
from matplotlib import pyplot as plt
import numpy as np
import operator
from tkinter import ttk
from tkinter import *
from tabulate import tabulate
import nltk

##*****************Leer Archivo de texto **************************
f = open (r'texto3.txt',encoding="utf8")
textoLower = f.read()
##print(textoLower)
f.close()


##*****************Quitar los caracteres especciales del texto**************************
quitar = ",;:.\n!\"'?¡¿—_«»<>-/|*+()&%$#=°”“"

textoLower = textoLower.lower() #Convertir a minusculas el texto
#print(textoLower)

for n in quitar:
    textoLower = textoLower.replace(n,"")  
#print(textoLower)


##*****************Palabras separadas ppor espacio y ordenar por alfabeto**************************
palabras = textoLower.split(" ")
#print(palabras)
palabras = sorted(palabras)
#print(palabras)

##*****************Lista y Diccionario**************************
dicFrec = {}
frecuencia = []
palabrOrden = []

##*****************Contabilizar el número de veces que se repite una palabra**************************
for palabra in palabras:
    if palabra in dicFrec:
        dicFrec[palabra] += 1
    else:
        dicFrec[palabra] = 1


##*****************Extraer los datos del diccionario para guardar en listas y poder graficar**************************
for palabra in dicFrec:
    frecuencia.append(dicFrec[palabra])
    palabrOrden.append(palabra)
    frecuencias = dicFrec[palabra]
    ##print(f"'{palabra}'{frecuencias}")
#print(palabrOrden)
#print(frecuencia)


##*****************Variables para graficar**************************
values2 = palabrOrden
dates = frecuencia


##*****************Convertir diccionario a tupla**************************
l1 = list(dicFrec.items())
##print(l1)
head = ["Palabras", "Frecuencia"]
tabla = tabulate(l1, headers=head, tablefmt='grid')
f = open ('tabla.txt','w',encoding="utf8")
f.write(tabla)
f.close()

##*****************Función para gráficos**************************
fig,ax = plt.subplots()
plt.plot(values2, dates, color="red")
##plt.grid(True)
##plt.barh(values, dates, color="green")
plt.ylabel('Frecuencia de Uso')
plt.xticks(rotation=90, fontsize=5)
plt.xlabel('Palabras encontradas')
plt.title('Frecuencia de uso de palabras en el texto ')
plt.show()
