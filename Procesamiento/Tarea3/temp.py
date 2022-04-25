#!/usr/bin/env python
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt2
import numpy as np
import operator
from tkinter import ttk
from tkinter import *
from tabulate import tabulate
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import numpy as np

##*****************Leer Archivo de texto **************************
f = open (r'texto3.txt',encoding="utf8")
textoLower = f.read()
#print(textoLower)
f.close()


##*****************Quitar los caracteres especciales del texto**************************
quitar = ",;:.\n!\"'?¡¿—_«»<>-/|*+()&%$#=°”“"

textoLower = textoLower.lower() #Convertir a minusculas el texto
#print(textoLower)

for n in quitar:
    textoLower = textoLower.replace(n,"")  
#print(textoLower)


##*****************Palabras separadas por espacio**************************
palabras = textoLower.split(" ")
#print(palabras)
stop_words = set(stopwords.words('spanish')) #Elegimos las stopwords en español. 
#print(len(stop_words)) #Pedimos que muestre cuántas hay (313)
#print("Lista de stopwords en español:")
#print(stop_words)

##*****************Filtar palabras auxiliares**************************
filtered_sentence = [] 
  
for w in palabras: 
    if w not in stop_words: 
        filtered_sentence.append(w) 

#print(filtered_sentence) 

##*****************Lista y Diccionario**************************
dicFrec = {}
frecuencia = []
palabrOrden = []
list2 =[]
list3 =[]

##*****************Contabilizar el número de veces que se repite una palabra**************************
for palabra in filtered_sentence:
    if palabra in dicFrec:
        dicFrec[palabra] += 1
    else:
        dicFrec[palabra] = 1


##*****************Extraer los datos del diccionario para guardar en listas y poder graficar**************************
for palabra in dicFrec:
    frecuencia.append(dicFrec[palabra])
    palabrOrden.append(palabra)
    frecuencias = dicFrec[palabra]
    #print(f"'{palabra}'{frecuencias}")
#print(palabrOrden)
#print(frecuencia)


##*****************Variables para graficar**************************
values2 = palabrOrden
for cantidad in frecuencia:
    list2.append(1/frecuencia[cantidad])
    list3.append(np.log10(frecuencia[cantidad]))
#print(list2)
#print(list3)
dates = list2
dates2 = list3

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
plt.plot(values2, dates, color="green")
##plt.grid(True)
##plt.barh(values, dates, color="green")
plt.ylabel('Frecuencia de Uso')
plt.xticks(rotation=90, fontsize=5)
plt.xlabel('Palabras auxiliares encontradas')
plt.title('Frecuencia de uso de palabras auxiliares en el texto')
plt.show()

fig,ax = plt2.subplots()
plt2.plot(values2, dates2, color="purple")
##plt.grid(True)
##plt.barh(values, dates, color="green")
plt2.ylabel('Frecuencia de Uso')
plt2.xticks(rotation=90, fontsize=5)
plt2.xlabel('Palabras auxiliares encontradas')
plt2.title('Escala logarítmica de uso de palabras auxiliares en el texto')
plt2.show()



