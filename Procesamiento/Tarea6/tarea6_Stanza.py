#!/usr/bin/env python
import numpy as np
import operator
from tabulate import tabulate
import stanza
import matplotlib
import matplotlib.pyplot as plt

##*****************Leer Archivo de texto **************************
#f = open (r'texto1.txt',encoding="utf8")
#f = open (r'texto2.txt',encoding="utf8")
f = open (r'texto3.txt',encoding="utf8")
textoLower = f.read()
##print(textoLower)
f.close()


##*****************Quitar los caracteres especciales del texto**************************
quitar = ",;:.\n!\"'?¡¿—_«»<>-/|*+()&%$#=°"
textoLower = textoLower.lower() #Convertir a minusculas el texto
#print(textoLower)
for n in quitar:
    textoLower = textoLower.replace(n,"")  
#print(textoLower)


##*****************Palabras separadas por espacio**************************
palabras = textoLower.split(" ")
#print(palabras)


##*****************Lista y Diccionario**************************
dicFrec = {}
palabrOrden = []
ranks = []
freqs = []


##*****************Contabilizar el número de veces que se repite una palabra**************************
for palabra in palabras:
    if palabra in dicFrec:
        dicFrec[palabra] += 1
    else:
        dicFrec[palabra] = 1


##*****************Ordenar palabra por frecuancias de la más alta a la más baja**************************
dicFrecSort = sorted(dicFrec.items(), key=operator.itemgetter(1), reverse=True)
##print(dicFrecSort)
fd_sorted_dict = dict(dicFrecSort)


##*****************Extraer los datos del diccionario para guardar en listas y poder graficar**************************
for rank, palabra in enumerate(fd_sorted_dict):
    palabrOrden.append(palabra)
    ranks.append(rank + 1)
    freqs.append(fd_sorted_dict[palabra])
    ##print(pa, '->', frec)


##*****************Convertir diccionario a tupla para poder tabular las frecuencias**************************
l1 = list(dicFrecSort)
##print(l1)
head = ["Palabras", "Frecuencia"]
tabla = tabulate(l1, headers=head, tablefmt='grid')
f = open ('tablaFrecuenciaStanza.txt','w',encoding="utf8")
f.write(tabla)
f.close()


##*****************Convertir la lista de palabras ordenadas en un texto para el análisis de stanza**************************
StrA = " ".join(palabrOrden)
#print(StrA)


##*****************Comenzamos el análisis morfológico con stanza**************************
dicLemma = {}
stanza.download('es')
nlp = stanza.Pipeline('es')

doc = nlp(StrA)

for sent in doc.sentences:
    for pal in sent.words:
        dicLemma[pal.text] = pal.lemma
        #print(f'Palabra: {pal.text} \tlemma: {pal.lemma}') 
#print(dicLemma)

##*****************Convertir diccionario a tupla para poder tabular los lemmas**************************
l2 = list(dicLemma.items())
##print(l1)
head2 = ["Palabras", "Lemma"]
tabla2 = tabulate(l2, headers=head2, tablefmt='grid')
f = open ('tablaLemmaStanza.txt','w',encoding="utf8")
f.write(tabla2)
f.close()

##*****************Función para gráficos**************************
matplotlib.use('TkAgg')
plt.loglog(ranks, freqs, marker=".")
plt.title('Ley de Zipf', fontsize=14, fontweight='bold')
plt.xlabel('Frecuencia (f)')
plt.ylabel('Clasificación (r)')
for i, txt in enumerate(fd_sorted_dict):
    plt.annotate(str(txt), (ranks[i],freqs[i]),rotation=45)
plt.grid(True)
plt.show()


