#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipapp
#SImilitud coseno
import numpy as np
from numpy import dot
from numpy.linalg import norm
#ayuda a proporcionar la forma tabular que proporcione nupy
import pandas as pd
#Nos ayuda para expresiones regualres 
import re
import spacy

#**************Para el modelo de clasificacion 
import gensim
from gensim.models import Word2Vec


# In[2]:


##*****************Leer Archivo de texto **************************
f = open (r'texto1.txt',encoding="utf8")
texto1 = f.read()
#print("Texto 1\n" + texto1)
f.close()

f2 = open (r'texto2.txt',encoding="utf8")
texto2 = f2.read()
#print("Texto 2\n" + texto2)
f2.close()

f3 = open (r'texto3.txt',encoding="utf8")
texto3 = f3.read()
#print("Texto 3\n" + texto3)
f3.close()


# In[3]:


##*****************Quitar los caracteres especciales del texto**************************
quitar = ",;:.\n!\"?¡¿—_«»<>-/|*+()&%$#=°1234567890”“"

for n in quitar:
    texto1 = texto1.replace(n,"")
    texto2 = texto2.replace(n,"")  
    texto3 = texto3.replace(n,"")  
    
#print("Texto 1\n" + texto1)
#print("Texto 2\n" + texto2)
#print("Texto 3\n" + texto3)


# In[4]:


##*****************Comenzamos el análisis morfológico con spacy**************************
nlp = spacy.load("en_core_web_sm")

doc1 = nlp(texto1.lower())
doc2 = nlp(texto2.lower())
doc3 = nlp(texto3.lower())

#print(doc1)
#print(doc2)
#print(doc3)


# In[5]:


listPalabra1 = []
listLemma1 = []

listPalabra2 = []
listLemma2 = []

listPalabra3 = []
listLemma3 = []

for token1, token2, token3 in zip(doc1, doc2, doc3):
    listPalabra1.append(str(token1.text))
    listLemma1.append(str(token1.lemma_))
    
    listPalabra2.append(str(token2.text))
    listLemma2.append(str(token2.lemma_))
    
    listPalabra3.append(str(token3.text))
    listLemma3.append(str(token3.lemma_))


# In[6]:


#Imprimir de forma tabular las frecuencias y los índices que corresponden a los textos que se van a analizar con ayuda de pandas
lem1 = pd.DataFrame([listPalabra1,listLemma1], index=['PALABRA', 'LEMMA'])
dataframe1 = lem1.transpose()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
lem1.head()


# In[7]:


lem2 = pd.DataFrame([listPalabra2,listLemma2], index=['PALABRA', 'LEMMA'])
dataframe2 = lem2.transpose()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
lem2.head()


# In[8]:


lem3 = pd.DataFrame([listPalabra3,listLemma3], index=['PALABRA', 'LEMMA'])
dataframe3 = lem3.transpose()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
lem3.head()


# In[9]:


#Immprimir tabla en el txt
f1 = open ('lemmaTexto1.txt','w',encoding="utf8")
f1.write(str(dataframe1))
f1.close()

f2 = open ('lemmaTexto2.txt','w',encoding="utf8")
f2.write(str(dataframe2))
f2.close()

f3 = open ('lemmaTexto3.txt','w',encoding="utf8")
f3.write(str(dataframe3))
f3.close()


# In[10]:


#***************Establecer una lista dentro de otra lista para conservar las plabras para el modelo
auxTexto1 = []
auxTexto1.append(listLemma1)
#print(auxTexto1)

auxTexto2 = []
auxTexto2.append(listLemma2)
#print(auxTexto2)

auxTexto3 = []
auxTexto3.append(listLemma3)
#print(auxTexto3)


# In[11]:


# Create CBOW model
model1 = Word2Vec(auxTexto1, min_count = 1)
print(model1)

model2 = Word2Vec(auxTexto2, min_count = 1)
print(model2)

model3 = Word2Vec(auxTexto3, min_count = 1)
print(model3)


# In[12]:


#********Crear un vocabulario************
vocabulary1 = list(model1.wv.key_to_index.keys())
#print(vocabulary1)

vocabulary2 = list(model3.wv.key_to_index.keys())
#print(vocabulary2)

vocabulary3 = list(model3.wv.key_to_index.keys())
#print(vocabulary3)


# In[13]:


#*************Crear diccionario con el vocabulario y su vector *********************
auxW2v1 = dict(zip(model1.wv.index_to_key, model1.wv.vectors))
#print(auxW2v1)

auxW2v2 = dict(zip(model2.wv.index_to_key, model2.wv.vectors))
#print(auxW2v2)

auxW2v3 = dict(zip(model3.wv.index_to_key, model3.wv.vectors))
#print(auxW2v3)


# In[14]:


#**************Imprimir en un archivo cada uno de los vectores**********************
f1 = open ('vectoresTexto1.txt','w',encoding="utf8")
f1.write(str(auxW2v1))
f1.close()

f2 = open ('vectoresTexto2.txt','w',encoding="utf8")
f2.write(str(auxW2v2))
f2.close()

f3 = open ('vectoresTexto3.txt','w',encoding="utf8")
f3.write(str(auxW2v3))
f3.close()


# In[15]:


#************Guardar cada uno de los vectores en una variable auxiliar ************************
auxVectors1 = model1.wv.vectors
#print(auxVectors1)

auxVectors2 = model2.wv.vectors
#print(auxVectors2)

auxVectors3 = model3.wv.vectors
#print(auxVectors3)


# In[16]:


#***********************Función de suma de los vectores *************************
suma1 = 0
suma2 = 0
suma3 = 0

for valor1, valor2, valor3 in zip(auxVectors1, auxVectors2, auxVectors3):
    suma1 = suma1 + valor1
    suma2 = suma2 + valor2
    suma3 = suma3 + valor3

#print(f"La suma del vector1 es:\n {suma1}")
#print(f"La suma del vector2 es:\n {suma2}")
#print(f"La suma del vector3 es:\n {suma3}")


# In[17]:


#**********************Y el promedio se obtiene dividiendo la suma entre la cantidad de elementos
cantidad_elementos1 = len(auxVectors1)
promedio1 = suma1 / cantidad_elementos1
#print(f"El vector promedio de texto 1 es: \n{promedio1}")

cantidad_elementos2 = len(auxVectors2)
promedio2 = suma2 / cantidad_elementos2
#print(f"El vector promedio de texto 2 es: \n{promedio2}")

cantidad_elementos3 = len(auxVectors3)
promedio3 = suma3 / cantidad_elementos3
#print(f"El vector promedio de texto 3 es: \n{promedio3}")


# In[18]:


f1 = open ('vectoresPromedioTexto1.txt','w',encoding="utf8")
f1.write("Vector promedio de Texto 2 \n "+str(promedio1))
f1.close()

f2 = open ('vectoresPromedioTexto2.txt','w',encoding="utf8")
f2.write("Vector promedio de Texto 2 \n "+str(promedio2))
f2.close()

f3 = open ('vectoresPromedioTexto3.txt','w',encoding="utf8")
f3.write("Vector promedio de Texto 3 \n "+str(promedio3))
f3.close()


# In[19]:


#************************Calcular similitud coseno entre los vectores******************************
result1 = dot(promedio1, promedio2)/(norm(promedio1)*norm(promedio2))
print("La similitud coseno entre Texto 1 y Texto 2 es de: " , result1)

result2 = dot(promedio1, promedio3)/(norm(promedio1)*norm(promedio3))
print("La similitud coseno entre Texto 1 y Texto 3 es de: " , result2)

result3 = dot(promedio2, promedio3)/(norm(promedio2)*norm(promedio3))
print("La similitud coseno entre Texto 2 y Texto 3 es de: " , result3)

