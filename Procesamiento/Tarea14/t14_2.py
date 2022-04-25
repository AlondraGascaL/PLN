import numpy as np
#ayuda a proporcionar la forma tabular que proporcione nupy
import pandas as pd
#Nos ayuda para expresiones regualres 
import re
import math
from math import log

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
    product = [x1*y1 for x1,y1 in zip(x,y)]
    
    #Suma de elementos de la lista
    listSum = sum(product)
    
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
    sim = listSum/((raiz1)*(raiz2))
    return sim
    
#Funcion para calcular la TF-IDF
def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf
    
#Funcion para calcular IDF
def calcular_idf(documents, N):
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = log(N / float(val),10)
    return idfDict

#Funcion que calcula TF a de las palabrasen los textos
def calcular_tf(unionFinal,texto):
  #inicializa los valores de las palabras existentes en 0
  tf_diz = dict.fromkeys(unionFinal,0)
  #cuenta las palabras que se encuentran en el texto unificado
  for palabra in texto:
      tf_diz[palabra]=log(texto.count(palabra)+1,10)
  return tf_diz


 
##*****************Leer Archivo de texto **************************
f = open (r'texto1.txt',encoding="utf8")
texto1 = f.read()
##print(textoLower)
f.close()

f2 = open (r'texto2.txt',encoding="utf8")
texto2 = f2.read()
##print(textoLower)
f2.close()

f3 = open (r'texto3.txt',encoding="utf8")
texto3 = f3.read()
##print(textoLower)
f3.close()

#Unificar los textos (minusculas, quitar espacios, quitar signos y caracteres especiales)
texto1_min = re.sub(r"[^a-zA-Z0-9]", " ", texto1.lower()).split()
texto2_min = re.sub(r"[^a-zA-Z0-9]", " ", texto2.lower()).split()
texto3_min = re.sub(r"[^a-zA-Z0-9]", " ", texto3.lower()).split()

#Union de las palabras de los textos ((1,2),3)
union1_2 = np.union1d(texto1_min,texto2_min)
unionFinal  =  np.union1d(union1_2,texto3_min)
print("Palabras encontradas en los textos: ", unionFinal)

#Clacular la frecuencia de las palabras por textos unidos comparando el texto unificado
print("**********************Frecuencias (tf) *********************")
tf_texto1 = calcular_tf(unionFinal,texto1_min)
tf_texto2 = calcular_tf(unionFinal,texto2_min)
tf_texto3 = calcular_tf(unionFinal,texto3_min)
print("\n\ntexto_1 ", tf_texto1)
print("texto_2 ", tf_texto2)
print("texto_3 ", tf_texto3)
print("\n\n")

#Imprimir de forma tabular las frecuencias y los índices que corresponden a los textos que se van a analizar con ayuda de pandas
df_tf = pd.DataFrame([tf_texto1,tf_texto2,tf_texto3], index=['Texto 1', 'Texto 2', 'Texto 3'])
dataframe2 = df_tf.transpose()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#Immprimir tabla en el txt
f3 = open ('TF_palabras.txt','w',encoding="utf8")
f3.write(str(dataframe2))
f3.close()

#numero de documentos
nDocuments = len(df_tf)
print(nDocuments)

#Clacular IDF, con el conjunto de textos y sus frecuencias
print("**********************Inverse Data Frequency (IDF)*********************")
idfs = calcular_idf([tf_texto1, tf_texto2, tf_texto3], nDocuments)
print(idfs)
print("\n\n")

#Imprimir de forma tabular lod idfs y los índices que corresponden a los textos que se van a analizar con ayuda de pandas
df_idf = pd.DataFrame(idfs, index=['idfs'])
dataframe3 = df_idf.transpose()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#Immprimir tabla en el txt
f3 = open ('IDF_palabras.txt','w',encoding="utf8")
f3.write(str(dataframe3))
f3.close()


#Calcular TF-IDF
print("**********************TF-IDF*********************")
tfidfTexto_1 = computeTFIDF(tf_texto1, idfs)
tfidfTexto_2 = computeTFIDF(tf_texto2, idfs)
tfidfTexto_3 = computeTFIDF(tf_texto3, idfs)
print("\n\ntexto_1 ", tfidfTexto_1)
print("texto_2 ", tfidfTexto_2)
print("texto_3 ", tfidfTexto_3)
print("\n\n")

#Imprimir de forma tabular las frecuencias y los índices que corresponden a los textos que se van a analizar con ayuda de pandas
tf_idf = pd.DataFrame([tfidfTexto_1,tfidfTexto_2,tfidfTexto_3], index=['Texto 1', 'Texto 2', 'Texto 3'])
dataframe1 = tf_idf.transpose()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#Immprimir tabla en el txt
f = open ('TF-IDF_palabras.txt','w',encoding="utf8")
f.write(str(dataframe1))
f.close()


#Obtener los valores del TF_IDF y pasarlos a una lista para calcular similitud coseno
tex1 =tfidfTexto_1.values()
tex1 = list(tex1)
tex2 =tfidfTexto_2.values()
tex2 = list(tex2)
tex3 =tfidfTexto_3.values()
tex3 = list(tex3)

#Similitud de coseno
sim1 = simCoseno(tex1,tex2)
sim2 = simCoseno(tex1,tex3)
sim3 = simCoseno(tex2,tex3)
print("\n\nSimilitud coseno de las palabras")
print("Sim(Texto1, Texto2)= ",sim1)
print("Sim(Texto1, Texto3)= ",sim2)
print("Sim(Texto2, Texto3)= ",sim3)

#Imprimir de forma tabular las frecuencias y los índices que corresponden a los textos que se van a analizar con ayuda de pandas
tf_idf = pd.DataFrame([sim1,sim2,sim3], index=['Texto 1/Texto 2', 'Texto 1/Texto 3', 'Texto 2/Texto 3'])
dataframe1 = tf_idf.transpose()
dataframe1.rename(index={0:'Similitud Cos'}, inplace=True)
dataframe1.head()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#Immprimir tabla en el txt
f = open ('simCoseno_palabras.txt','w',encoding="utf8")
f.write(str(dataframe1))
f.close()
