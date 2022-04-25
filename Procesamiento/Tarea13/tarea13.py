import numpy as np
#ayuda a proporcionar la forma tabular que proporcione nupy
import pandas as pd
#Nos ayuda para expresiones regualres 
import re
from math import log


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
f3 = open ('TF.txt','w',encoding="utf8")
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
f3 = open ('IDF.txt','w',encoding="utf8")
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
f = open ('TF-IDF.txt','w',encoding="utf8")
f.write(str(dataframe1))
f.close()
