#!/usr/bin/env python
from matplotlib import pyplot as plt
import numpy as np
import operator
from tabulate import tabulate
import spacy


##*****************Leer Archivo de texto **************************
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


##*****************Palabras separadas por espacio y ordenar por alfabeto**************************
palabras = textoLower.split(" ")
#print(palabras)
palabras = sorted(palabras)
#print(palabras)


##*****************Código con spacy**************************
nlp = spacy.load("es_core_news_sm")#español

doc = nlp(textoLower)

for token in doc: 
    f = open ('analisis_spacy.txt','a',encoding="utf8")
    f.write(f'Palabra: {token.text}, lemma: {token.lemma_}, pos: {token.pos_}, tag: {token.tag_}, dep: {token.dep_}, morph: {token.morph}\n')
    f.close()
    #print(f'Palabra: {token.text}, lemma: {token.lemma_}, pos: {token.pos_}, tag: {token.tag_}, dep: {token.dep_}, morph: {token.morph}') 


