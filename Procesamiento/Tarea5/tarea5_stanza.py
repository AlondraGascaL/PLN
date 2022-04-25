#!/usr/bin/env python
from matplotlib import pyplot as plt
import numpy as np
import operator
from tabulate import tabulate
import stanza

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



##*****************Código con stanza**************************
stanza.download('es')
nlp = stanza.Pipeline('es')

doc = nlp(textoLower)

for sent in doc.sentences:
    for palab in sent.words:
        f = open ('analisis_stanza.txt','a',encoding="utf8")
        f.write(f'Palabra: {palab.text}, lemma: {palab.lemma}, pos: {palab.pos}, feats: {palab.feats}\n')
        f.close()
        #print(f'Palabra: {palab.text} \tlemma: {palab.lemma} \tpos: {palab.pos} \tfeats: {palab.feats}') 

