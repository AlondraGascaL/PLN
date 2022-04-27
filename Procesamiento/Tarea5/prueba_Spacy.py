#!/usr/bin/env python
import spacy

nlp = spacy.load("es_core_news_sm")#español

textoLower = ('''Un desastroso espirítu posee tu tierra:
            donde la tribu unida blandió sus mazas,
            hoy se enciende entre hermanos perpetua guerra,
            se hieren y destrozan las mismas razas.''')

quitar = ",;:.\n!\"'?¡¿—_«»<>-/|*+()&%$#=°"

textoLower = textoLower.lower() #Convertir a minusculas el texto
#print(textoLower)

for n in quitar:
    textoLower = textoLower.replace(n,"")  
#nlp = es_core_news_sm.load()

doc = nlp(textoLower)

for token in doc: 
    print(f'Palabra: {token.text}, lemma: {token.lemma_}, pos: {token.pos_}, tag: {token.tag_}, dep: {token.dep_}') 