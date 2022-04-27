#!/usr/bin/env python
import numpy as np
import operator
import stanza

##*****************Leer Archivo de texto **************************
#f = open (r'texto1.txt',encoding="utf8")
#f = open (r'texto2.txt',encoding="utf8")
f = open (r'texto3.txt',encoding="utf8")
textoLower = f.read()
##print(textoLower)
f.close()


##*****************Comenzamos el análisis con stanza**************************
dicLemma = {}
stanza.download('es')
nlp = stanza.Pipeline('es', processors = 'tokenize,mwt,pos,lemma,depparse')

doc = nlp(textoLower)

print('\n')
doc.sentences[0].print_dependencies()


