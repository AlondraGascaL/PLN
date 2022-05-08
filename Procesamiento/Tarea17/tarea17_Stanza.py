import stanza
#Nos ayuda para expresiones regualres 
import re

stanza.download('es')
nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma,depparse')


##*****************Leer Archivo de texto **************************
f = open (r'texto3.txt',encoding="utf8")
texto1 = f.read()
##print(textoLower)
f.close()




doc = nlp(texto1)
listaAux = {}


for sent in doc.sentences: 
    for word in sent.words:
        if word.head > 0:
            aux = word.deprel+'('+sent.words[word.head-1].text+'-'+str(word.head)+', '+word.text+'-'+str(word.id)+')'
            listaAux[word.id-1] = aux
        else:
            aux = word.deprel+'('+"root"+'-'+str(word.head)+', '+word.text+'-'+str(word.id)+')'
            listaAux[word.id-1] = aux
        
with open("salida_arbol_texto3.txt", 'w') as f: 
    for value in listaAux.values(): 
        f.write('%s\n' % value)