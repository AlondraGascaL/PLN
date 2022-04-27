#!/usr/bin/env python
import stanza

stanza.download('es')
nlp = stanza.Pipeline('es')

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

for sent in doc.sentences:
    for palabra in sent.words:
        print(f'Palabra: {palabra.text} \tlemma: {palabra.lemma} \tpos: {palabra.pos} \tfeats: {palabra.feats}') 
        
###\tmwt: {palabra.mwt}\tmwt: {palabra.depparse} \tner: {palabra.ner}') 
        
###=======================
###| Processor | Package |
###-----------------------
###| tokenize  | ancora  | Separador de texto en oracuiones y oraciones en palabras
###| mwt       | ancora  |  Reconoce las palabras compuestas
###| pos       | ancora  |  Clasifica las palabras en adjetivos o verbos o sujetos (ADJ/VERB/NOUN)
###| lemma     | ancora  |  Cambia la pablabra a como se encuentra en el diccionario sin conjugacioens
###| depparse  | ancora  |  Analaiza las dependencias indicando cual es la relacion con la palabra
###| ner       | conll02 |  
###=======================    
###<Word index=2;
###text=fue;
###lemma=ser;
###upos=AUX;
###xpos=AUX;
###feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin; HAZAÑAS O CARACTERISTICAS ADICIONALESS
###governor=4;
###dependency_relation=cop>)   