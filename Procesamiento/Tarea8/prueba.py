#!/usr/bin/env python
import stanza

stanza.download('es')
nlp = stanza.Pipeline('es', processors = 'tokenize,mwt,pos,lemma,depparse')
sentence = 'Pedro llegó a casa en la noche'

doc = nlp(sentence)

doc.sentences[0].print_dependencies()


print ("{:<15} | {:<10} | {:<15} ".format('Token', 'Relation', 'Head'))
print ("-" * 50)
  
sent_dict = doc.sentences[0].to_dict()

for word in sent_dict:
  print ("{:<15} | {:<10} | {:<15} "
         .format(str(word['text']),str(word['deprel']), str(sent_dict[word['head']-1]['text'] if word['head'] > 0 else 'ROOT')))
