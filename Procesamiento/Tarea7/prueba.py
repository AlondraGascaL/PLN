#!/usr/bin/env python
import stanza

stanza.download("en")
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
doc = nlp('Pedro came home in the evening')

print('\n')
for sentence in doc.sentences:
    print(sentence.constituency)