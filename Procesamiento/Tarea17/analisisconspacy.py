import spacy    

nlp = spacy.load("es_core_news_sm")
piano_doc = nlp('Juan lee un libro')

for token in piano_doc:
    print(token.text, token.dep_)
    if token.dep_ == 'nsubj':    
        print (token.dep_+'('+token.head.text+','+token.text+')')
    elif token.dep_ == 'ROOT':    
        print (token.dep_+'('+token.head.text+','+token.text+')')
    elif token.dep_ == 'dobj':    
        print (token.dep_+'('+token.head.text+','+token.text+')')
    elif token.dep_ == 'aux':    
        print (token.dep_+'('+token.head.text+','+token.text+')')
    elif token.dep_ == 'prep':    
        print (token.dep_+'('+token.head.text+','+token.text+')')
    elif token.dep_ == 'pcomp':    
        print (token.dep_+'('+token.head.text+','+token.text+')')
    elif token.dep_ == 'compound':    
        print (token.dep_+'('+token.head.text+','+token.text+')')
    elif token.dep_ == 'quantmod':    
        print (token.dep_+'('+token.head.text+','+token.text+')')
    elif token.dep_ == 'quantmod':    
        print (token.dep_+'('+token.head.text+','+token.text+')')
    elif token.dep_ == 'pobj':    
        print (token.dep_+'('+token.head.text+','+token.text+')')
