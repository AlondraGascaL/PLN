#!/usr/bin/env python
import stanza
import deplacy

stanza.download("es")
nlp=stanza.Pipeline("es")

doc=nlp("Pedro llegó a casa en la noche")
deplacy.render(doc)