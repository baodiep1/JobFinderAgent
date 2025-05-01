import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("SpaCy is now working correctly!")
print([(token.text, token.pos_) for token in doc])