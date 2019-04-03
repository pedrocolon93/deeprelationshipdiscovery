import os
import re
import xml

import spacy
from bs4 import BeautifulSoup
from spacy import displacy
from readability import Document

def remove_tags(text):
    return re.sub('<[^<]+?>', '', text)

if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")

    #load text
    text_in_files = []
    web_page_directory = "WebPages"
    for file in os.listdir(web_page_directory):
        print(file)
        with open(os.path.join(web_page_directory,file), 'r') as f:
            x = f.readlines()
            txt = ""
            for line in x: txt+=line
            text_in_files.append(txt)

    #clean
    readable_doc = Document(text_in_files[0])
    pure_text = remove_tags(str(readable_doc.title())+" "+str(readable_doc.summary()))
    doc = nlp(pure_text)
    for entity in doc.ents:
        print(entity.text, entity.label_)
    # displacy.serve(doc, style="dep")
    exit()

    text = (u"Can you plan a trip to New York?")
    doc = nlp(text)
    for entity in doc.ents:
        print(entity.text, entity.label_)
    print(doc)

    for token in doc:
        print(token.text, token.dep_, token.head.text, token.head.pos_,
              [child for child in token.children])

    # displacy.serve(doc, style="dep")

