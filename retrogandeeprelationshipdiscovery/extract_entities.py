import spacy
nlp = spacy.load("en_core_web_sm")

def extract_entities(filename):
    text = ""
    with open(filename) as f:
        for line in f:
            text+=line

    doc = nlp(text)
    things_to_expand = []
    compounds = []

    lemmas = [token.lemma_ for token in doc if not token.is_stop]
    print("Lemmas")
    print("-"*15)
    print(lemmas)
    print("Tokens")
    print("-"*15)

    for token in doc:
        print(token.text, token.lemma_, token.tag_, token.dep_,
              token.is_alpha, token.is_stop)
    print("Noun chunks")
    print("-"*15)

    for chunk in doc.noun_chunks:
        print(chunk.text, chunk.root.dep_,
              chunk.root.head.text)
    print("Entities")
    print("-"*15)

    for ent in doc.ents:
        if ent.text.strip()=="":
            continue
        print(ent.text.strip())
        things_to_expand.append(ent.lemma_)
    print("-" * 40)

    for token in doc:
        if token.is_stop or token.dep_=="punct" or token.tag_=="SYM":
            continue
        if token.text.strip()=="":
            continue
        if token.dep_=="compound":
            compounds.append(token)
        else:
            things_to_expand.append(token.lemma_)
    things_to_expand = list(set(things_to_expand))

    return things_to_expand

if __name__ == '__main__':
    filename = "4 Ways to Travel Between NYC and Boston"
    extract_entities(filename)
