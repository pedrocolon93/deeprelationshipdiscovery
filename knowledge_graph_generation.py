# Load library
import re

import spacy
from keras.engine.saving import load_model
from keras.optimizers import Adam
from nltk.corpus import stopwords

# You will have to download the set of stop words the first time
import nltk

import deep_relationship_discovery
import tools
import networkx as nx
import pandas as pd
import numpy as np

from retrogan_trainer import ConstMultiplierLayer

nltk.download('stopwords')
from spacy.lang.en import English


def read_file(filename):
    content = ""
    with open(filename) as f:
        for line in f.readlines():
            content+=line
    # tokenized_content = nltk.word_tokenize(content)
    # string = re.sub('\n', '.', content)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(content)

    tokenized_content = [token.text for token in doc]
    tokenized_content = [word for word in tokenized_content if word.isalpha()]

    print(tokenized_content)
    # for token in doc:
    #     print(token.text, token.pos_, token.dep_)
    return tokenized_content


def clean_file(file_contents):
    # Load stop words
    stop_words = stopwords.words('english')
    clean_file_contents = [word for word in file_contents if word not in stop_words]
    return clean_file_contents

def generate_kg(clean_file_contents):
    print(len(clean_file_contents))
    clean_file_contents = list(set(clean_file_contents))
    drd_models_path = "trained_models/deepreldis/2019-04-2314:43:00.000000"
    target_file_loc = 'trained_models/retroembeddings/2019-04-0813:03:02.430691/retroembeddings.h5clean'
    trained_model_path = "trained_models/retrogans/2019-04-0721:33:44.223104/toretrogen.h5"
    # Load retrogan
    retrogan = load_model(trained_model_path,
                          custom_objects={"ConstMultiplierLayer": ConstMultiplierLayer},
                          compile=False)
    retrogan.compile(optimizer=Adam(), loss=['mae'])
    retrogan.load_weights(trained_model_path)
    # Load our vocabulary
    target_voc = pd.read_hdf(target_file_loc,'mat')
    # Load drd
    drd_model = deep_relationship_discovery.load_model_ours(save_folder=drd_models_path, model_name='all')
    # Load normalizers
    normalizers = deep_relationship_discovery.normalize_outputs(None, save_folder=drd_models_path,use_cache=True)

    triples = []
    #beef up our vocab with missing entries
    in_dataset = tools.check_index_in_dataset(clean_file_contents,target_voc)
    print()
    for i,val in enumerate(in_dataset):
        if not val:
            missing_text = clean_file_contents[i]
            # print("Missing:",missing_text)
            we = tools.generate_fastext_embedding(missing_text)
            # print("We:",we)
            index = tools.standardized_concept_uri("en",missing_text)
            # print(index)
            rwe = tools.get_retrofitted_embedding(we,retrogan)
            # print("Retrofitted_embedding",rwe)
            df = pd.DataFrame(data=[rwe],index=[index])
            target_voc = target_voc.append(df)
            print(target_voc.shape)


    for concept in clean_file_contents:
        try:
            print("Expanding concept",concept)
            w1 = tools.find_in_dataset([concept],target_voc)
            # print(w1)
            w1 = np.array([w1[0] for i in range(len(clean_file_contents)-2)])
            # print(np.array(w1))
            w2 = tools.find_in_dataset([x for x in clean_file_contents if x != concept],dataset=target_voc)
            # print(w2)
            # print(w1)
            # print(w2)

            for model in deep_relationship_discovery.relations:
                model=model.replace("/r/","")
                try:
                    inferred_res = drd_model[model].predict(x={"retro_word_1": w1,
                                                             "retro_word_2": w2})
                    norm_res = normalizers[model].transform(inferred_res)
                    print(norm_res.shape)
                    triples+=[x for x in zip([concept for i in range(len(clean_file_contents)-1)],
                                             [x for x in clean_file_contents if x != concept],
                                             [x[0] for x in norm_res],
                                             [model for i in range(len(clean_file_contents)-1)])]
                except Exception as e:
                    # print(e)
                    pass
            print("We now have",len(triples))
        except Exception as e:
            print(e)
    print(triples)
    cutoff = 0.7
    res = [x for x in filter(lambda x: 1>x[2]>cutoff ,triples)]
    print(res)

    # G = nx.Graph()
    # for concept in clean_file_contents:
    #     G.add_node(concept)
    # for edge in res:
    #     G.add_edge(edge[0],edge[1],weight=edge[2],label=edge[3])

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(20, 10))
    # pos = nx.spring_layout(G, k=0.25, iterations=200)
    # nx.draw(G,pos, with_labels=True)
    # nx.draw_networkx_edge_labels(G, pos)
    # plt.savefig('labels.png')

    # input("Hello")
    return res

if __name__ == '__main__':
    filename = "WebPages/travel_ny_bos.txt"
    contents = read_file(filename)
    clean_contents = clean_file(contents)
    generate_kg(clean_contents)
