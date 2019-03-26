import csv

import conceptnet5.uri
import fastText
import numpy as np
import pandas as pd
from conceptnet5.vectors import cosine_similarity
from keras.engine.saving import load_model
from keras.optimizers import Adam
from scipy.stats import spearmanr

from gan_tester import ConstMultiplierLayer

if __name__ == '__main__':
    word_tuples = []
    my_word_tuples = []
    nb_word_tuples = []
    retrowords = pd.read_hdf("../retroembeddings.h5", 'mat', encoding='utf-8')
    numberbatch = pd.read_hdf("../retrogan/numberbatch.h5")
    ft_model = fastText.load_model("../fasttext_model/fil9.bin")
    trained_model_path = "../trained_retro_gans/crawl/mae_0_01/toretrogen"
    retrogan = load_model(trained_model_path,
                          custom_objects={"ConstMultiplierLayer": ConstMultiplierLayer},
                          compile=False)
    retrogan.compile(optimizer=Adam(), loss=['mae'])
    retrogan.load_weights(trained_model_path)

    with open('rw.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:

            # print(f'Word1:\t{row[0]}\tWord2:\t{row[1]}\tSimscore:\t{row[2]}.')
            line_count += 1
            word_tuples.append(row)
            score = 0

            # conceptnet5.uri.concept_uri("en",row[0].lower())
            idx1 = conceptnet5.uri.concept_uri("en",row[0].lower())
            idx2 = conceptnet5.uri.concept_uri("en",row[1].lower())
            try:
                mw1 = retrowords.loc[idx1]
            except Exception as e:
                mw1 = ft_model.get_word_vector(row[0].lower())
                mw1 = np.array(retrogan.predict(mw1.reshape(1,300))).reshape((300,))
            try:
                mw2 = retrowords.loc[idx2]
            except:
                mw2 = ft_model.get_word_vector(row[1].lower())
                mw2 = np.array(retrogan.predict(mw2.reshape(1,300))).reshape((300,))
            score = cosine_similarity(mw1,mw2)

            my_word_tuples.append((row[0],row[1],score))
            try:
                idx1 = "/c/en/" + row[0].lower()
                idx2 = "/c/en/" + row[1].lower()
                nw1 = numberbatch.loc[idx1]
                nw2 = numberbatch.loc[idx2]
                score = cosine_similarity(nw1,nw2)
            except Exception as e:
                print("Not found for")
                print(e)
                # print(row[0])
                # print(row[1])
                score = 0
            nb_word_tuples.append((row[0], row[1], score))
        print(f'Processed {line_count} lines.')
    print(spearmanr([x[2] for x in word_tuples],[x[2] for x in my_word_tuples]))
    print(spearmanr([x[2] for x in word_tuples],[x[2] for x in nb_word_tuples]))
    # word_tuples = sorted(word_tuples,key=lambda x:(x[0],x[2]))
    # my_word_tuples = sorted(my_word_tuples,key=lambda x:(x[0],x[2]))
    # nb_word_tuples = sorted(nb_word_tuples,key=lambda x:(x[0],x[2]))
    # print("Theirs")
    # print(word_tuples)
    # print("Mine")
    # print(my_word_tuples)
    # errors = 0
    # print("Mine")
    # for tup in zip(word_tuples,my_word_tuples):
    #     if tup[0][1] != tup[1][1]:
    #         errors+=1
    #     print(tup)
    # print(errors)
    # print(len(word_tuples))
    # errors = 0
    # print("NB")
    # for tup in zip(word_tuples,nb_word_tuples):
    #     if tup[0][1] != tup[1][1]:
    #         errors+=1
    #     print(tup)

