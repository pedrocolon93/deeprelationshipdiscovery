import csv
import pandas as pd
from conceptnet5.vectors import cosine_similarity
from scipy.stats import spearmanr

if __name__ == '__main__':
    word_tuples = []
    my_word_tuples = []
    nb_word_tuples = []
    retrowords = pd.read_hdf("../trained_models/retroembeddings/2019-04-08 13:03:02.430691/retroembeddings.h5", 'mat', encoding='utf-8')
    numberbatch = pd.read_hdf("../retrogan/numberbatch.h5")
    with open('combined.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                print(f'Word1:\t{row[0]}\tWord2:\t{row[1]}\tSimscore:\t{row[2]}.')
                line_count += 1
                word_tuples.append(row)
                score = 0

                try:
                    idx1 = "/c/en/"+row[0].lower()
                    idx2 = "/c/en/"+row[1].lower()
                    mw1 = retrowords.loc[idx1]
                    mw2 = retrowords.loc[idx2]
                    score = cosine_similarity(mw1,mw2)
                except:
                    print("Not found for")
                    print(row[0])
                    print(row[1])
                    score=0
                my_word_tuples.append((row[0],row[1],score))
                try:
                    idx1 = "/c/en/" + row[0].lower()
                    idx2 = "/c/en/" + row[1].lower()
                    nw1 = numberbatch.loc[idx1]
                    nw2 = numberbatch.loc[idx2]
                    score = cosine_similarity(nw1,nw2)
                except:
                    print("Not found for")
                    print(row[0])
                    print(row[1])
                    score = 0
                nb_word_tuples.append((row[0], row[1], score))
        print(f'Processed {line_count} lines.')
    print(spearmanr([x[2] for x in word_tuples],[x[2] for x in my_word_tuples]))
    print(spearmanr([x[2] for x in word_tuples],[x[2] for x in nb_word_tuples]))
    word_tuples = sorted(word_tuples,key=lambda x:(x[0],x[2]))
    my_word_tuples = sorted(my_word_tuples,key=lambda x:(x[0],x[2]))
    nb_word_tuples = sorted(nb_word_tuples,key=lambda x:(x[0],x[2]))
    print("Theirs")
    print(word_tuples)
    print("Mine")
    print(my_word_tuples)
    errors = 0
    print("Mine")
    for tup in zip(word_tuples,my_word_tuples):
        if tup[0][1] != tup[1][1]:
            errors+=1
        print(tup)
    print(errors)
    print(len(word_tuples))
    errors = 0
    print("NB")
    for tup in zip(word_tuples,nb_word_tuples):
        if tup[0][1] != tup[1][1]:
            errors+=1
        print(tup)
    print(errors)
    print(len(word_tuples))

