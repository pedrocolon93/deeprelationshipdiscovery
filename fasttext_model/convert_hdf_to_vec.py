import csv
import pandas as pd
from conceptnet5.vectors import standardized_concept_uri
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    input_filename = "glove_full_ar_vecs.hdf"
    output_filename = "glove_full_ar_vecs.txt"
    indexes = []
    vectors = []
    limit = 200000
    invecs = pd.read_hdf(input_filename,"mat")
    count = 0
    with open(output_filename,"w") as outfile:
        for index in tqdm(invecs.index):
            if count == limit:
                break
            v = invecs.loc[index]
            # if "/c/en/" not in index:
            #     continue
            word = index.replace("/c/en/","")#.replace("_"," ")
            word = word.replace("en_","")
            if word == "":
                continue
            string = word+" "+''.join([str(x)+" " for x in v]+["\n"])
            outfile.write(string)
            count+=1