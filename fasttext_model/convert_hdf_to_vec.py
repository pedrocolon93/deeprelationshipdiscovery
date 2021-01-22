import argparse
import csv
import pandas as pd
from conceptnet5.vectors import standardized_concept_uri
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("inputtext",
                        help="Text vectors that will be converted to hdf",
                        default="cleaned_corpus.txt")
    parser.add_argument("outputhdf",default="original_ft.hd5clean",
                        help="The output hdf file")
    args = parser.parse_args()


    input_filename = args.inputtext
    output_filename = args.outputhdf
    indexes = []
    vectors = []
    limit = 20000000
    invecs = pd.read_hdf(input_filename,"mat")
    count = 0
    with open(output_filename,"w") as outfile:
        for index in tqdm(invecs.index):
            if count == limit:
                break
            v = invecs.loc[index]
            # if "/c/en/" not in index:
            #     continue
            # word = index.replace("/c/en/","")#.replace("_"," ")
            word = index.strip()
            # if "en_" not in word:
            #     word = "en_"+word
            # word = word.replace("en_","")
            if word == "":
                continue
            string = word+" "+''.join([str(x)+" " for x in v]+["\n"])
            outfile.write(string)
            count+=1