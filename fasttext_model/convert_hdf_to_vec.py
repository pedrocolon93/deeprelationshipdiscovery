import csv
import pandas as pd
from conceptnet5.vectors import standardized_concept_uri
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    input_filename = "cskg_retrofitted"
    output_filename = "cskg_retrofitted.txt"
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
            word = index.replace("/c/en/","")#.replace("_"," ")
            word = word.replace("en_","")
            if word == "":
                continue
            string = word+" "+''.join([str(x)+" " for x in v]+["\n"])
            outfile.write(string)
            count+=1