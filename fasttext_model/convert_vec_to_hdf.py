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

    count = 0
    prefix = ""
    indexes = []
    vectors = []
    with open(input_filename,encoding="utf-8") as vec_file:
        for line in tqdm(vec_file):
            count+=1
            word = line.strip().split(" ")[0]
            word = prefix+word
            vec = []
            for element in line.strip().split(" ")[1:]:
                vec.append(float(element))
            indexes.append(word)
            vectors.append(np.array(vec))
            if count%10000==0:
                print(count)
    print("Outputting df")
    df = pd.DataFrame(index=indexes,data=vectors)
    df.to_hdf(output_filename,"mat")
    print("Finished")

