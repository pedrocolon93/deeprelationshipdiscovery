import argparse

import pandas as pd
from conceptnet5.nodes import standardized_concept_uri
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("inputhdf",
                        help="Hdf whose index will be used to filter out textual vectors.",
                        default="attract_repel.hd5clean")
    parser.add_argument("outputtext",default="fasttext_formatted_clean.txt",
                        help="The output text file that is filtered")
    parser.add_argument("targettext",
                        help="Text vectors that will be filtered out",
                        default="fasttext_formatted.txt")
    args = parser.parse_args()

    input_hdf = pd.read_hdf(parser.inputhdf)
    lines = ""
    with open(parser.outputtext,"w") as output_vectors:
        with open(parser.target) as input_vectors:
            count = 0
            for line in tqdm(input_vectors):
                name = line.split(" ")[0]
                clean_name = name.replace("en_","")
                s_name = standardized_concept_uri("en",clean_name)
                count+=1
                if s_name in input_hdf.index:
                    lines+=line
                if count%10000==0:
                    print(count)
            output_vectors.write(lines)