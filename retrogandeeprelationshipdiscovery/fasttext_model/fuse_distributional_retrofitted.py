import argparse
import csv
import pandas as pd
from conceptnet5.vectors import standardized_concept_uri
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("distributional",
                        help="Text vectors that will be fused with retrofitted ones",
                        default="distributional.txt")
    parser.add_argument("retrofitted",default="retrofitted.txt",
                        help="File that contains the retrofitted vectors")
    parser.add_argument("output",default="fused.txt",
                        help="The name of the fusion output files",)
    parser.add_argument("-l",default=None, type=int)
    args = parser.parse_args()

    distributional_filename = args.distributional
    retrofitted_filename = args.retrofitted
    output_filename = args.output
    limit = args.l
    prefix = "en_"
    with open(distributional_filename) as dist_file:
        with open(output_filename,"w") as out_file:
            with open(retrofitted_filename) as retrofitted_file:
                labels = []
                vecs = []
                print("INgesting retrofitted")
                for line in tqdm(retrofitted_file):
                    separated = line.split(" ")
                    labels.append(separated[0].strip().replace(prefix,""))
                    vecs.append(line.replace(prefix,""))
                print("Going through the original and replacing.")
                count = 0
                for line in tqdm(dist_file):
                    sep = line.split(" ")
                    try:
                        idx = labels.index(sep[0].strip().replace(prefix,""))
                        out_file.write(vecs[idx].replace(prefix,""))
                    except:
                        out_file.write(line.replace(prefix,""))
                    count+=1
                    if limit:
                        if count==limit:
                            break