
import numpy as np
import pandas as pd
from conceptnet5.nodes import standardized_concept_uri
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ar_path", help="increase output verbosity",default="./final_fdjglovevectors_exp.txt")
    parser.add_argument("-distrib_path", help="increase output verbosity",default="./glove_distrib.txt")
    parser.add_argument("-ar_hdf_path", help="increase output verbosity",default="glove_fdj_ar.hd5")
    parser.add_argument("-distrib_hdf_path", help="increase output verbosity",default="glove.hd5")

    args = parser.parse_args()

    ar_path = args.ar_path
    distrib_path = args.distrib_path
    ar_hdf_path = args.ar_hdf_path
    distrib_hdf_path = args.distrib_hdf_path

    cn = False
    clean = False
    df = pd.DataFrame()
    check_df = None
    def to_df(path,out_name):
        with open(path,'r') as glove_vecs_txt:
            i = 0
            for line in tqdm(glove_vecs_txt):
                vec = line.split(" ")
                if clean:
                    if not standardized_concept_uri("en",vec[0]) in check_df.index:
                        continue
                if cn:
                    word = standardized_concept_uri("en",vec[0])
                else:
                    word = vec[0]
                vec = np.array([float(x) for x in vec[1:]])
                df[word] = vec
                if i%10000==0:
                    print(df)
                i+=1
        print(df)
        df.to_hdf(out_name,"mat")



    to_df(ar_path, ar_hdf_path)
    to_df(distrib_path, distrib_hdf_path)

