
import numpy as np
import pandas as pd
from conceptnet5.nodes import standardized_concept_uri
from tqdm import tqdm

if __name__ == '__main__':
    path = "./glove.840B.300d.txt"
    cn = True
    clean = True
    df = pd.DataFrame()
    check_df = None
    if clean:
        check_df = pd.read_hdf("../fasttext_model/attract_repel.hd5clean","mat")
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
    df.to_hdf("glove.hd5","mat")


