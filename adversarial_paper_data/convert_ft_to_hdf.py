
import numpy as np
import pandas as pd
from conceptnet5.nodes import standardized_concept_uri
from tqdm import tqdm

if __name__ == '__main__':
    ar_path = "./ft_ar.txt"
    distrib_path = "./ft_distrib.txt"
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
    to_df(ar_path,"ft_ar.hd5")
    to_df(distrib_path,"ft.hd5")

