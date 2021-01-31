from tools import hd5_to_txt
import pandas as pd
if __name__ == '__main__':
    path = "trained_models/retroembeddings/2019-05-15 11:47:52.802481/retroembeddings_modified.h5"
    dataset = pd.read_hdf(path,"mat")
    hd5_to_txt(dataset)