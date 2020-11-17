import faiss  # make faiss available
import pandas as pd
import numpy as np

import tools

if __name__ == '__main__':
    d = 300  # dimension
    # fn = "trained_models/retroembeddings/2019-05-15 11:47:52.802481/retroembeddings.h5"
    fn = "fasttext_model/cskg_retrofitted"
    # fn = "/Users/pedro/PycharmProjects/OOVconverter/fasttext_model/unfitted.hd5clean"
    retrovecs = pd.read_hdf(fn,"mat")
    testvec = tools.find_in_dataset(["/c/en/cat/n"],retrovecs,prefix="")
    print("intern")
    tools.dimensionality=300
    print(tools.find_closest_in_dataset(testvec, retrovecs, n_top=40))
    print(testvec)
    index = faiss.IndexFlatIP(d)  # build the index
    print(index.is_trained)
    index.add(retrovecs.values.astype(np.float32))  # add vectors to the index
    print(index.ntotal)

    k = 20  # we want to see 4 nearest neighbors
    D, I = index.search(testvec.astype(np.float32), k)  # sanity check
    print(I)
    print(D)
    print(retrovecs.iloc[I[0]])
