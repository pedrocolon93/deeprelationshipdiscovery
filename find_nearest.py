import faiss

import tools
import pandas as pd
import numpy as np
if __name__ == '__main__':
    dimensionality = 300
    n_top = 10
    o = pd.read_hdf("trained_models/retroembeddings/ft_full_alldata/retroembeddings.h5", 'mat', encoding='utf-8')
    testvec = tools.find_in_dataset(["cat"], o,prefix="")
    # testvec = pred_y
    # print(testvec)
    index = faiss.IndexFlatIP(dimensionality)  # build the index
    # print(index.is_trained)
    index.add(o.values.astype(np.float32))  # add vectors to the index
    # print(index.ntotal)
    tst = np.array([testvec.astype(np.float32)])
    tst = tst.reshape((tst.shape[0], tst.shape[-1]))
    # print(tst.shape)
    D, I = index.search(tst, n_top)  # sanity check
    # print(I)
    # print(D)
    # print(o.iloc[I[0]])
    final_n_results = o.iloc[I[0]].values
    final_n_results_words = o.index[I[0]].values
    print(final_n_results_words)