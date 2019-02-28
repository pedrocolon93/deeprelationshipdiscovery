import h5py
import numpy as np
import pandas as pd
directory = '/home/conceptnet/conceptnet5/data/vectors/'
original = 'fasttext-opensubtitles.h5'
retrofitted = 'fasttext-opensubtitles-retrofit.h5'
o = pd.read_hdf(directory+original, 'mat', encoding='utf-8')
asarray1 = np.array(o.iloc[:,:])

# print(asarray1.shape)
# print(o["index"][0])
# print(o["columns"][0][:])
r = pd.read_hdf(directory+retrofitted,'mat',encoding='utf-8')
asarray2 = np.array(r.iloc[:,:])
r_sub = r.loc[o.index.intersection(r.index),:]
# print(asarray2.shape)
print(r_sub)

# print(r.iloc[range(10),:])