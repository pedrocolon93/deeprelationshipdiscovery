import numpy as np
import pandas as pd
from conceptnet5.nodes import standardized_concept_uri

from tools import find_closest_in_dataset

if __name__ == '__main__':
    numberbatch = pd.read_hdf("../retroembeddings.h5","mat",encoding="utf-8")
    testwords = ["dog","doggo","pink elephant","elephant"]
    for idx, word in enumerate(testwords):
        print(word)
        retro_representation = np.array(list(numberbatch.loc[standardized_concept_uri("en",word)]))
        find_closest_in_dataset(retro_representation, "../retroembeddings.h5",n_top=20)
