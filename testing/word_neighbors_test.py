import csv

import conceptnet5.uri
import fastText
import numpy as np
import pandas as pd
import sklearn
from conceptnet5.nodes import standardized_concept_uri
from conceptnet5.vectors import cosine_similarity
from keras.engine.saving import load_model
from keras.optimizers import Adam
from scipy.stats import spearmanr, pearsonr

from retrogan_generatorandtester import ConstMultiplierLayer
from tools import find_closest_in_dataset

if __name__ == '__main__':
    numberbatch = pd.read_hdf("../retroembeddings.h5","mat",encoding="utf-8")
    testwords = ["dog","doggo","pink elephant","elephant"]
    for idx, word in enumerate(testwords):
        print(word)
        retro_representation = np.array(list(numberbatch.loc[standardized_concept_uri("en",word)]))
        find_closest_in_dataset(retro_representation, "../retroembeddings.h5",n_top=20)
