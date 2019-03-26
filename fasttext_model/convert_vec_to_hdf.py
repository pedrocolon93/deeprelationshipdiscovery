import csv
import pandas as pd
from conceptnet5.vectors import standardized_concept_uri
import numpy as np

if __name__ == '__main__':
    input_filename = "cc.en.300.vec"
    output_filename = "unfitted.hd5"
    indexes = []
    vectors = []

    with open(input_filename,encoding="utf-8") as vec_file:
        line = 0
        for vec in vec_file:
            if line == 0:
                line+=1
                print(vec)
                continue
            if line%100000 == 0:
                print(line)
            vec = vec.split(" ")
            name = vec[0]
            vector = vec[1:-1]
            indexes.append(name)
            vectors.append(np.array([float(x) for x in vector]))
            line+=1



    out_df = pd.DataFrame(index=indexes,data=vectors)
    out_df.to_hdf(output_filename,"mat")
