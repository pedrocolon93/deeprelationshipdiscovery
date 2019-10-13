import csv
import pandas as pd
from conceptnet5.vectors import standardized_concept_uri
import numpy as np

if __name__ == '__main__':
    input_filename = "final_vectors_exp.txt"
    output_filename = "attract_repel.hd5clean"
    indexes = []
    vectors = []
    count = 0
    with open(input_filename,encoding="utf-8") as vec_file:
        for line in vec_file:
            count+=1
            lan = line[0:line.index("_")]
            word = line[line.index("_")+1:line.index(" ")]
            # print(line)
            # print(lan)
            # print(word)
            vec = []
            for element in line.split(" ")[1:]:
                vec.append(float(element))
            indexes.append(standardized_concept_uri(lan,word))
            vectors.append(np.array(vec))
            if count%10000==0:
                print(count)
    print("Outputting df")
    df = pd.DataFrame(data=vectors,index=indexes)
    df.to_hdf(output_filename,"mat")
    print("Finished")

