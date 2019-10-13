import csv
import pandas as pd
from conceptnet5.vectors import standardized_concept_uri
import numpy as np

if __name__ == '__main__':
    input_filename = "unpacked_cn"
    output_filename = "unfitted.hd5clean"
    indexes = []
    vectors = []
    in_hdf = pd.read_hdf(output_filename, "mat", encoding='utf-8')
    count = 0
    with open(input_filename,"w",encoding="utf-8") as vec_file:

        for item in in_hdf.index:
            count+=1
            # print(item)
            x = in_hdf.loc[item,:]
            lan, word = item.split("/")[2:4]
            # print(lan,word)
            line = lan+"_"+word+" "
            # line = item+" "
            for element in x:
                line+=str(element)+" "
            line+="\n"
            vec_file.write(line)
            if count%10000==0:
                print(count)
    print("Finished")

