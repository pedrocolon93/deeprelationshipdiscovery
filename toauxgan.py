import argparse

import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ov",
                        help="Path to text/vec file for original vectors",
                        )
    parser.add_argument("rf",
                        help="Amount of vectors to utilize",
                        )
    output_original = "auxgan_original.txt"
    output_ar = "auxgan_ar.txt"
    args = parser.parse_args()


    o = pd.read_hdf(args.ov, 'mat', encoding='utf-8')
    r = pd.read_hdf(args.rf, 'mat', encoding='utf-8')
    cns = []


    print("Loading concepts")
    # o = o.swapaxes(0,1)
    # r = r.swapaxes(0,1)
    for i in tqdm(r.index):
        cns.append(i)

    X_train = o.loc[cns, :]
    Y_train = r.loc[cns, :]
    # print(X_train.to_string())
    with open(output_original,"w") as original:
        with open(output_ar,"w") as ar:
            for concept in tqdm(cns):
                ovline = concept+" "+' '.join(map(str, X_train.loc[concept,:]))+"\n"
                rvline = concept+" "+' '.join(map(str, Y_train.loc[concept,:]))+"\n"
                original.write(ovline)
                ar.write(rvline)

