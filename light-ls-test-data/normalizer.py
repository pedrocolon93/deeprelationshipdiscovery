import argparse
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("inputtext",
                        help="Text vectors that will be converted to hdf",
                        default="cleaned_corpus.txt")
    parser.add_argument("outputtext",
                        help="The output hdf file")
    parser.add_argument("--limit", default=None,
                        help="Limit of vectors to load",
                        type=int)
    parser.add_argument("-sf", default=False, action='store_true',help="Skips first line")
    args = parser.parse_args()

    input_filename = args.inputtext
    output_filename = args.outputtext
    skip_first = args.sf
    limit = args.limit
    count = 0
    indexes = []
    vectors = []

    with open(input_filename) as inputfile:
        with open(output_filename,"w") as outputfile:
            for line in tqdm(inputfile):
                count += 1
                if skip_first: skip_first = False
                if limit:
                    if count == limit:
                        print("Reached limit", limit)
                        break
                word = line.strip().split(" ")[0]
                word = word
                vec = []
                for element in line.strip().split(" ")[1:]:
                    vec.append(float(element))
                v =normalize(np.array(vec).reshape(1,-1))
                if v.shape[1]!= 300:
                    print("Error",word)
                    continue
                string = word + " " + ''.join([str(float(x)) + " " for x in v[0]] + ["\n"])
                outputfile.write(string)
    print("Done")