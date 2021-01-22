import os
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm

from remove_constraints_from_vectxt import *

prefix = "en_"


def to_hdf(file, outputname):
    count = 0
    with open(file) as tv:
        indexes = []
        vectors = []
        for line in tv:
            count += 1
            word = line.strip().split(" ")[0]
            if prefix not in word:
                word = prefix + word
            vec = []
            for element in line.strip().split(" ")[1:]:
                vec.append(float(element))
            indexes.append(word)
            vectors.append(np.array(vec))
            if count % 10000 == 0:
                print(count)
        df = pd.DataFrame(index=indexes, data=vectors)
        df.to_hdf(outputname, "mat")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ov",
                        help="Path to text/vec file for original vectors",
                        )
    parser.add_argument("--cl",
                        help="Amount of vectors to utilize",
                        default=-1,
                        type=int)
    parser.add_argument("--synonyms",
                        help="File that contains a list of antonyms",
                        default="antonyms.txt")
    parser.add_argument("--antonyms",
                        help="Text vectors that will be filtered out",
                        default="synonyms.txt")
    parser.add_argument("--tn",
                        required=False,
                        help="Temp file name",
                        default="temp.txt")
    parser.add_argument("--ccn",
                        required=False,
                        help="Complete corpus name",
                        default="completecorpus.txt")
    parser.add_argument("--dcn",
                        required=False,
                        help="Disjoint corpus name",
                        default="disjoint.txt")
    parser.add_argument("--prefix",default="en_")
    parser.add_argument("--aroutput",default="attract_repelled_vectors.txt",
                        help="The output of the attract-repel algorithm")
    parser.add_argument("--arconfigname", default="ar_config.cfg",
                        help="The output of the attract-repel algorithm")
    parser.add_argument("--path_to_ar", default="/media/pedro/ssd_ext/attract-repel/",
                        help="The output of the attract-repel algorithm")
    parser.add_argument("--path_to_ar_python", default="/media/pedro/ssd_ext/attract-repel/",
                        help="The output of the attract-repel algorithm")
    parser.add_argument("--output_dir",
                        help="The output of the attract-repel algorithm")
    parser.add_argument("--skip_ar", default=False,action="store_true",
                        help="The output of the attract-repel algorithm")
    parser.add_argument("--skip_prefix", default=False, action="store_true",
                        help="The output of the attract-repel algorithm")
    parser.add_argument("--arvectors",
                        help="The output of the attract-repel algorithm")
    parser.add_argument("--origvectors",
                        help="The output of the attract-repel algorithm")
    args = parser.parse_args()
    path_to_ar = args.path_to_ar
    path_to_ar_python = args.path_to_ar_python
    words = set()

    if not args.skip_prefix:
        with open(args.synonyms) as f:
            for line in f:
                linecontents = line.strip().split()
                words.add(linecontents[0])
                words.add(linecontents[1])
        with open(args.antonyms) as f:
            for line in f:
                linecontents = line.strip().split()
                words.add(linecontents[0])
                words.add(linecontents[1])
        with open(os.path.abspath(args.ccn)) as dist_vecs:
            print("Prefixing the input vecs JIC")
            with open(os.path.abspath(args.ccn)+"prefixed.txt","w") as prefixed_dist_vecs:
                for line in tqdm(dist_vecs):
                    word = line.strip().split()[0]
                    if args.prefix not in word:
                        word = args.prefix+word
                    if word in words:
                        if not args.prefix in line:
                            prefixed_dist_vecs.write(args.prefix+line)
    print("Outputting config for AR ")
    configstring = \
        '''[experiment]
log_scores_over_time=False
print_simlex=True

[data]

distributional_vectors = {distributional}

; lists with files containing antonymy and synonymy constraints should be inside square brackets, delimited by commas.
antonyms = [{antonyms}]
synonyms = [{synonyms}]

; if either types of constraints are not used, that can be specified as follows:
;antonyms = []
;synonyms = []

output_filepath={aroutput}

[hyperparameters]

attract_margin = 0.6
repel_margin = 0.0
batch_size = 50
l2_reg_constant = 0.000000001
max_iter = 5
    '''.format(
            distributional=os.path.abspath(args.ccn)+"prefixed.txt",
            antonyms=os.path.abspath(args.antonyms),
            synonyms=os.path.abspath(args.synonyms),
            aroutput=args.aroutput
        )
    configpath = os.path.join(path_to_ar, "config", args.arconfigname)
    try:
        with open(configpath, "w") as f:
            f.write(configstring)
        print("Done writing config")

    except Exception as e:
        print(e)
    print(configstring)
    print("\n" * 3)
    if not args.skip_ar:
        print("Running AR")
        print("Arguments are:")

        arsubprocessargs = ['./run_ar.sh',path_to_ar_python, path_to_ar, configpath]
        print(arsubprocessargs)
        process = subprocess.run(args=arsubprocessargs, shell=False)
    print("Done outputting")
    print("Saving to hdfs")
    print("Original vectors:")
    if args.origvectors is not None:
        to_hdf(args.origvectors,
               args.output_dir + "/original.hdf")  # TODO cut to only aroutputones
    else:
        to_hdf(os.path.abspath(args.ccn)+"prefixed.txt", args.output_dir+"/original.hdf")#TODO cut to only aroutputones
    print("Ar vectors")
    if args.arvectors is not None:
        to_hdf(args.arvectors, args.output_dir + "/arvecs.hdf")
    else:
        to_hdf(args.aroutput, args.output_dir+"/arvecs.hdf")
    # print("Seen vectors")
    # shutil.copy(os.path.join(path_to_ar,"results",args.aroutput), "./arvecs.txt")
