import argparse
import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

from remove_from_corpus import *

path_to_ar = "/media/pedro/ssd_ext/attract-repel/"
ar_env_name = "attract_repel"
activate_path ="/home/pedro/anaconda3/bin/activate"
prefix = "en_"


def to_hdf(file,outputname):
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
        df = pd.DataFrame(index=indexes,data=vectors)
        df.to_hdf(outputname,"mat")


def to_txt(param):
    df = pd.read_hdf(param,"mat")

    pass


def generate_seen(ov, ar):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ov",
                        help="Path to text/vec file for original vectors",
                        )
    parser.add_argument("cl",
                        help="Amount of vectors to utilize",
                        default=-1,
                        type=int)
    parser.add_argument("cv",
                        help="Path for the output cleaned vectors",
                        default=None)
    parser.add_argument("synonyms",
                        help="File that contains a list of antonyms",
                        default="antonyms.txt")
    parser.add_argument("antonyms",
                        help="Text vectors that will be filtered out",
                        default="synonyms.txt")
    parser.add_argument("-tn",
                        required=False,
                        help="Temp file name",
                        default="temp.txt")
    parser.add_argument("-ccn",
                        required=False,
                        help="Complete corpus name",
                        default="completecorpus.txt")
    parser.add_argument("-dcn",
                        required=False,
                        help="Disjoint corpus name",
                        default="disjoint.txt")
    parser.add_argument("clean",
                        help="file that has the names of words that will be filtered out. Leave empty for no cleaning",
                        default=None)
    parser.add_argument("aroutput",
                        help="The output of the attract-repel algorithm")
    args = parser.parse_args()



    # Clean the dataset
    # CV is cleaned vectors
    print("Done")
    if args.clean:
        print("Cleaning the dataset")
        #Has everything
        remove_from_corpus_2(antonyms_file=args.antonyms, synonyms_file=args.synonyms, corpuse_to_clean=args.ov,
                           output_corpus=args.ccn,sl_sv_words="../simlexsimverb.words")
        # Output corpus has things in constraints.  May not have things in simlexsimverb. This is the training set.
        # Only has constraints
        remove_from_corpus(antonyms_file=args.antonyms, synonyms_file=args.synonyms, corpuse_to_clean=args.ov,
                           output_corpus=args.cv)
        # Does not have sl-sv from constriants
        remove_constraints_from_corpus(sl_sv_words="../simlexsimverb.words", corpuse_to_clean=args.ov,
                           output_corpus=args.dcn)
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

output_filepath=results/{aroutput}

[hyperparameters]

attract_margin = 0.6
repel_margin = 0.0
batch_size = 50
l2_reg_constant = 0.000000001
max_iter = 5
    '''.format(
        distributional=os.path.abspath(args.ccn),
        antonyms=os.path.abspath(args.antonyms),
        synonyms=os.path.abspath(args.synonyms),
        aroutput=args.aroutput
    )
    with open(os.path.join(path_to_ar, "config", "ar_config.cfg"), "w") as f:
        f.write(configstring)
    print("Done\nRunning AR with configuration:")
    print(configstring)
    print("\n"*3)

    process = subprocess.run('../run_ar.sh',
                   shell=True,
                   )
    # print("Printing the process")
    # print(process)
    print("Done outputting")
    # Load the initial word vectors
    print("Saving to hdfs")
    print("Original vectors:")
    to_hdf(args.ccn,args.ccn+".hdf")
    print("Ar vectors")
    to_hdf(os.path.join(path_to_ar,"results",args.aroutput),args.dcn+".hdf")
    print("Seen vectors")
    # shutil.copy(os.path.join(path_to_ar,"results",args.aroutput), "./arvecs.txt")