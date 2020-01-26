import argparse
import os
import subprocess

from tqdm import tqdm

from remove_from_corpus import remove_from_corpus

path_to_ar = "/media/pedro/ssd_ext/attract-repel/"
ar_env_name = "attract_repel"
activate_path ="/home/pedro/anaconda3/bin/activate"
prefix = "en_"
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
    parser.add_argument("clean",
                        help="file that has the names of words that will be filtered out. Leave empty for no cleaning",
                        default=None)
    parser.add_argument("aroutput",
                        help="The output of the attract-repel algorithm")
    args = parser.parse_args()

    # Trim the dataset
    # TV is trimmed vectors
    # OV is original vectors
    count = 0
    with open(args.tn, "w") as tv:
        with open(args.ov) as ov:
            for line in tqdm(ov):
                if args.cl > 0:
                    if count == args.cl:
                        break
                tv.write(line)
                count += 1
    # Clean the dataset
    # CV is cleaned vectors
    if args.clean:
        remove_from_corpus(antonyms_file=args.antonyms, synonyms_file=args.synonyms, corpuse_to_clean=args.tn,
                           output_corpus=args.cv)

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
        distributional=os.path.abspath(args.cv),
        antonyms=os.path.abspath(args.antonyms),
        synonyms=os.path.abspath(args.synonyms),
        aroutput=args.aroutput
    )
    with open(os.path.join(path_to_ar, "config", "ar_config.cfg"), "w") as f:
        f.write(configstring)

    process = subprocess.run('./run_ar.sh',
                   shell=True,
                   stdout=subprocess.PIPE)

    print(process.stdout)


    # Load the initial word vectors
