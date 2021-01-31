import argparse
import math
import random

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--target_file', type=str, default="synonyms.txt",
                        help='File that we will reduce')
    parser.add_argument('--percentage_to_leave', type=float, default=0.05,
                        help='Percentage of items to leave')
    parser.add_argument('--seed', type=int, default=42,
                        help='Percentage of items to leave')
    args = parser.parse_args()
    target_file = args.target_file
    percentage_to_leave = args.percentage_to_leave
    seed = args.seed
    random.seed(seed)

    split = target_file.split(".")
    output_file = split[0]+"_cut_to_"+str(percentage_to_leave).replace(".","_")+"."+split[1]
    print("Outputting to",output_file)
    words = []
    with open(target_file) as f:
        for line in tqdm(f):
            s = line.strip().split()
            words.append((s[0],s[1]))
    words = list(set(words))
    random.shuffle(words)
    resulting_words = words[:int(math.floor(percentage_to_leave*len(words)))]
    with open(output_file,"w") as of:
        for tup in resulting_words:
            of.write(tup[0]+"\t"+tup[1]+'\n')
