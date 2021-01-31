import argparse
import math
import random
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--target_file', type=str, default="testing/SimLex-999.txt",
                        help='File that we will reduce')
    parser.add_argument('--percentage_to_leave', type=float, default=0.05,
                        help='Percentage of items to leave')
    parser.add_argument('--seed', type=int, default=42,
                        help='Percentage of items to leave')
    parser.add_argument('--output_dir', default="./",
                        help='Percentage of items to leave')

    args = parser.parse_args()
    target_file = args.target_file
    percentage_to_leave = args.percentage_to_leave
    seed = args.seed
    random.seed(seed)
    output_dir = args.output_dir
    split = target_file.split(".")
    split[0] = split[0].split("/")[-1] if "/" in split[0] else split[0]
    output_file = split[0]+"_cut_to_"+str(percentage_to_leave).replace(".","_")+"."+split[1]
    print("Outputting to",output_dir,output_file)
    words = []
    with open(target_file) as f:
        for line in f:
            s = line.strip().split()
            words.append(s[0])
            try:
                words.append(s[1])
            except:
                pass
    words = sorted(list(set(words)))
    random.Random(seed).shuffle(words)
    resulting_words = words[:int(math.floor(percentage_to_leave*len(words)))]
    with open(os.path.join(output_dir,output_file),"w") as of:
        for line in resulting_words:
            of.write(line+"\n")
