import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("outputtext",default="fasttext_formatted_clean.txt",
                        help="The output text file that is filtered")
    parser.add_argument("targettext",
                        help="Text vectors that will be filtered out",
                        default="fasttext_formatted.txt")
    args = parser.parse_args()

    lines = ""
    with open(args.outputtext,"w") as output_vectors:
        with open(args.targettext) as input_vectors:
            count = 0
            for line in tqdm(input_vectors):
                line = line.replace("en_","")
                count+=1
                lines+=line
                if count%10000==0:
                    print(count)
            output_vectors.write(lines)