import argparse

from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("ov",
                        help="Path to text/vec file for original vectors",
                        )
    parser.add_argument("-ovp",
                        default="en_",
                        help="Original vectors prefix",
                        )
    parser.add_argument("mv",
                        help="Modified vectors",
                        )
    parser.add_argument("-mvp",
                        help="Modified vectors prefix",
                        default="en_"
                        )
    parser.add_argument("output",
                        help="Output vectors")
    parser.add_argument("-op",default="")
    parser.add_argument("-l",default=400000,type=int)
    args = parser.parse_args()

    # embeddings_1 = "/Users/pedro/PycharmProjects/OOVconverter/glove/glove.840B.300d.txt"
    embeddings_1 = args.ov
    embeddings_1_prefix = args.ovp
    print("Opening and preserving the order of",embeddings_1)
    # embeddings_2 = "/Users/pedro/PycharmProjects/OOVconverter/glove_full_alldata/auxgan_ar.txt"
    embeddings_2 = args.mv
    embeddings_2_prefix =args.mvp
    print("Opening and injecting into other embeddings",embeddings_2)
    output_embeddings = args.output
    output_embeddings_prefix = args.op
    modified = {}
    with open(embeddings_2) as f:
        for line in f:
            s = line.split(" ",1)
            name = s[0].replace(embeddings_2_prefix,"")
            content = s[1]
            modified[name]=content
    print("We opened this amount from second embeddings:",len(modified.keys()))
    inject_count = 0
    linecount = 0
    with open(embeddings_1) as f:
        with open(output_embeddings, "w") as output:
            for line in tqdm(f):
                if linecount == args.l:
                    print("Finishing early")
                    break
                s = line.split(" ",1)
                name = s[0].replace(embeddings_1_prefix, "")
                content = s[1]
                if name in modified.keys():
                    output.write(output_embeddings_prefix+name+" "+modified[name])
                    inject_count+=1
                else:
                    output.write(line)
                linecount+=1
    print("Injected", inject_count)