import fasttext
from tqdm import tqdm
if __name__ == '__main__':
    ## Read all NumberBatch data as a Python dict
    skip_first=False
    words = set()
    with open('testing/card660.tsv', 'r') as f:
        for x in tqdm(f):
            if skip_first:
                skip_first=False
                continue
            [words.add(y) for y in x.split("\t")[0:2]]

    ft_model = fasttext.load_model("fasttext_model/cc.en.300.bin")
    with open("card_ft_vecs.txt","w") as f:
        for token in tqdm(words):
            vals = ft_model.get_word_vector(token.replace("_"," "))
            str_vals = list(map(str,vals))
            f.write(' '.join(["en_"+str(token.replace(" ","_"))]+str_vals+["\n"]))