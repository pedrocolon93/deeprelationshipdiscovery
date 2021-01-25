import fasttext
from tqdm import tqdm


def open_constraints(filename):
    print("Searching in ",filename)
    words = []
    with open(filename) as f:
        for line in tqdm(f):
            line = line.strip().split()
            if len(line) >= 2:
                words.append(line[0].replace("en_",""))
                words.append(line[1].replace("en_",""))
    return list(set(words))




if __name__ == '__main__':
    ft = fasttext.load_model("fasttext_model/cc.en.300.bin")
    words = []
    words += open_constraints("synonyms.txt")
    words += open_constraints("antonyms.txt")
    words += open_constraints("testing/simlexorig999.txt")
    words += open_constraints("testing/simverb3500.txt")
    words += open_constraints("testing/card660.tsv")
    words = list(set(words))
    with open("ft_all_unseen.txt","w") as outfile:
        for word in tqdm(words):
            outfile.write(' '.join(["en_"+word]+[str(x) for x in ft.get_word_vector(word)]+["\n"]))
