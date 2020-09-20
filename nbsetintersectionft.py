import fasttext
from tqdm import tqdm
if __name__ == '__main__':
    numberbatch_tokens = []
    ## Read all NumberBatch data as a Python dict
    skip_first=True
    with open('nb_retrofitted_clean.txt',"w") as wf:
        with open('numberbatch-en-19.08.txt', 'r') as f:
            for x in tqdm(f):
                if skip_first:
                    skip_first=False
                    continue
                wf.write("en_"+x.strip()+"\n")
                numberbatch_tokens.append( x.strip().split(' ')[0].replace("_"," "))

    ft_model = fasttext.load_model("fasttext_model/cc.en.300.bin")
    with open("ft_seen_nb_align.txt","w") as f:
        f.write(str(len(numberbatch_tokens))+" 300\n")

        for token in tqdm(numberbatch_tokens):
            vals = ft_model.get_word_vector(token)
            str_vals = list(map(str,vals))
            f.write(' '.join([str("en_"+token.replace(" ","_"))]+str_vals+["\n"]))