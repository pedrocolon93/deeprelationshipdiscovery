from tqdm import tqdm


def remove_constraints_from_corpus(sl_sv_words="simlexsimverb.words",
                       corpuse_to_clean="../../glove/glove_formatted.txt", output_corpus="cleaned_corpus_glove.txt",
                       prefix="en_"):
    modified = set()
    count=0
    with open(sl_sv_words, "r") as ants:
        for antonym in tqdm(ants):
            w1 = antonym
            modified.add(w1.strip())
        # print(modified)
        with open(corpuse_to_clean) as original_corpus:
            with open(output_corpus, "w") as final_corpus:
                for i, line in enumerate(original_corpus):
                    x = line.strip().split(" ")[0].strip()
                    if prefix not in x:
                        x = prefix + x
                    if x in modified:
                        # print("Skipping")
                        # print(line.split(" ")[1])
                        continue
                    else:
                        if prefix not in line:
                            line = prefix + line
                        final_corpus.write(line)
                        count+=1
    print(count)

def remove_from_corpus(antonyms_file="antonyms.txt", synonyms_file="synonyms.txt",
                       corpuse_to_clean="../../glove/glove_formatted.txt", output_corpus="cleaned_corpus_glove.txt",
                       separator=" ",
                       prefix="en_"):
    modified = set()
    count = 0
    with open(antonyms_file, "r") as ants:
        with open(synonyms_file, "r") as syns:
            for antonym in tqdm(ants):
                w1, w2 = antonym.split(separator)
                modified.add(w1.strip())
                modified.add(w2.strip())
            for antonym in tqdm(syns):
                w1, w2 = antonym.split(separator)
                modified.add(w1.strip())
                modified.add(w2.strip())
            # print(modified)
            with open(corpuse_to_clean) as original_corpus:
                with open(output_corpus, "w") as final_corpus:
                    for i, line in enumerate(original_corpus):
                        # print(line)
                        # print(line.split())
                        x =line.strip().split(" ")[0].strip()
                        if prefix not in x:
                            x = prefix+x
                        if x not in modified:
                            # print("Skipping")
                            # print(line.split(" ")[1])
                            continue
                        else:
                            if prefix not in line:
                                line = prefix+line
                            final_corpus.write(line)
                            count+=1
    print(count)

def remove_from_corpus_2(antonyms_file="antonyms.txt", synonyms_file="synonyms.txt", sl_sv_words="simlexsimverb.words",
                       corpuse_to_clean="../../glove/glove_formatted.txt", output_corpus="cleaned_corpus_glove.txt",
                       separator=" ",
                       prefix="en_"):
    modified = set()
    count = 0
    with open(antonyms_file, "r") as ants:
        with open(synonyms_file, "r") as syns:
            with open(sl_sv_words,"r") as slsv:
                for antonym in tqdm(ants):
                    w1, w2 = antonym.split(separator)
                    modified.add(w1.strip())
                    modified.add(w2.strip())
                for antonym in tqdm(syns):
                    w1, w2 = antonym.split(separator)
                    modified.add(w1.strip())
                    modified.add(w2.strip())
                for slsvw in tqdm(slsv):
                    w1= slsvw.strip()
                    modified.add(w1)
                # print(modified)
                with open(corpuse_to_clean) as original_corpus:
                    with open(output_corpus, "w") as final_corpus:
                        for i, line in enumerate(original_corpus):
                            # print(line)
                            # print(line.split())
                            x =line.strip().split(" ")[0].strip()
                            if prefix not in x:
                                x = prefix+x
                            if x not in modified:
                                # print("Skipping")
                                # print(line.split(" ")[1])
                                continue
                            else:
                                if prefix not in line:
                                    line = prefix+line
                                final_corpus.write(line)
                                count+=1
    print(count)
if __name__ == "__main__":
    modified = set()
    # remove_from_corpus()
    remove_constraints_from_corpus(sl_sv_words="../vocab/simlexsimverb.words", corpuse_to_clean="/media/pedro/ssd_ext/attract-repel/results/glove_ar_disjoint.txt",output_corpus="/media/pedro/ssd_ext/attract-repel/results/gloveclean_ar_disjoint.txt")
