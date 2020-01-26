if __name__ == "__main__":
    modified = set()
    with open("/media/pedro/ssd_ext/LEAR/linguistic_constraints/antonyms.txt","r") as ants:
        with open("/media/pedro/ssd_ext/LEAR/linguistic_constraints/synonyms.txt","r") as syns:
            for antonym in ants:
                w1, w2 = antonym.split(" ")
                modified.add(w1.strip())
                modified.add(w2.strip())
            for antonym in syns:
                w1, w2 = antonym.split(" ")
                modified.add(w1.strip())
                modified.add(w2.strip())
            # print(modified)
            with open("../wiki-news-300d-1M.vec") as original_corpus:
                with open("cleaned_corpus.txt","w") as final_corpus:
                    for line in original_corpus:
                        # print(line)
                        # print(line.split())
                        if "en_"+line.split(" ")[0].strip() not in modified:
                            # print("Skipping")
                            # print(line.split(" ")[1])
                            continue
                        else:
                            final_corpus.write("en_"+line)
                
