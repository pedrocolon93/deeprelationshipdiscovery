import argparse

if __name__ == "__main__":
    '''Removes constraints from a vector.txt file'''
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("antonyms", default="/media/pedro/ssd_ext/LEAR/linguistic_constraints/antonyms.txt",
                        help="The constraints for the antonyms")
    parser.add_argument("synonyms", default="/media/pedro/ssd_ext/LEAR/linguistic_constraints/synonyms.txt",
                        help="The constraints for the synonyms")
    parser.add_argument("input_text_vectors", default="/media/pedro/ssd_ext/fasttext/cc.en.300.vec",
                        help="The path for the text file for the input vectors")
    parser.add_argument("output_text_vectors", default="fasttext_seen.txt",
                        help="The path for the text file for the input vectors")
    parser.add_argument("prefix",default="en_",help="The prefix that the constraints have.")

    args = parser.parse_args()
    antonyms_file = args.antonyms
    synonyms_file = args.synonyms
    input_vectors_txt = args.input_text_vectors
    output_vectors_txt = args.output_text_vectors
    constraint_prefix = args.prefix
    modified = set()
    with open(antonyms_file, "r") as ants:
        with open(synonyms_file, "r") as syns:
            print("Antonyms Loaded")
            for idx,antonym in enumerate(ants):
                try:
                    w1, w2 = antonym.split(" ")
                    modified.add(w1.strip())
                    modified.add(w2.strip())
                except: 
                    continue
                    print(idx)
            print("Syns Loaded")
            for idx,antonym in enumerate(syns):
                try: 
                    if idx%10000==0:print(idx)   
                    w1, w2 = antonym.split(" ")
                    modified.add(w1.strip())
               	    modified.add(w2.strip())
                except: continue
            print("Done")
            print(len(modified))
            count=0
            with open(input_vectors_txt) as original_corpus:
                with open(output_vectors_txt, "w") as final_corpus:
                    for i,line in enumerate(original_corpus):
                        if constraint_prefix +line.strip().split(" ")[0].strip() not in modified:
                            continue
                        else:
                            final_corpus.write(line)
                            count+=1
                            if count%1000==0 and count!=0:print(count)
