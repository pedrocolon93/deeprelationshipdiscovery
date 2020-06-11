if __name__ == "__main__":
    modified = set()
    with open("antonyms.txt","r") as ants:
        with open("synonyms.txt","r") as syns:
            print("Antonyms")
            for idx,antonym in enumerate(ants):
                #print(antonym)
                try:
                    w1, w2 = antonym.split(" ")
                    modified.add(w1.strip())
                    modified.add(w2.strip())
                except: continue
            print("Syns")
            for idx,antonym in enumerate(syns):
                try: 
                    w1, w2 = antonym.split(" ")
                    modified.add(w1.strip())
               	    modified.add(w2.strip())
                except: continue
            print("Done")
            print(len(modified))
            count=0
            with open("glove_orginal_cut.txt") as original_corpus:
                with open("glove_original_cut_onlyconstraints.txt","w") as final_corpus:
                    for i,line in enumerate(original_corpus):
                        #print(line)
                        #print(line.split())
                        if "en_"+line.strip().split(" ")[0].strip() not in modified:
                            # print("Skipping")
                            # print(line.split(" ")[1])
                            continue
                        else:
                            final_corpus.write(line)
                            count+=1
                            if count%1000==0 and count!=0:print(count)
                
