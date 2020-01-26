if __name__ == "__main__":
    slsv_words = ""
    with open("./simlexsimverb.words") as f:
        for line in f:
            slsv_words+=line+"\n"
    def clean_constraints(filename):
        ants = []
        with open(filename) as f:
            for antonym in f:
                w1, w2 = antonym.split(" ")
                if w1.strip() in slsv_words or w2.strip() in slsv_words:
                    continue
                else:
                    ants.append(antonym)
        return ants
    ants = clean_constraints("/media/pedro/ssd_ext/LEAR/linguistic_constraints/antonyms.txt")
    with open("clean_ant","w") as f:
        f.writelines(ants)   
    syns = clean_constraints("/media/pedro/ssd_ext/LEAR/linguistic_constraints/synonyms.txt")         
    with open("clean_syn.txt","w") as f:
        f.writelines(syns)       
