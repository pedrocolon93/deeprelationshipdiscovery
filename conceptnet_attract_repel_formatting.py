
if __name__ == '__main__':
    antonyms = []
    synonyms = []
    with open("retrogan/conceptnet-assertions-5.6.0.csv") as assertions:
        for assertion in assertions:
            relationship,start,finish = assertion.split("\t")[1:4]
            def rename(start):
                lan_s, word_s = start.split("/")[2:4]
                line_s = lan_s + "_" + word_s + " "
                return line_s

            if "Antonym" in relationship:
                antonyms.append((rename(start),rename(finish)))
            elif "Synonym" in relationship:
                synonyms.append((rename(start),rename(finish)))
            else:
                continue

    with open("antonyms.txt","w") as ants:
        for antonym in antonyms:
            ants.write(str(antonym[0])+"\t"+str(antonym[1])+"\n")
    with open("synonyms.txt", "w") as ants:
        for antonym in synonyms:
            ants.write(str(antonym[0]) + "\t" + str(antonym[1]) + "\n")
