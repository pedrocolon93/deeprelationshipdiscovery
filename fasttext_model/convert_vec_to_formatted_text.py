
if __name__ == '__main__':
    input_filename = "/Users/pedro/Downloads/wiki-news-300d-1M.vec"
    output_filename = "fasttext_formatted.txt"
    count = 0
    with open(input_filename, 'r') as f:
        with open(output_filename,"w",encoding="utf-8") as vec_file:
            for item in f:
                if count==0:
                    count+=1
                    continue
                count+=1
                # print(item)
                x = item.split(" ")
                lan, word = ("en",x[0])
                # print(lan,word)
                line = lan+"_"+word+" "
                # line = item+" "
                for element in x[1:]:
                    line+=str(element)+" "
                # line+="\n"
                vec_file.write(line)
                if count%10000==0:
                    print(count)
    print("Finished")

