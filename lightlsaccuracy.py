import os

if __name__ == '__main__':
    path_to_light_ls = "/Users/pedro/Downloads/lex.mturk.txt"
    path_to_light_ls_output = "/Users/pedro/Documents/git/lightls/testdir/"
    dataset = dict()
    count = 0
    with open(path_to_light_ls,'r',encoding="latin-1") as lightls:
        for line in lightls:
            dataset[str(count)] = (line.split("\t")[1],set(line.split("\t")[2:])) # word to substityte, substitutions
            count+=1

    print(dataset)
    correct = 0
    total = 0
    incorrect = 0
    for i in range(count):
        with open(os.path.join(path_to_light_ls_output,str(i)+".subs")) as substitutions:
            for line in substitutions:
                index, subsituted_word, substitution = line.split()
                if subsituted_word == dataset[str(i)][0]:
                    if substitution in dataset[str(i)][1]:
                        correct+=1
                    else:
                        print("*"*100)
                        print("Error",line,dataset[str(i)],i)
                        print("*"*100)

                        incorrect+=1
        total+=1
    print(correct,incorrect,total,100*(correct/total*1.0))