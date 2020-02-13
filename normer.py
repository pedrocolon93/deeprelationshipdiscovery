from tqdm import tqdm
import numpy as np
import scipy.stats as s
if __name__ == '__main__':
    input_filename = "/Users/pedro/Downloads/wiki-news-300d-1M-subword.vec"
    comparison_name = "/Users/pedro/Documents/git/adversarial-postspec/post-specialized embeddings/distrib/ft_distrib.txt"
    skip_first = True
    count = 0
    limit = 20000
    prefix = "en_"
    indexes = []
    vectors = []
    with open(input_filename,encoding="utf-8") as vec_file:
        for line in tqdm(vec_file):
            count+=1
            if skip_first:
                skip_first=False
                continue
            if count == limit:
                print("Reached limit",limit)
                break
            word = line.strip().split(" ")[0]
            word = prefix+word
            vec = []
            for element in line.strip().split(" ")[1:]:
                vec.append(float(element))
            indexes.append(word)
            v = np.array(vec)
            v = v/np.linalg.norm(v)
            vectors.append(v)
            if count%10000==0:
                print(count)

    f = np.array(vectors)
    # f = (f - np.average(vectors))/np.std(vectors)

    print(f)


