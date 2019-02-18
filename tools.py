import gc
import multiprocessing
import os
from multiprocessing.pool import Pool

from keras import backend as K
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split


def process_line(line):
    linesplit = line.split(" ")
    if len(linesplit)<3:
        return None
    x = np.array([float(x) for x in linesplit[1:]])
    if(np.isnan(x).any()):
        print("Nan!?!??!")
        return None
    tup = (linesplit[0],x)
    return tup

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def load_embedding(path,limit=100000000):
    if os.path.exists(path):
        words = []
        vectors = []
        # f = open(path,encoding="utf-8")
        skip_first = True
        print("Starting loading",path)


            # return "FOO: %s" % line
        #
        pool = Pool(multiprocessing.cpu_count())
        with open(path,encoding="utf-8") as source_file:
            # chunk the work into batches of 4 lines at a time
            results = pool.map(process_line, source_file, multiprocessing.cpu_count())
            for line in results:
                if line is None:
                    continue
                else:
                    words.append(line[0])
                    vectors.append(line[1])
        # for line in f:
        #     if limit is not None:
        #         if(len(words)>=limit):
        #             break
        #     if skip_first ==True:
        #         skip_first = False
        #         continue
        #     linesplit = line.split(" ")
        #     words.append(linesplit[0])
        #     vectors.append([float(x) for x in linesplit[1:]])
        #     # print("Appended")
            pool.close()
            pool.join() 
            print("Finished")
            return words,vectors
    else:
        raise FileNotFoundError(path+" does not exist")


common_vectors_train = None
common_retro_vectors_train = None
common_vectors_test = None
common_retro_vectors_test = None


def load_training_input(limit=10000):
    global common_retro_vectors_train, common_retro_vectors_test, common_vectors_test, common_vectors_train
    words, vectors = load_embedding("retrogan/wiki-news-300d-1M-subword.vec", limit=limit)
    retrowords, retrovectors = load_embedding("retrogan/numberbatch", limit=limit)
    print(len(words), len(retrowords))
    common_vocabulary = set(words).intersection(set(retrowords))
    common_vocabulary = np.array(list(common_vocabulary))
    common_retro_vectors = np.array([retrovectors[retrowords.index(word)] for word in common_vocabulary])
    common_vectors = np.array([vectors[words.index(word)] for word in common_vocabulary])
    X_train, X_test, y_train, y_test = train_test_split(common_vectors, common_retro_vectors, test_size=0.33,
                                                        random_state=42)
    common_vectors_train = X_train
    common_retro_vectors_train = y_train
    common_vectors_test = X_test
    common_retro_vectors_test = y_test
    del retrowords, retrovectors, words, vectors
    print("Size of common vocabulary:" + str(len(common_vocabulary)))
    return common_vocabulary, common_vectors, common_retro_vectors


def load_training_input_2(limit=10000,normalize=True, seed = 42,test_split=0.1):
    global common_retro_vectors_train,common_retro_vectors_test,common_vectors_test,common_vectors_train
    words, vectors = load_embedding("retrogan/wiki-news-300d-1M-subword.vec",limit=limit)
    retrowords, retrovectors =load_embedding("retrogan/numberbatch",limit=limit)
    print(len(words),len(retrowords))
    # common_vocabulary = set(words).intersection(set(retrowords))
    worddict = {}
    for idx, word in enumerate(retrowords):
        worddict[word] = (-1, idx)
    for idx, word in enumerate(words):
        try:
            # try to access and save the index
            retr_idx = worddict[word][1]
            worddict[word] = (idx, retr_idx)
        except:
            # not there so skip
            pass
    common_vocabulary = []
    common_retro_vectors = []
    common_vectors = []
    p = Pool(multiprocessing.cpu_count(),)
    for word in worddict.keys():
        if worddict[word][0] >= 0:
            common_vocabulary.append(word)
            common_retro_vectors.append(retrovectors[worddict[word][1]])
            common_vectors.append(vectors[worddict[word][0]])
    common_vectors = np.array(common_vectors)
    common_retro_vectors = np.array(common_retro_vectors)
    common_vocabulary = np.array(common_vocabulary)
    # common_vocabulary = np.array(list(common_vocabulary))
    # common_retro_vectors = np.array([retrovectors[retrowords.index(word)]for word in common_vocabulary])
    # common_vectors = np.array([vectors[words.index(word)]for word in common_vocabulary])
    if normalize:
        scaler = MinMaxScaler()
        scaler = scaler.fit(common_vectors)
        scaled_common_vector = scaler.transform(common_vectors)

        scaler = MinMaxScaler()
        scaler = scaler.fit(common_retro_vectors)
        scaled_common_retro_vector = scaler.transform(common_retro_vectors)
    else:
        scaled_common_vector = common_vectors
        scaled_common_retro_vector = common_retro_vectors

    X_train, X_test, y_train, y_test = train_test_split(scaled_common_vector, scaled_common_retro_vector, test_size = test_split, random_state = seed)
    common_vectors_train = X_train
    common_retro_vectors_train = y_train
    common_vectors_test = X_test
    common_retro_vectors_test = y_test
    del retrowords,retrovectors,words,vectors
    gc.collect()
    print("Size of common vocabulary:"+str(len(common_vocabulary)))
    return common_vectors_train,common_retro_vectors_train,common_vectors_test,common_retro_vectors_test


def find_word(word,retro=True):
    retrowords,retrovectors = None,None
    if retro:
        retrowords, retrovectors =load_embedding("retrogan/numberbatch",limit=10000000)
    else:
        retrowords, retrovectors =load_embedding("retrogan/wiki-news-300d-1M-subword.vec",limit=10000000)
    for idx, retrovector in enumerate(retrovectors):
        if np.array_equal(word,retrovector):
            print("Found word is ",retrowords[idx])
            del retrowords, retrovectors
            return
    del retrowords, retrovectors
    print("Word not found...")


def find_closest(pred_y,n_top=5,retro=True):
    retrowords,retrovectors = None,None
    if retro:
        retrowords, retrovectors =load_embedding("retrogan/numberbatch",limit=10000000)
    else:
        retrowords, retrovectors =load_embedding("retrogan/wiki-news-300d-1M-subword.vec",limit=10000000)
    # t1 = retrovectors[0].reshape(1,300)
    # t2 = pred_y.reshape(1,300)
    # res = cosine_similarity(t1,t2)
    results = [(idx,item) for idx,item in enumerate(list(map(lambda x: cosine_similarity(x.reshape(1,300), pred_y.reshape(1,300))[0][0],retrovectors)))]
    sorted_results = sorted(results,key=lambda x:x[1],reverse=True)
    for i in range(n_top):
        print(retrowords[sorted_results[i][0]],sorted_results[i][1])
    del retrowords,retrovectors,results,sorted_results
    gc.collect()


def find_in_fasttext(testwords):
    retrowords, retrovectors = load_embedding("retrogan/wiki-news-300d-1M-subword.vec", limit=10000000)
    indices = []
    for word in testwords:
        indices.append(retrowords.index(word))
    results = [retrovectors[idx] for idx in indices]
    return results

def find_in_numberbatch(testwords):
    retrowords, retrovectors = load_embedding("retrogan/numberbatch", limit=10000000)
    indices = []
    for word in testwords:
        indices.append(retrowords.index(word))
    results = [retrovectors[idx] for idx in indices]
    return results
