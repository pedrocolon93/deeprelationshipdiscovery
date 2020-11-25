import csv
import math
import multiprocessing
import operator
import os
import pickle
from multiprocessing.pool import Pool
from tokenize import String

import faiss
import fasttext
import gc

import numpy
import numpy as np
import pandas as pd
from conceptnet5.nodes import standardized_concept_uri
from conceptnet5.vectors import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

directory = './retrogan/'
# directory = '/home/pedro/Documents/mltests/retrogan/'
# directory = '/home/conceptnet/conceptnet5/data/vectors/'
# original = 'fasttext-opensubtitles.h5'
original = 'crawl-300d-2M.h5'
# retrofitted = 'fasttext-opensubtitles-retrofit.h5'
retrofitted = 'crawl-300d-2M-retrofit.h5'


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

# def root_mean_squared_error(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true)))

def load_embedding(path,limit=100000000,cache=None):
    if cache is not None:
        words,vectors = pickle.load(cache)
        return words,vectors
    if os.path.exists(path):
        words = []
        vectors = []
        # f = open(path,encoding="utf-8")
        skip_first = True
        print("Starting loading",path)
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
        pool.close()
        pool.join()
        print("Finished")
        if cache is not None:
            pickle.dump((words,vectors),cache)
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

datasets = {
    'fasttext':['fasttext-opensubtitles.h5','fasttext-opensubtitles-retrofit.h5'],
    'crawl':['crawl-300d-2M.h5','crawl-300d-2M-retrofit.h5'],
    'w2v':['w2v-google-news.h5','w2v-google-news-retrofit.h5'],
    'numberbatch':['mini.h5','mini.h5'],
    'retrogan':["retroembeddings.h5","retroembeddings.h5"]
}
def load_training_input_3(seed=42,test_split=0.1,dataset="fasttext"):

    global original,retrofitted
    original,retrofitted = datasets[dataset]
    print("Searching in")
    print(directory)
    print("for:", original, retrofitted)

    o = pd.read_hdf(directory + original, 'mat', encoding='utf-8')

    # print(asarray1.shape)
    # print(o["index"][0])
    # print(o["columns"][0][:])
    r = pd.read_hdf(directory + retrofitted, 'mat', encoding='utf-8')
    r_sub = r.loc[o.index.intersection(r.index), :]
    del r
    gc.collect()
    # print(asarray2.shape)
    X_train, X_test, y_train, y_test = train_test_split(o.values, r_sub.values, test_size = test_split, random_state = seed)
    common_vectors_train = X_train
    common_retro_vectors_train = y_train
    common_vectors_test = X_test
    common_retro_vectors_test = y_test
    # del retrowords, retrovectors, words, vectors
    del o
    gc.collect()
    # print("Size of common vocabulary:" + str(len(common_vocabulary)))
    return common_vectors_train, common_retro_vectors_train, common_vectors_test, common_retro_vectors_test

def mult(tup):
    return tup[0]*tup[1]
thread_amount = 8
pool = Pool(thread_amount)

def dot_product2_mp(v1, v2):
    it = [x for x in zip(list(v1.reshape(dimensionality)), list(v2.reshape(dimensionality)))]

    res = pool.map(mult,it)
    # pool.join()
    res = sum(res)
    # print(res)
    return res

def dot_product2(v1,v2):
    return sum(map(operator.mul,v1.reshape(dimensionality),v2.reshape(dimensionality)))

def vector_cos5(v1, v2):
    prod = dot_product2(v1, v2)
    len1 = math.sqrt(dot_product2(v1, v1))
    len2 = math.sqrt(dot_product2(v2, v2))
    return prod / (len1 * len2)

def load_noisiest_words(seed=42, test_split=0.1, dataset="fasttext"):
    global original, retrofitted
    if os.path.exists("filtered_x") and os.path.exists("filtered_y") and \
            os.path.exists("filtered_x_test") and os.path.exists("filtered_x_test"):
        print("Reusing cache")
        X_train = pd.read_hdf("filtered_x", 'mat', encoding='utf-8')
        Y_train = pd.read_hdf("filtered_y",'mat',encoding='utf-8')
        X_test = pd.read_hdf("filtered_x_test", "mat",encoding='utf-8')
        Y_test = pd.read_hdf("filtered_y_test", "mat",encoding='utf-8')

        return np.array(X_train.values), np.array(Y_train.values), np.array(X_test.values), np.array(Y_test.values)

    original, retrofitted = datasets[dataset]
    print("Searching in")
    print(directory)
    print("for:", original, retrofitted)

    o = pd.read_hdf(directory + original, 'mat', encoding='utf-8')
    r = pd.read_hdf(directory + retrofitted, 'mat', encoding='utf-8')
    r_sub = r.loc[o.index.intersection(r.index), :]
    del r
    gc.collect()

    # Filter out words that have not changed too much.
    cns = []

    print(len(o.values))
    testindexes = []
    for i in range(len(o.values)):
        if i%100000==0 and i!=0:
            # break
            print(i)
        x = o.iloc[i,:]
        y = r_sub.iloc[i,:]
        cn = cosine_similarity(x, y)
        if cn<0.95:
            cns.append(i)
        else:
            testindexes.append(i)
    X_train = o.iloc[cns, :]
    Y_train = r_sub.iloc[cns, :]
    X_train.to_hdf("filtered_x", "mat")
    Y_train.to_hdf("filtered_y","mat")
    X_test = o.iloc[testindexes,:]
    Y_test = r_sub.iloc[testindexes,:]
    X_test.to_hdf("filtered_x_test", "mat")
    Y_test.to_hdf("filtered_y_test", "mat")
    return np.array(X_train.values),np.array(Y_train.values), np.array(X_test.values),np.array(Y_test.values)

def load_noisiest_words_dataset(dataset, seed=42, test_split=0.1, save_folder="./", cache=True, threshold = 0.95,return_idx=False):
    global original, retrofitted
    # if os.path.exists(os.path.join(save_folder,"filtered_x")) and os.path.exists(os.path.join(save_folder,"filtered_y"))\
    #         and cache:
    #     print("Reusing cache")
    #     X_train = pd.read_hdf(os.path.join(save_folder,"filtered_x"), 'mat', encoding='utf-8')
    #     Y_train = pd.read_hdf(os.path.join(save_folder,"filtered_y"),'mat',encoding='utf-8')
    #     # X_test = pd.read_hdf(os.path.join(save_folder,"filtered_x_test"), "mat",encoding='utf-8')
    #     # Y_test = pd.read_hdf(os.path.join(save_folder,"filtered_y_test"), "mat",encoding='utf-8')
    #     if not return_idx:
    #         return np.array(X_train.values), np.array(Y_train.values)#, np.array(X_test.values), np.array(Y_test.values)
    #     else:
    #         return np.array(X_train.values), np.array(Y_train.values), np.array(X_train.index)
    original = dataset["original"]
    retrofitted = dataset["retrofitted"]
    directory = dataset["directory"]
    print("Searching")
    print("for:", original, retrofitted)

    o = pd.read_hdf(directory + original, 'mat', encoding='utf-8')
    r = pd.read_hdf(directory + retrofitted, 'mat', encoding='utf-8')
    r_sub = r.loc[o.index.intersection(r.index), :]
    del r
    gc.collect()

    # Filter out words that have not changed too much.
    cns = []

    def dot_product2(v1, v2):
        return sum(map(operator.mul, v1, v2))

    def vector_cos5(v1, v2):
        prod = dot_product2(v1, v2)
        len1 = math.sqrt(dot_product2(v1, v1))
        len2 = math.sqrt(dot_product2(v2, v2))
        if len1*len2==0:
            return prod/((len1*len2)+1)
        return prod / (len1 * len2)

    print(len(o.values))
    testindexes = []
    tot = 0
    it = 0
    cns = []
    for i in range(len(o.values)):
        if i%10000==0 and i!=0:
            print(i)
        x = o.iloc[i,:]
        y = r_sub.iloc[i,:]
        cn = vector_cos5(x, y)
        cns.append(cn)
    threshold = np.average(np.array(cns))-np.std(np.array(cns))/2.0
    tot = 0
    it = 0
    cns = []
    for i in range(len(o.values)):
        if i%10000==0 and i!=0:
            print(len(cns),len(testindexes))
            print(i)
        x = o.iloc[i,:]
        y = r_sub.iloc[i,:]
        cn = vector_cos5(x, y)
        tot+=cn
        it+=1
        if cn<threshold:
            cns.append(i)
        else:
            testindexes.append(i)
    print("avg threshold",tot/it)
    X_train = o.iloc[cns, :]
    Y_train = r_sub.iloc[cns, :]
    print("Dumping training")
    X_train.to_hdf(os.path.join(save_folder,"filtered_x"), "mat")
    Y_train.to_hdf(os.path.join(save_folder,"filtered_y"),"mat")

    X_test = o.iloc[testindexes,:]
    Y_test = r_sub.iloc[testindexes,:]
    print("Dumping testing")
    X_test.to_hdf(os.path.join(save_folder,"filtered_x_test"), "mat")
    Y_test.to_hdf(os.path.join(save_folder,"filtered_y_test"), "mat")
    print("Returning")
    if not return_idx:
        return np.array(X_train.values),np.array(Y_train.values)#, np.array(X_test.values),np.array(Y_test.values)
    else:
        return np.array(X_train.values),np.array(Y_train.values),np.array(X_train.index)#, np.array(X_test.values),np.array(Y_test.values)
def load_all_words_dataset(dataset, seed=42, test_split=0.1, save_folder="./", cache=True, threshold = 0.95, return_idx=False):
    global original, retrofitted
    xpath = os.path.join(save_folder,"filtered_x")
    ypath = os.path.join(save_folder,"filtered_y")
    if cache:
        print("Reusing cache")
        X_train = pd.read_hdf(os.path.join(save_folder,"filtered_x"), 'mat', encoding='utf-8')
        Y_train = pd.read_hdf(os.path.join(save_folder,"filtered_y"),'mat',encoding='utf-8')
        # X_test = pd.read_hdf(os.path.join(save_folder,"filtered_x_test"), "mat",encoding='utf-8')
        # Y_test = pd.read_hdf(os.path.join(save_folder,"filtered_y_test"), "mat",encoding='utf-8')
        if not return_idx:
            return np.array(X_train.values), np.array(Y_train.values)#, np.array(X_test.values), np.array(Y_test.values)
        else:
            return np.array(X_train.values), np.array(Y_train.values), np.array(X_train.index)
    original = dataset["original"]
    retrofitted = dataset["retrofitted"]
    directory = dataset["directory"]
    print("Searching")
    print("for:", original, retrofitted)

    o = pd.read_hdf(directory + original, 'mat', encoding='utf-8')
    r = pd.read_hdf(directory + retrofitted, 'mat', encoding='utf-8')
    r_sub = r.loc[o.index.intersection(r.index), :]
    del r
    gc.collect()

    # Filter out words that have not changed too much.
    cns = []

    def dot_product2(v1, v2):
        return sum(map(operator.mul, v1, v2))

    # print(len(o.values))
    testindexes = []

    cns = []

    def load_words(fn):
        syns = set()
        with open(fn) as f:
            for line in f:
                for x in line.split("\t"):
                    s = standardized_concept_uri(x[0:x.index("_")], x[x.index("_"):])
                    # print(s)
                    syns.add(s)
        return syns
    # result = load_words("synonyms.txt").union(load_words("antonyms.txt"))
    print("Loading concepts")
    for i in tqdm(range(len(o.values))):
        # print(o.index[i])
        # if o.index[i] in result:
        cns.append(i)
        # else:
        #     testindexes.append(i)
    X_train = o.iloc[cns, :]
    Y_train = r_sub.iloc[cns, :]
    print("Dumping training")
    X_train.to_hdf(os.path.join(save_folder,"filtered_x"), "mat")
    Y_train.to_hdf(os.path.join(save_folder,"filtered_y"),"mat")

    # X_test = o.iloc[testindexes,:]
    # Y_test = r_sub.iloc[testindexes,:]
    # print("Dumping testing")
    # X_test.to_hdf(os.path.join(save_folder,"filtered_x_test"), "mat")
    # Y_test.to_hdf(os.path.join(save_folder,"filtered_y_test"), "mat")
    print("Returning")
    if not return_idx:
        return np.array(X_train.values),np.array(Y_train.values)#, np.array(X_test.values),np.array(Y_test.values)
    else:
        return np.array(X_train.values),np.array(Y_train.values),np.array(X_train.index)#, np.array(X_test.values),np.array(Y_test.values)
def load_all_words_dataset_final(original,retrofitted, save_folder="./", cache=True, return_idx=False):
    if cache:
        print("Reusing cache")
        X_train = pd.read_hdf(os.path.join(save_folder,"filtered_x"), 'mat', encoding='utf-8')
        Y_train = pd.read_hdf(os.path.join(save_folder,"filtered_y"),'mat',encoding='utf-8')

        if not return_idx:
            return np.array(X_train.values), np.array(Y_train.values)#, np.array(X_test.values), np.array(Y_test.values)
        else:
            return np.array(X_train.values), np.array(Y_train.values), np.array(X_train.index)

    print("Searching")
    print("for:", original, retrofitted)
    o = pd.read_hdf(original, 'mat', encoding='utf-8')
    r = pd.read_hdf(retrofitted, 'mat', encoding='utf-8')
    cns = []
    print("Loading concepts")
    for i in tqdm(r.index):
        cns.append(i)
    X_train = o.loc[cns, :]
    Y_train = r.loc[cns, :]
    print("Dumping training")

    X_train.to_hdf(os.path.join(save_folder,"filtered_x"), "mat")
    Y_train.to_hdf(os.path.join(save_folder,"filtered_y"),"mat")

    print("Returning")
    return X_train,Y_train

def load_all_words_dataset_3(dataset, seed=42, test_split=0.1, save_folder="./", cache=True, threshold = 0.95, return_idx=False,remove_constraint=None):
    global original, retrofitted
    xpath = os.path.join(save_folder,"filtered_x")
    ypath = os.path.join(save_folder,"filtered_y")

    def remove_constraint_fun(X_train, Y_train):
        print("Removing words in constraints!")
        print(X_train)
        print(Y_train)
        with open(remove_constraint) as constraints:
            for line in constraints:
                try:
                    X_train.drop(line.strip(),inplace=True)
                    Y_train.drop(line.strip(),inplace=True)
                except:
                    print(line,"not found")
        print(X_train)
        print(Y_train)
        return X_train, Y_train

    if cache:
        print("Reusing cache")
        X_train = pd.read_hdf(os.path.join(save_folder,"filtered_x"), 'mat', encoding='utf-8')
        Y_train = pd.read_hdf(os.path.join(save_folder,"filtered_y"),'mat',encoding='utf-8')
        # X_test = pd.read_hdf(os.path.join(save_folder,"filtered_x_test"), "mat",encoding='utf-8')
        # Y_test = pd.read_hdf(os.path.join(save_folder,"filtered_y_test"), "mat",encoding='utf-8')

        if remove_constraint is not None:
            X_train,Y_train = remove_constraint_fun(X_train,Y_train)
        if not return_idx:
            return np.array(X_train.values), np.array(Y_train.values)#, np.array(X_test.values), np.array(Y_test.values)
        else:
            return np.array(X_train.values), np.array(Y_train.values), np.array(X_train.index)
    original = dataset["original"]
    retrofitted = dataset["retrofitted"]
    directory = dataset["directory"]
    print("Searching")
    print("for:", original, retrofitted)

    o = pd.read_hdf(directory + original, 'mat', encoding='utf-8')
    r = pd.read_hdf(directory + retrofitted, 'mat', encoding='utf-8')
    cns = []


    print("Loading concepts")
    # o = o.swapaxes(0,1)
    # r = r.swapaxes(0,1)
    for i in tqdm(r.index):
        cns.append(i)
    X_train = o.loc[cns, :]
    Y_train = r.loc[cns, :]
    print("Dumping training")
    if remove_constraint is not None:
        X_train,Y_train = remove_constraint_fun(X_train,Y_train)

    X_train.to_hdf(os.path.join(save_folder,"filtered_x"), "mat")
    Y_train.to_hdf(os.path.join(save_folder,"filtered_y"),"mat")

    # X_test = o.iloc[testindexes,:]
    # Y_test = r_sub.iloc[testindexes,:]
    # print("Dumping testing")
    # X_test.to_hdf(os.path.join(save_folder,"filtered_x_test"), "mat")
    # Y_test.to_hdf(os.path.join(save_folder,"filtered_y_test"), "mat")
    print("Returning")
    return X_train,Y_train

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
        retrowords, retrovectors =load_embedding("retrogan/numberbatch",limit=10000000,cache='./numberbatch')
    else:
        retrowords, retrovectors =load_embedding("retrogan/wiki-news-300d-1M-subword.vec",limit=10000000, cache='./fasttext')
    for idx, retrovector in enumerate(retrovectors):
        if np.array_equal(word,retrovector):
            print("Found word is ",retrowords[idx])
            del retrowords, retrovectors
            return
    del retrowords, retrovectors
    print("Word not found...")


def find_cross_closest(vec1, vec2, n_top, closest=0,verbose=False):
    #TODO SKIP THE NEIGHBORS OF THE ONE WE DO NOT COMPARE AGAINST
    results = []
    if closest == 0:
        #explore the dot between vec1 and synonyms of vec2
        closest_words, closest_vecs = find_closest_2(vec2,n_top=n_top,dataset='numberbatch')
        results = [(idx, item) for idx, item in enumerate(
            list(map(lambda x: cosine_similarity( np.array(x).reshape(1, dimensionality), np.array(vec1).reshape(1, dimensionality))[0][0], closest_vecs)))]
    else:
        #explore the dot between vec2 and synonyms of vec1
        closest_words, closest_vecs = find_closest_2(vec1,n_top=n_top,dataset='numberbatch')
        results = [(idx, item) for idx, item in enumerate(
            list(map(lambda x: cosine_similarity(np.array(x).reshape(1, dimensionality), np.array(vec2).reshape(1, dimensionality))[0][0], closest_vecs)))]
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    final_n_results = []
    final_n_results_words = []
    final_n_results_weights = []
    for i in range(len(sorted_results)):
        if len(final_n_results)==n_top:
            print("Full")
            break
        # i += skip
        if verbose:
            print(closest_words[sorted_results[i][0]], sorted_results[i][1])
        final_n_results_words.append(closest_words[sorted_results[i][0]])
        final_n_results.append(np.array(closest_vecs[sorted_results[i][0]]))
        final_n_results_weights.append(sorted_results[i][1])

    # return final_n_results_words,final_n_results,final_n_results_weights
    return final_n_results_words,final_n_results

def find_cross_closest_dataset(vec1, vec2, projection_count=10,projection_cloud_count=20,n_top=200 ,closest=0,verbose=False,dataset=None):
    # TODO SKIP THE NEIGHBORS OF THE ONE WE DO NOT COMPARE AGAINST
    results = []
    # find the projection of one concept unto the other
    if closest == 0:
        # explore the dot between vec1 and synonyms of vec2
        closest_words, closest_vecs = find_closest_in_dataset(vec2, n_top=n_top, dataset=dataset)

        results = [(idx, item) for idx, item in enumerate(
            list(map(lambda x: cosine_similarity(np.array(x).reshape(1, dimensionality),
                                                 np.array(vec1).reshape(1, dimensionality))[0][0],
                     closest_vecs)))]
    else:
        # explore the dot between vec2 and synonyms of vec1
        closest_words, closest_vecs = find_closest_in_dataset(vec1, n_top=n_top, dataset=dataset)
        results = [(idx, item) for idx, item in enumerate(
            list(map(lambda x: cosine_similarity(np.array(x).reshape(1, dimensionality),
                                                 np.array(vec2).reshape(1, dimensionality))[0][0],
                     closest_vecs)))]
    # list of ordered projections this result is the cosine distance ordering of the projection of one concept
    # unto the cloud of concepts of the other
    # c1 ->  #cloud c2#
    # Has the amount of the cloud in c2
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    final_n_results = []
    final_n_results_words = []

    # print("Finding the cloud of concepts around the cross closest")
    for res_idx, res in enumerate(sorted_results):
        if res_idx > projection_count:
            break
        if closest_words[sorted_results[res_idx][0]] in final_n_results_words:
            projection_count += 1
            continue

        final_n_results_words.append(closest_words[sorted_results[res_idx][0]])
        final_n_results.append(np.array(closest_vecs[sorted_results[res_idx][0]]))
        # now add the ones around that
        fin_aroud_vec = np.array(closest_vecs[sorted_results[res_idx][0]])
        # the projection cloud
        projection_cloud_words, projection_cloud_vecs = find_closest_in_dataset(fin_aroud_vec,
                                                                                n_top=projection_cloud_count,
                                                                                dataset=dataset,
                                                                                )
        ordered_projection_results = [(idx, item) for idx, item in enumerate(
            list(map(lambda x: cosine_similarity(np.array(x).reshape(1, dimensionality),
                                                 fin_aroud_vec.reshape(1, dimensionality))[0][0],
                     projection_cloud_vecs)))]
        for pcres_idx, pcres in enumerate(ordered_projection_results):
            # print(res_idx)
            # print(sorted_results[res_idx])
            # print("Working in", closest_words[sorted_results[res_idx][0]])
            # print(res, "Cloud concept")
            # print(ordered_projection_results[pcres_idx])
            # print(projection_cloud_words[ordered_projection_results[pcres_idx][0]])
            # print(pcres)
            if projection_cloud_words[ordered_projection_results[pcres_idx][0]] not in final_n_results_words:
                final_n_results_words.append(projection_cloud_words[ordered_projection_results[pcres_idx][0]])
                final_n_results.append(np.array(projection_cloud_vecs[ordered_projection_results[pcres_idx][0]]))
    # if verbose: print("Finally", final_n_results_words)

    # return final_n_results_words,final_n_results,final_n_results_weights
    return final_n_results_words, final_n_results

def find_cross_closest_2(vec1, vec2, projection_count=10,projection_cloud_count=20,n_top=200 ,closest=0,verbose=False):
    #TODO SKIP THE NEIGHBORS OF THE ONE WE DO NOT COMPARE AGAINST
    results = []
    #find the projection of one concept unto the other
    if closest == 0:
        #explore the dot between vec1 and synonyms of vec2
        closest_words, closest_vecs = find_closest_2(vec2,n_top=n_top,dataset='numberbatch',keep_o=True)
        results = [(idx, item) for idx, item in enumerate(
            list(map(lambda x: cosine_similarity( np.array(x).reshape(1, 300), np.array(vec1).reshape(1, 300))[0][0], closest_vecs)))]
    else:
        #explore the dot between vec2 and synonyms of vec1
        closest_words, closest_vecs = find_closest_2(vec1,n_top=n_top,dataset='numberbatch',keep_o=True)
        results = [(idx, item) for idx, item in enumerate(
            list(map(lambda x: cosine_similarity(np.array(x).reshape(1, 300), np.array(vec2).reshape(1, 300))[0][0], closest_vecs)))]
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    final_n_results = []
    final_n_results_words = []
    final_n_results_weights = []
    print("Finding the cloud of concepts around the cross closest")
    for res_idx, res in enumerate(sorted_results):
        if res_idx>projection_count:
            break
        if closest_words[sorted_results[res_idx][0]] in final_n_results_words:
            projection_count+=1
            continue
        final_n_results_words.append(closest_words[sorted_results[res_idx][0]])
        final_n_results.append(np.array(closest_vecs[sorted_results[res_idx][0]]))
        # now add the ones around that
        fin_aroud_vec = np.array(closest_vecs[sorted_results[res_idx][0]])
        #the projection cloud
        projection_cloud_words, projection_cloud_vecs = find_closest_2(fin_aroud_vec, n_top=projection_cloud_count, dataset='numberbatch',keep_o=True)
        ordered_projection_results = [(idx, item) for idx, item in enumerate(
            list(map(lambda x: cosine_similarity(np.array(x).reshape(1, 300), fin_aroud_vec.reshape(1, 300))[0][0],
                     projection_cloud_vecs)))]
        for pcres_idx, pcres in enumerate(ordered_projection_results):
            print("Working in",closest_words[sorted_results[res_idx][0]],res, "Cloud concept",closest_words[ordered_projection_results[pcres_idx][0]],pcres)
            if closest_words[ordered_projection_results[pcres_idx][0]] not in final_n_results_words:
                final_n_results_words.append(closest_words[ordered_projection_results[pcres_idx][0]])
                final_n_results.append(np.array(closest_vecs[ordered_projection_results[pcres_idx][0]]))
    if verbose: print("Finally",final_n_results_words)

    # return final_n_results_words,final_n_results,final_n_results_weights
    return final_n_results_words,final_n_results


def find_closest(pred_y,n_top=5,retro=True,skip=0,retrowords=None,retrovectors=None):
    if not (retrovectors is not None and retrowords is not None):
        if retro:
            retrowords, retrovectors =load_embedding("retrogan/numberbatch",limit=10000000)
        else:
            retrowords, retrovectors =load_embedding("retrogan/wiki-news-300d-1M-subword.vec",limit=10000000)
    # t1 = retrovectors[0].reshape(1,300)
    # t2 = pred_y.reshape(1,300)
    # res = cosine_similarity(t1,t2)
    results = [(idx,item) for idx,item in enumerate(list(map(lambda x: cosine_similarity(x.reshape(1,dimensionality), pred_y.reshape(1,dimensionality))[0][0],retrovectors)))]
    sorted_results = sorted(results,key=lambda x:x[1],reverse=True)
    final_n_results = []
    final_n_results_words = []
    for i in range(n_top):
        i += skip
        print(retrowords[sorted_results[i][0]],sorted_results[i][1])
        final_n_results_words.append(retrowords[sorted_results[i][0]])
        final_n_results.append(retrovectors[sorted_results[i][0]])

    del retrowords,retrovectors,results,sorted_results
    # gc.collect()
    return final_n_results_words,final_n_results

o = None
dimensionality = 768
def find_closest_2(pred_y,n_top=5,retro=True,skip=0,verbose=True
                   , dataset="fasttext", keep_o=False):
    original, retrofitted = datasets[dataset]
    if keep_o:
        global o
        if o is None:
            print("Loading")
            o = pd.read_hdf(directory + retrofitted, 'mat', encoding='utf-8')
            print("Filtering")
            o = o.loc[[x for x in o.index if '/c/en/' in x and "." not in x],:]
            print("Done")
    else:
        print("Loading")
        o = pd.read_hdf(directory + retrofitted, 'mat', encoding='utf-8')
        print("Filtering")
        o = o.loc[[x for x in o.index if '/c/en/' in x and "." not in x], :]
        print("Done")
    # print(o)
    results = [(idx,item) for idx,item in enumerate(list(map(lambda x: cosine_similarity(x.reshape(1,dimensionality), pred_y.reshape(1,dimensionality)),np.array(o.iloc[:,:]))))]
    sorted_results = sorted(results,key=lambda x:x[1],reverse=True)
    final_n_results = []
    final_n_results_words = []
    final_n_results_weights = []
    for i in range(n_top):
        if i<skip:
            continue
        # i += skip
        if(verbose):
            print(o.index[sorted_results[i][0]])
        final_n_results_words.append(o.index[sorted_results[i][0]]) # the index
        final_n_results.append(o.iloc[sorted_results[i][0],:]) # the vector
        final_n_results_weights.append(sorted_results[i][1])
    del results,sorted_results
    if not keep_o:
        del o
    gc.collect()
    # return final_n_results_words,final_n_results,final_n_results_weights
    return final_n_results_words,final_n_results

def find_closest_in_dataset(pred_y,dataset, n_top=5):
    print("Finding closest")
    if type(dataset) is str:
        o = pd.read_hdf(dataset, 'mat', encoding='utf-8')
    elif type(dataset) is pd.DataFrame:
        o = dataset
    else:
        raise Exception("Neither string nor dataframe provided as dataset")
    print(o)
    # if limit is None:
    # results = [(idx,item) for idx,item in enumerate(list(map(lambda x: cosine_similarity(x.reshape(1,dimensionality),
    #                                                                                  pred_y.reshape(1,dimensionality)),
    #                                                      np.array(o.iloc[:,:])
    #                                                      )))]
    # else:
    # limit = 400000
    # results = [(idx, item) for idx, item in enumerate(list(map(lambda x: cosine_similarity(x.reshape(1, dimensionality),
    #                                                                                        pred_y.reshape(1, dimensionality)),
    #                                                            np.array(o.iloc[0:limit, :])
    #                                                            )))]
    # sorted_results = sorted(results,key=lambda x:x[1],reverse=True)
    # final_n_results = []
    # final_n_results_words = []
    # # FULL FLEDGED COMPUTATION
    # for i in range(n_top):
    # #     # if(verbose):
    # #     #     print(o.index[sorted_results[i][0]])
    #     final_n_results_words.append(o.index[sorted_results[i][0]]) # the index
    #     final_n_results.append(o.iloc[sorted_results[i][0],:]) # the vector

    # del o,results,sorted_results
    # gc.collect()

    # testvec = find_in_dataset(["cat"], o)
    # print("Loading everything")
    testvec = pred_y
    print(testvec)
    try:
        print("Creating index")
        index = faiss.IndexFlatIP(300)  # build the index
        print(index.is_trained)
        print('Adding items to it')
        print(o.values.shape)
        index.add(np.ascontiguousarray(o.values).astype(np.float32))  # add vectors to the index
        print(index.ntotal)
        print("Converting the testvect to query")
        tst = np.array([testvec.astype(np.float32)])
        tst = tst.reshape((tst.shape[0],tst.shape[-1]))
        print("Ready to query")
    except Exception as e:
        print("We died")
        print(e)
        return [],[]
    print(tst.shape)
    print("Searching")
    D, I = index.search(tst, n_top)  # sanity check
    print(I)
    print(D)
    # print(o.iloc[I[0]])
    # print("Dumping results")
    final_n_results = o.iloc[I[0]].values
    final_n_results_words = o.index[I[0]].values

    return final_n_results_words,final_n_results




def find_in_fasttext(testwords,return_words=False, dataset="fasttext",prefix="/c/en/"):
    original,retrofitted = datasets[dataset]
    o = pd.read_hdf(directory + original, 'mat', encoding='utf-8')
    o = o.dropna()
    # r = pd.read_hdf(directory + retrofitted, 'mat', encoding='utf-8')
    result = []
    for concept in testwords:
        name_idx = prefix+concept
        try:
            a = np.array(o.loc[name_idx])
            result.append(a)
        except:
            print("Generating for",name_idx)
            a = np.array(generate_fastext_embedding(name_idx))
            result.append(a)
    t = np.array(result)
    print(t)
    return t

def find_in_retrofitted(testwords, return_words=False, dataset="fasttext",prefix="/c/en/"):
    # o = pd.read_hdf(directory + original, 'mat', encoding='utf-8')
    original,retrofitted = datasets[dataset]
    r = pd.read_hdf(directory + retrofitted, 'mat', encoding='utf-8')
    asarray1 = np.array(r.loc[[prefix + x for x in testwords], :])
    return asarray1

def check_index_in_dataset(testwords, dataset):
    results = [True if standardized_concept_uri("en",word) in dataset.index else False for word in testwords]
    return results

ft_model = None
def generate_fastext_embedding(word,keep_model_in_memory=True,ft_dir="fasttext_model/cc.en.300.bin",standardize=False):
    global ft_model
    standardized_form = word
    if standardize:
        standardized_form = standardized_concept_uri("en",word).replace("/c/en/","")
    if ft_model is None:
        ft_model = fasttext.load_model(ft_dir)
    wv = ft_model.get_word_vector(standardized_form)
    if not keep_model_in_memory:
        del ft_model
        ft_model=None
        gc.collect()
    return wv

def get_retrofitted_embedding(plain_embedding,retrogan_to_rfe_generator,dimensionality=300):
    re = np.array(retrogan_to_rfe_generator.predict(plain_embedding.reshape(1,dimensionality))).reshape((dimensionality,))
    return re

def hd5_to_txt(input_hd5,output_txt="out.txt",remove_qualifier=True):
    with open(output_txt,"w") as file:
        for item in input_hd5.index:
            word = item
            text = ""
            output_array = []
            if remove_qualifier:
                word = word.replace("/c/en/","")
            output_array.append(word)
            for num in input_hd5.loc[item]:
                output_array.append(float(num))
            for field in output_array:
                text+=str(field)+'\t'
            text+='\n'
            file.write(text)

def find_in_dataset(testwords,dataset,prefix=None):
    read = False
    if type(dataset) is str:
        r = pd.read_hdf(dataset, 'mat', encoding='utf-8')
        read = True
    elif type(dataset) is pd.DataFrame:
        r = dataset
    else:
        raise Exception("Neither string nor dataframe provided as dataset")
    def get_uri(name):
        if prefix is None:
            if "/c/en/" in name:
                return name
            else: return standardized_concept_uri('en', name)
        else:
            return prefix+name
    a = r.index
    asarray1 = r.loc[[get_uri(x) for x in testwords], :]
    # print(asarray1)
    asarray1 = np.array(asarray1)
    # print(asarray1)
    if read:
        del r
    # gc.collect()
    return asarray1

ft_model = None
def test_sem(model, dataset, dataset_location='SimLex-999.txt',fast_text_location="../fasttext_model/cc.en.300.bin",prefix=""):
    word_tuples = []
    my_word_tuples = []
    global ft_model
    ds_model = None
    # if ft_model is None:
    #     ft_model= fasttext.load_model(fast_text_location)
    if isinstance(dataset,pd.DataFrame):
        ds_model = dataset
    elif dataset is not None:
        ds_model = pd.read_hdf(dataset["original"],"mat")

        # ds_model=ds_model.swapaxes(0,1)
    retrogan = model
    with open(dataset_location) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            # print(f'Word1:\t{row[0]}\tWord2:\t{row[1]}\tSimscore:\t{row[2]}.')
            line_count += 1
            wtrow = []
            wtrow.append(row[0])
            wtrow.append(row[1])
            wtrow.append(row[3])
            word_tuples.append(wtrow)
            score = 0

            # conceptnet5.uri.concept_uri("en",row[0].lower())
            # mw1 = ft_model.get_word_vector(row[0].lower())
            try:
                mw1 = ds_model.loc[prefix+row[0].lower(),:]
                mw1 = np.array(retrogan.predict(np.array(mw1).reshape(1, 300))).reshape((300,))
                # mw2 = ft_model.get_word_vector(row[1].lower())
                mw2 = ds_model.loc[prefix + row[1].lower(),:]
                mw2 = np.array(retrogan.predict(np.array(mw2).reshape(1, 300))).reshape((300,))

                score = cosine_similarity([mw1], [mw2])
                del mw1, mw2
            except Exception as e:
                print(e)
                score = [0]
            my_word_tuples.append((row[0], row[1], score[0]))
        del csv_reader

        print(f'Processed {line_count} lines.')
    pr = pearsonr([float(x[2]) for x in word_tuples], [float(x[2]) for x in my_word_tuples])
    print(pr)
    sr = spearmanr([x[2] for x in word_tuples], [x[2] for x in my_word_tuples])
    print(sr)
    del my_word_tuples
    del word_tuples
    gc.collect()
    # word_tuples = sorted(word_tuples,key=lambda x:(x[0],x[2]))
    return sr