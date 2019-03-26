from hpfrec import HPF
from scipy.sparse import coo_matrix

from failed_tests.cnscraper import *
# rss = None
# if os.path.isfile("random_sub_set"):
#     rss = pickle.load(open("random_sub_set",'rb'))
# else:
# rss = load_simple_concepts_and_expand()

def convert_to_coo_sparse_matrix(conceptlist, featurelist, weightlist):

    conceptmap = {}
    idx = 0
    for concept in conceptlist:
        if concept not in conceptmap.keys():
            conceptmap[concept] = idx
            idx+=1

    featuremap = {}
    idx = 0
    for feature in featurelist:
        if feature not in featuremap.keys():
            featuremap[feature] = idx
            idx += 1

    conceptlist_reindexed = [conceptmap[concept] for concept in conceptlist]
    featurelist_reindexed = [featuremap[feature] for feature in featurelist]
    res = coo_matrix((weightlist, (conceptlist_reindexed, featurelist_reindexed)))
    return conceptmap, featuremap, res


def normalize_weights(conceptlist, featurelist, weightlist):
    normed_weightlist = [0 for x in weightlist]
    used = set()
    idxmap = {}
    print("Building lookup table")
    for num,concept in enumerate(conceptlist):
        if concept not in idxmap.keys():
            idxmap[concept] = []
        idxmap[concept].append(num)
    print('Lookup table done')
    reweighted = 0
    for concept in conceptlist:
        if concept in used:
            continue
        else:
            repeated_idxs = idxmap[concept]
            # vals = [np.array([weightlist[x] for x in repeated_idxs])]
            # normed_vals = normalize(vals)
            temp = []
            for x in repeated_idxs:
                weight = weightlist[x]
                if weight<1:
                    reweighted+=1
                    weight*=100
                weight = round(weight)
                temp.append(weight)
            vals = [np.array([(x-1)/(100-1) for x in temp])]
            # vals = [np.array([((weightlist[x]*100)-1)/(100-1) for x in repeated_idxs])]
            normed_vals = vals
            for idx, repeated_idx in enumerate(repeated_idxs):
                normed_weightlist[repeated_idx] = normed_vals[0][idx]
            used.add(concept)
    print("Reweighted:")
    print(reweighted)
    return normed_weightlist
if __name__ == '__main__':
    # rss = pickle.load(open('trainsubset', 'rb'))
    print("Loading edgelist")
    rss = load_local_edgelist(limit=10000000)
    print("Splitting features")
    conceptlist, featurelist, weightlist = split_features(rss)

    testconcepts = ["car","person","flower","cat","dog","potato"]
    res = [i for i in range(len(conceptlist)) if conceptlist[i] in testconcepts]
    #filteredtest
    testconceptlist = [conceptlist[idx] for idx in res]
    testfeaturelist = [featurelist[idx] for idx in res]
    testweightlist = [weightlist[idx] for idx in res]
    # #filteredtrain
    # trainconceptlist = [conceptlist[idx] for idx in range(len(conceptlist)) if idx not in res]
    # trainfeaturelist = [featurelist[idx] for idx in range(len(conceptlist)) if idx not in res]
    # trainweightlist = [weightlist[idx] for idx in range(len(conceptlist)) if idx not in res]

    print("Normalizing weights of rows")
    normalized_weights = normalize_weights(conceptlist,featurelist,weightlist)
    del rss
    print("Converting to counts df")
    counts_df = convert_to_counts_df(conceptlist, featurelist, normalized_weights)

    # conceptmap, featuremap, csm = convert_to_column_sparse_matrix(conceptlist, featurelist, weightlist)
    # normed_matrix = normalize(csm, axis=1)

    del conceptlist,featurelist,weightlist
    # csm = convert_to_column_sparse_matrix(conceptlist,featurelist,weightlist)
    # pickle.dump(rss,open('trainsubset','wb'))
    #

    ## Initializing the model object
    kvals = [5,15,25,50]
    for k in kvals:
        print('Creating recommender')
        recommender = HPF(k=k,random_seed=42,alloc_full_phi=True)#maxiter=50,stop_thr=10e-6,stop_crit="maxiter",
        print("Fitting recommender")
        recommender.fit(counts_df)

        print("Tests:\n-----------------------------\n")
        n = 10
        es = True
        for concept in testconcepts:
            print("Evaluating:",concept)
            print("excluding seen")
            print(recommender.topN("/c/en/"+concept,n,exclude_seen=es))
            # print("not excluding seen")
            # print(recommender.topN("/c/en/"+concept,n,exclude_seen=(not es)))
            print("-----")

        print("Predictions:\n--------------------------\n")
        print(recommender.predict(user="/c/en/good", item='/r/RelatedTo:/c/en/bad'))
        print(recommender.predict(user="/c/en/dog", item='/r/RelatedTo:/c/en/cat'))
        print("Done")
