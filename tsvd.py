import pickle

from hpfrec import HPF
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix

from cnscraper import get_random_subset, split_features, convert_to_counts_df

rss = get_random_subset(iterations=100,limit=10000)
pickle.dump(rss,open('random_sub_set','wb'))
conceptlist,featurelist,weightlist = split_features(rss)
counts_df = convert_to_counts_df(conceptlist,featurelist,weightlist)

X = sparse_random_matrix(100, 100, density=0.01, random_state=42)
svd = TruncatedSVD(n_components=6, n_iter=7, random_state=42)
pmf = HPF(k)

svd.fit(X)
print(svd.singular_values_)