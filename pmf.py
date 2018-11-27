import pickle

import pandas as pd, numpy as np
from hpfrec import HPF

from cnscraper import *
rss = None
# if os.path.isfile("random_sub_set"):
#     rss = pickle.load(open("random_sub_set",'rb'))
# else:
rss = load_simple_concepts_and_expand()
pickle.dump(rss,open('random_sub_set','wb'))
conceptlist,featurelist,weightlist = split_features(rss)
counts_df = convert_to_counts_df(conceptlist,featurelist,weightlist)
#
## Initializing the model object
recommender = HPF(k=100)#,stop_crit='maxiter',maxiter=100)

## Fitting the model to the data
recommender.fit(counts_df)
#
# ## Making predictions
# # recommender.topN(user=10, n=10, exclude_seen=True) ## not available when using 'partial_fit'
# # recommender.topN(user=10, n=10, exclude_seen=False, items_pool=np.array([1,2,3,4]))
# concept = str(input("Concept"))
# feature = str(input("Relationship:Concept"))
# print(recommender.predict(user=concept, item=feature))
