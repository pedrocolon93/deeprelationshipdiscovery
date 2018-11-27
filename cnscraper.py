import json
import os
import pickle

from conceptnet5.db.query import AssertionFinder
import numpy as np
import pandas as pd
import hashlib
from conceptnet5.nodes import *
from conceptnet5.api import *

class edge():
    def __init__(self,start,end,reltype,weight):
        self.start = start
        self.end = end
        self.reltype = reltype
        self.weight = weight

    @classmethod
    def from_edge_object(cls,entry):
        return cls(entry["start"]["label"],entry["end"]["label"],entry["rel"]["label"],entry["weight"])

    def __str__(self):
        return "Start:"+str(self.start)+" End:"+str(self.end)+" Relationship:"+str(self.reltype)+" Weight:"+str(self.weight)

    def __eq__(self, other):
        if not isinstance(other,edge):
            return False
        return self.start == other.start and self.end == other.end and self.reltype == other.reltype and self.weight == other.weight


    def __hash__(self):
        hashobj = hashlib.md5(str(self).encode('utf-8'))
        return int(hashobj.hexdigest(), 16)

def get_random_subset(iterations=10,limit=1000):
    print("Collecting random subset")
    edgeset = set()
    for i in range(iterations):
        af = AssertionFinder()
        re = af.random_edges(limit=limit)
        def convert_to_hashable(entry):
            return edge(entry["start"]["label"],entry["end"]["label"],entry["rel"]["label"],entry["weight"])
        re = [convert_to_hashable(x) for x in re if x["start"]["language"]=='en' and x["rel"]["label"]!="ExternalURL"]
        print(len(re))
        edgeset = edgeset.union(set(re))
    return list(edgeset)

def parse_random_subset(path="conceptnetdumps/dump11"):
    if not os.path.exists(path):
        raise FileNotFoundError("No such path")
    f = open(path)
    loaded = json.load(f)
    print("Loaded")
    res = None
    return res

def split_features(edgelist,type = "left" ):
    conceptlist = []
    featurelist = []
    weightlist = []
    if type not in ("left","right"):
        raise Exception("No correct type only:"+str(("left","right")))
    for edge in edgelist:
        if type == "right":
            conceptlist.append(edge.start)
            featurelist.append(edge.reltype+":"+edge.end)
        elif type == "left":
            conceptlist.append(edge.end)
            featurelist.append(edge.reltype+":"+edge.start)
        weightlist.append(edge.weight)
    return conceptlist,featurelist,weightlist

def convert_to_counts_df(conceptlist,featurelist,weightlist):
    ## Generating sample counts data

    counts_df = pd.DataFrame({
        'UserId': np.array(conceptlist),
        'ItemId': np.array(featurelist),
        'Count': np.array(weightlist)
    })
    # Remove Nans
    counts_df = counts_df.loc[counts_df.Count > 0].reset_index(drop=True)
    return counts_df

def test_random_sample():
    print("Here")
    af = AssertionFinder()
    uri="/d/conceptnet/4/en"
    print(os.path.abspath("./"))
    res = af.sample_dataset(uri=uri,limit=100000)
    i = 1
    while True:
        filepath = "./conceptnetdumps/dump"+str(i)
        if os.path.exists(filepath):
            i+=1
            continue
        else:
            f = open(filepath, "w+")
            json.dump(res,f)
            f.close()
            break
    print("Dumped in "+"conceptnetdumps/dump"+str(i))

def load_simple_concepts_and_expand(path_to_conceptlist="simpleconcepts",save = True):
    edgelist = []
    with open(path_to_conceptlist) as f:
        for line in f:
            if "#" in line:
                continue
            else:
                standard_uri = standardize_concept_uri("en",line.strip())
                af = AssertionFinder()
                res = af.lookup(standard_uri)
                for item in res:
                    if item["@type"]=="Edge":
                        edgelist.append(edge.from_edge_object(item))
    if save:
        pickle.dump(edgelist, open("simple_concept_relations",'wb'))
    return edgelist
if __name__ == '__main__':
    load_simple_concepts_and_expand()