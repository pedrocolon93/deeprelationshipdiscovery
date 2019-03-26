import json
import os
import pickle

import pg8000
from conceptnet5.db.connection import get_db_connection
from conceptnet5.db.query import AssertionFinder
import numpy as np
import pandas as pd
import hashlib
from conceptnet5.nodes import *
from conceptnet5.api import *
from conceptnet5.relations import *

class edge():
    def __init__(self,start,end,reltype,weight):
        start_label = start
        end_label = end

        if '/' in start_label:
            prefix = ""
            for item in start_label.split('/')[0:3]:
                prefix+=item+"/"
            start_label = prefix+start_label.split('/')[3]
        if '/' in end_label:
            prefix = ""
            for item in end_label.split('/')[0:3]:
                prefix += item + "/"
            end_label =prefix +end_label.split('/')[3]
        self.start = start_label
        self.end = end_label
        self.reltype = reltype
        self.weight = weight

    @classmethod
    def from_edge_object(cls,entry):
        start_label = entry["start"]["label"]
        end_label = entry["end"]["label"]

        if '/' in start_label:
            prefix = ""
            if "/c/en" in start_label:
                prefix = "/c/en/"
            start_label = prefix+start_label.split('/')[3]
        if '/' in end_label:
            prefix = ""
            if "/c/en" in end_label:
                prefix = "/c/en/"
            end_label =prefix +end_label.split('/')[3]
        e = cls(start_label, end_label, entry["rel"]["label"], entry["weight"])
        return e

    def __str__(self):
        return "Start:"+str(self.start)+" End:"+str(self.end)+" Relationship:"+str(self.reltype)+" Weight:"+str(self.weight)

    def __eq__(self, other):
        if not isinstance(other,edge):
            return False
        return self.start == other.start and self.end == other.end and self.reltype == other.reltype and self.weight == other.weight


    def __hash__(self):
        hashobj = hashlib.md5(str(self).encode('utf-8'))
        return int(hashobj.hexdigest(), 16)

def get_random_subset(iterations=10,limit=10000):
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
    res = af.sample_dataset(uri=uri,limit=10000)
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

def load_simple_concepts_and_expand(path_to_conceptlist="simpleconcepts",save = True,filename = "simple_concept_relations",limit=100):
    edgelist = []
    af = AssertionFinder()
    offset = 0
    print("Opening the simple ocncept list")
    with open(path_to_conceptlist) as f:
        for line in f:
            if "#" in line:
                continue
            else:
                standard_uri = standardize_concept_uri("en",line.strip())
                res = af.lookup(standard_uri,limit=limit,offset=offset)
                for item in res:
                    if item["@type"]=="Edge":
                        edgelist.append(edge.from_edge_object(item))
    secondedgelist = []
    print("Following up on ends")
    print(len(edgelist))
    for idx,edgeobj in enumerate(edgelist):
        print(idx)
        standard_uri = standardize_concept_uri("en", edgeobj.end.strip())
        res = af.lookup(standard_uri, limit=limit,offset=offset)
        for item in res:
            if item["@type"] == "Edge":
                secondedgelist.append(edge.from_edge_object(item))
    edgelist = edgelist+secondedgelist
    print("dumping")

    if save:
        pickle.dump(edgelist, open(filename,'wb'))
    return edgelist

def complex_concept_load(N=6):
    connection = get_db_connection(None)
    cursor = connection.cursor()
    left_edges_query = '''
    select edges.start_id
    from edges
    group by edges.start_id
    having count(*)>'''+str(N)+''';'''
    right_edges_query = '''
    select edges.end_id
    from edges
    group by edges.end_id
    having count(*)>'''+str(N)+''';'''
    left_ids = set()
    right_ids = set()
    cursor.execute(left_edges_query)
    results = cursor.fetchall()
    for result in results:
        left_ids.add(result)
    cursor.execute(right_edges_query)
    results = cursor.fetchall()
    for result in results:
        right_ids.add(result)
    resset = left_ids.union(right_ids)
    print(len(resset))
    concept_uris = []
    print("Finding concept names")
    for concept in resset:
        conceptname_query = '''
        select uri 
        from nodes
        where nodes.id='''+str(concept[0])+''';'''
        cursor.execute(conceptname_query)
        res = cursor.fetchall()
        concept_uris.append(res[0][0])
    print("Finding edges")
    the_ultimate_edge_list = []
    for idx, concept_uri in enumerate(resset):
        if idx%10000==0:
            print(idx)
        relation_query = '''
        select relations.uri,s.uri,v.uri,t.weight  
        from 
        ((select distinct edges.id
        from edges
        where edges.weight>=1 and (edges.start_id='''+str(concept_uri[0])+''' OR 
        edges.end_id='''+str(concept_uri[0])+''')) uids
        inner join edges
        on edges.id=uids.id) t
        inner join relations on relations.id=t.relation_id
        inner join nodes as s on s.id = t.start_id
        inner join nodes as v on v.id = t.end_id
        ;'''
        cursor.execute(relation_query)
        res = cursor.fetchall()
        # print(len(res))
        the_ultimate_edge_list+=res
    print("Dumping things.")
    pickle.dump(resset,open("resset","wb"))
    pickle.dump(concept_uris,open("concept_uris","wb"))
    pickle.dump(the_ultimate_edge_list,open("ultimate_edge_list",'wb'))
    print("Done")
    return resset
def load_local_edgelist(path="ultimate_edge_list",limit = 10000000):
    print("Loading:",limit)
    s  = pickle.load(open(path,'rb'))
    converted = []
    print("The max is ")
    print(len(s))
    for x in s:
        if len(converted)>limit:
            break
        e = edge(x[1],x[2],x[0],x[3])
        if "/c/en/" in e.start or "/c/en" in e.end:
            converted.append(e)
    del s
    return converted

if __name__ == '__main__':
    pass
    complex_concept_load(6)
    # el = pickle.load(open("ultimate_edge_list",'rb'))
    # el = set([x[0] for x in el])
    # connection = get_db_connection(None)
    # cursor = connection.cursor()
    # formal_edgelist = []
    # for idx,edge in enumerate(el):
    #     # print(str(idx)+"/"+str(len(el)))
    #     if idx%10000==0:
    #         print(idx)
    #     query = '''select * from edges where edges.id='''+str(edge)
    #     cursor.execute(query)
    #     res = cursor.fetchall()
    #     formal_edgelist+=res
    # pickle.dump(formal_edgelist,open("formal_edge_list",'wb'))
    # print(el)