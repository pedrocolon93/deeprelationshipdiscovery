from os import wait
from time import sleep
from urllib.parse import urlencode

from tools import *
# import conceptnet5
import requests

class CNQuery():
    def __init__(self):
        self.base_url = "http://18.85.39.186/"
        # self.base_url = "http://conceptnet.io/"
        # node = / c / en / dog & other = / c / en / pizza

    def parse(self, result):
        amount = 0
        weight = 0
        if len(result['edges'])>0:
            for edge in result["edges"]:
                amount+=1
                weight+=edge["weight"]
        else:
            pass
        return (amount,weight)

    def add_identifier(self,node):
        if "/c/en" in node:
            return node
        else:
            return "/c/en/"+node
    def query(self,node1,node2,relation='/r/Desires'):
        getVars = {'node': self.add_identifier(node1), 'other': self.add_identifier(node2),'rel':relation}
        url = self.base_url+"query?"
        res = url+urlencode(getVars)
        urlres = requests.get(res)
        # print(urlres)
        urlres = urlres.json()
        # print(urlres)
        results = self.parse(urlres)
        return results




if __name__ == '__main__':

    # find n synonyms using basic word embedding
    print("Potato")
    nearest_concepts_amount = 300
    # concept_1 = "dog"
    concept_1 = "man"
    relationship_type = "/r/Desires"
    # concept_2 = "pizza"
    concept_2 ="woman"
    concept_vectors = find_in_numberbatch([concept_1,concept_2],return_words=True)
    # Nearest to concepts
    # concept1_neighbors_words,concept1_neighbors_vectors = find_closest(concept_vectors[0],n_top=nearest_concepts_amount, skip=0)
    # concept2_neighbors_words,concept2_neighbors_vectors = find_closest(concept_vectors[1],n_top=nearest_concepts_amount, skip=0)
    # Nearest across concepts
    concept1_neighbors_words,concept1_neighbors_vectors = find_cross_closest(concept_vectors[0],concept_vectors[1],n_top=nearest_concepts_amount,closest=0)
    concept2_neighbors_words,concept2_neighbors_vectors = find_cross_closest(concept_vectors[0],concept_vectors[1],n_top=nearest_concepts_amount,closest=1)
    cutoff_amount = 1
    # Nearest together concepts
    ##TODO FIND THE NEAREST CONCEPTS TO THE ADDITION/SUBTRACTION OF THE 2 CONCEPTS
    connection_amount = 0
    connection_weight = 0
    current_iter = 0
    for c1_idx, concept1_neighbor in enumerate(concept1_neighbors_words):
        for c2_idx, concept2_neighbor in enumerate(concept2_neighbors_words):
            # check neighbor 1 with neighbor 2
            try:
                amount, weight = CNQuery().query(concept1_neighbor,concept2_neighbor,relationship_type)
                # sleep(0.5)
                connection_amount+=amount
                connection_weight+=weight
            except Exception as e:
                #wait a bit
                print(e)
                print("skipping")
                # sleep(1)
            current_iter+=1
            if connection_amount>=cutoff_amount:
                break
        print(current_iter)

    print("The strength of that assumption is:")
    print(connection_amount)
    print(connection_weight)
    # find nodes in graph
    # see if connection exists
    # average it
