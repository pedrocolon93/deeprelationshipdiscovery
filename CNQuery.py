import time
from urllib.parse import urlencode

import requests

import tools


class CNQuery():
    def __init__(self,base_url="http://8kboxx/"):
        # self.base_url = "http://8kboxx/"
        self.base_url = base_url
        # node = / c / en / dog & other = / c / en / pizza

    def parse(self, result):
        amount = 0
        weight = 0
        if len(result['edges'])>0:
            amount+=1
            for edge in result["edges"]:
                print(edge)
                # amount+=1
                weight+=edge["weight"]
        else:
            pass
        return (amount,weight)

    def query_custom_parse(self,node1,node2,relation, parse_fun):
        results = parse_fun(self.query(node1, node2, relation))
        return results
    def add_identifier(self,node):
        if "/c/en" in node:
            return node
        else:
            return tools.standardized_concept_uri("en",node)
    def query(self, node1, node2, relation=None):
        getvars_dict = {}
        getvars_dict['node'] = self.add_identifier(node1)
        if node2 is not None:
            getvars_dict['other'] = self.add_identifier(node2)
        if relation is not None:
            getvars_dict['rel']=relation
        # getvars_dict['limit']=100
        getVars = getvars_dict
        url = self.base_url+"query?"
        res = url+urlencode(getVars)
        i = 0
        retry = 4
        while True or i>retry:
            try:
                urlres = requests.get(res, timeout=5)
                urlres = urlres.json()

                break
            except:
                i+=1
                time.sleep(2)
                pass
        # print(urlres)
        # print(urlres)
        return urlres

    def query_and_parse(self, node1, node2, relation='/r/Desires'):

        results = self.parse(self.query(node1,node2,relation))
        return results