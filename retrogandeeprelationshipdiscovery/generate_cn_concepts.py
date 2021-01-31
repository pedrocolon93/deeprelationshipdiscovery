import random
import csv
import datetime
import bisect
from itertools import combinations

from tqdm import tqdm

from CNQuery import CNQuery

random.seed(0)
DATA_PATH = "/Users/pedro/Downloads/"
NB_PATH = DATA_PATH + "numberbatch-en-19.08.txt"
CONCEPTS_PATH = "/home/pedro/Downloads/concept_"
num_og_concepts = 4 # number of concepts to sample from NumberBatch
search_depth = 4 #
max_concepts = 50
num_iters = 5 #

def bi_directional_search(q, start, goal):
    def find_intersecting(s_visited, t_visited):
        intersecting_nodes = []
        for node in s_visited:
            if node in t_visited:
                intersecting_nodes.append(node)
        return intersecting_nodes

    def bfs(q, queue, visited, parent, in_order, edge_info):
        path_weight, node = queue.pop()
        while ((datetime.datetime.now() - q.last_query_time).total_seconds() < 1.0):
            continue
        q_result = q.query(node, None)
        q.last_query_time = datetime.datetime.now()
        for neighbor in q_result['edges']:
            start_word = neighbor['start']['@id']
            end_word = neighbor['end']['@id']
            rel = neighbor['rel']['@id']
            text = neighbor["surfaceText"]
            weight = neighbor['weight']
            # print(start_word, end_word)
            if start_word[:6] != "/c/en/" or end_word[:6] != "/c/en/":
                continue
            # print("PASS")

            if start_word == end_word:
                continue
            start_i = start_word[6:].find("/")
            if start_i != -1:
                start_word = start_word[:6+start_i]
            end_i = end_word[6:].find("/")
            if end_i != -1:
                end_word = end_word[:6+end_i]

            new_word = end_word if start_word == node else start_word
            if new_word not in visited:
                parent[new_word] = node
                in_order[new_word] = start_word == node
                edge_info[(start_word, end_word)] = (rel, weight, text)
                visited.add(new_word)
                bisect.insort_left(queue,(weight+path_weight,new_word))
        return 

    def get_paths(s_parent, t_parent, s_in_order, t_in_order, intersecting_nodes):
        paths = []
        for intersecting_node in intersecting_nodes:
            curr_s_path = []
            curr_node = intersecting_node
            while s_parent[curr_node] != None:
                if s_in_order[curr_node]:
                    curr_s_path.append((s_parent[curr_node],curr_node))
                else:
                    curr_s_path.append((curr_node,s_parent[curr_node]))
                curr_node = s_parent[curr_node]
            
            curr_t_path = []
            curr_node = intersecting_node
            while t_parent[curr_node] != None:
                if t_in_order[curr_node]:
                    curr_t_path.append((t_parent[curr_node],curr_node))
                else:
                    curr_t_path.append((curr_node,t_parent[curr_node]))
                curr_node = t_parent[curr_node]
            paths.append(curr_s_path[::-1] + curr_t_path)
        return paths

    q = CNQuery()
    s_visited, t_visited = (set([start]), set([goal]))
    s_parent, t_parent = ({start: None}, {goal: None})
    s_in_order, t_in_order = ({start: True}, {goal: True})
    s_edge_info, t_edge_info = ({},{})
    s_queue, t_queue = ([(0,start)],[(0,goal)])
    i = 0
    while s_queue and t_queue and i < 5000: # longest path length we search is 7
        bfs(q, s_queue, s_visited, s_parent, s_in_order, s_edge_info)
        bfs(q, t_queue, t_visited, t_parent, t_in_order, t_edge_info)
        intersecting_nodes = find_intersecting(s_visited, t_visited)
        if len(intersecting_nodes) > 0:
            s_edge_info.update(t_edge_info)
            return get_paths(s_parent, t_parent, s_in_order, t_in_order, intersecting_nodes), s_edge_info
        i += 1
    return None, None

def parse_edges(edges, seen_concept_pairs):
    return [(edge['start']['@id'], edge['end']['@id'], edge['rel']['@id'], edge['weight'], edge['surfaceText']) for edge in edges \
            if edge['start']['@id'][:6] == "/c/en/" and edge['end']['@id'][:6]== "/c/en/" \
                and (edge['start']['@id'], edge['end']['@id']) not in seen_concept_pairs and (edge['end']['@id'],edge['start']['@id']) not in seen_concept_pairs]

if __name__ == '__main__':
    # 1. Get all concepts from NumberBatch
    with open(NB_PATH, 'r') as nb_file:
        nb_concepts = [nb.split(' ')[0] for nb in nb_file.readlines()]
    # 2. Sample x concepts from list
   
    random.shuffle(nb_concepts)
    q = CNQuery()
    for curr_iter in tqdm(range(num_iters)):
        og_concepts = nb_concepts[curr_iter*num_og_concepts:curr_iter*num_og_concepts+num_og_concepts]
        og_concepts = ["/c/en/"+concept if "/c/en/" != concept[:6] else concept for concept in og_concepts]
        # print("OG_CONCEPTS: %s" % og_concepts)
        concept_pairs = combinations(og_concepts, 2)
        concept_relations = []
        seen_concept_pairs = set()
        new_concepts = set()
        i = 0
        
        while concept_pairs and i <= search_depth and len(seen_concept_pairs) < max_concepts:
            for c1, c2 in concept_pairs:
                if (c1,c2) in seen_concept_pairs or (c2,c1) in seen_concept_pairs:
                    continue
                rel_edges = q.query(c1, c2)['edges'] # try seeing if direct edge between the two
                q.last_query_time = datetime.datetime.now()
                if len(rel_edges) == 0:
                    paths, edge_info = bi_directional_search(q, c1, c2) # find path between two concepts
                    if paths is None: # no path found
                        # print("NO PATH FOUND BETWEEN (%s,%s)" % (c1,c2))
                        continue
                    for path in paths:
                        # print("PATH %s %s: %s" % (c1,c2,path))
                        for i in range(0,len(path)-1):
                            if i != 0:
                                new_concepts.add(path[i][0])
                                new_concepts.add(path[i][1])
                            pair = path[i]
                            if pair not in seen_concept_pairs and (path[i][1],path[i][0]) not in seen_concept_pairs:
                                rel, weight, text = edge_info[pair]
                                concept_relations.append((pair[0], pair[1], rel,weight,text))
                                seen_concept_pairs.add(pair)
                else:
                    # print("FOUND EDGES BETWEEN (%s,%s): %s" % (c1,c2,[(edge['start']['@id'], edge['end']['@id']) for edge in rel_edges]))
                    concept_relations.extend(parse_edges(rel_edges, seen_concept_pairs))
                seen_concept_pairs.add((c1,c2))
            concept_pairs = combinations(list(new_concepts),2) # reset concept_pairs to be new concept pairs found
            i += 1
        with open(CONCEPTS_PATH + str(curr_iter) + ".csv", "w") as outf:
            writer = csv.writer(outf)
            for relation in concept_relations:
                writer.writerow(relation)

    
        
        





        

    # 3. Query the combos of 2)
    # 4. If depth > 0, add new concepts to 2)
    # 5. Generate list of assertions (origin concept, target concept, rel, weight, text)
    # 6. Repeat 2-5 N times

    # THE BELOW WILL PROB BE IN ANOTHER FILE BUT KEEP IN MIND WILL NEED TO:
    # 7. Split to train, dev, test
    # 8. Run chimera
    # 9. Check
    pass
