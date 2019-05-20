from threading import Thread, Lock

import deep_relationship_discovery
import tools
from CNQuery import CNQuery
from deep_relationship_discovery import load_model_ours
from tools import *

# import conceptnet5


if __name__ == '__main__':
    # test connection
    print(CNQuery().query_and_parse('/c/en/man', '/c/en/woman'))
    # find n synonyms using basic word embedding
    nearest_concepts_amount = 20
    cross_concepts_amount = 200
    dimensionality = 300
    tools.dimensionality = dimensionality

    # concept_1 = "cat"
    # concept_1 = "man"
    concept_1 = "smartphone"
    # concept_1 = "uber"
    # concept_1 = "dead"
    # relationship_type = "/r/Desires"
    # relationship_type = "/r/FormOf"
    # relationship_type = "/r/NotCapableOf"
    relationship_type = "/r/UsedFor"
    # relationship_type = "/r/Desires"
    # concept_2 = "pizza"
    # concept_2 ="smartphone"
    # concept_2 = "food"
    concept_2 = "uber"

    print(concept_1, relationship_type, concept_2)
    target_file_loc = 'trained_models/retroembeddings/2019-04-0813:03:02.430691/retroembeddings.h5clean'
    target_voc = pd.read_hdf(target_file_loc, 'mat')
    concept_vectors = find_in_dataset([concept_1, concept_2], dataset=target_voc)


    # Nearest to concepts
    # concept1_neighbors_words,concept1_neighbors_vectors = find_closest(concept_vectors[0],n_top=nearest_concepts_amount, skip=0)
    # concept2_neighbors_words,concept2_neighbors_vectors = find_closest(concept_vectors[1],n_top=nearest_concepts_amount, skip=0)
    # Nearest across concepts
    class ThreadWithReturnValue(Thread):
        def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
            Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)

            self._return = None

        def run(self):
            if self._target is not None:
                self._return = self._target(*self._args, **self._kwargs)

        def join(self):
            Thread.join(self)
            return self._return


    print("Loading concept 1 cross closest neighbors")
    # c1w, c1v,c1ww = find_closest_2(concept_vectors[0],n_top=nearest_concepts_amount)
    # concept1_neighbors_words,concept1_neighbors_vectors = find_cross_closest_2(concept_vectors[0],concept_vectors[1],n_top=cross_concepts_amount,closest=0,verbose=True)
    twrv1 = ThreadWithReturnValue(target=find_cross_closest_dataset, args=(concept_vectors[0], concept_vectors[1]),
                                  kwargs={"n_top": cross_concepts_amount, "closest": 0, "verbose": True,
                                          "dataset": target_voc})

    print("Done.\nLoading concept 2 cross closest neighbors.")
    # concept2_neighbors_words,concept2_neighbors_vectors = find_cross_closest_2(concept_vectors[0],concept_vectors[1],n_top=cross_concepts_amount,closest=1,verbose=True)
    twrv2 = ThreadWithReturnValue(target=find_cross_closest_dataset, args=(concept_vectors[0], concept_vectors[1]),
                                  kwargs={"n_top": cross_concepts_amount, "closest": 1, "verbose": True,
                                          "dataset": target_voc})
    twrv1.start()
    twrv2.start()
    concept1_neighbors_words, concept1_neighbors_vectors = twrv1.join()
    concept2_neighbors_words, concept2_neighbors_vectors = twrv2.join()
    if "/c/en/camera" in concept1_neighbors_words:
        print("in c1 neighs")
    if "/c/en/camera" in concept2_neighbors_words:
        print("in c2 neighs")
    print("Done")
    filtered = [x for x in concept2_neighbors_words if x not in concept1_neighbors_words]
    c2w = c2v = []
    for word in filtered:
        c2w.append(word)
    cutoff_amount = 1
    # Nearest together concepts
    ##TODO FIND THE NEAREST CONCEPTS TO THE ADDITION/SUBTRACTION OF THE 2 CONCEPTS
    current_iter = 0
    thread_list = []
    threadlimit = 32
    count = 0
    print("Parameters cutoff:%i nearest concepts amount:%i" % (cutoff_amount, nearest_concepts_amount))

    print(str(len(concept1_neighbors_words) * len(c2w)))
    lock = Lock()
    i = 0
    words_to_explore = concept1_neighbors_words + concept2_neighbors_words
    remove_dups = True
    if remove_dups:
        words_to_explore = list(set(words_to_explore))
    print("Exploring", len(words_to_explore), len(words_to_explore) ** 2)
    cutoff_threshold_reached = False
    rel_values = []
    rel=relationship_type.replace("/r/","")

    drd_models_path = "trained_models/deepreldis/2019-04-2314:43:00.000000"
    drd_model = load_model_ours(save_folder=drd_models_path, model_name=rel)
    normalizers = deep_relationship_discovery.normalize_outputs(None, save_folder=drd_models_path,use_cache=True)

    for c1_idx, concept1_neighbor in enumerate(words_to_explore):
        for c2_idx, concept2_neighbor in enumerate(words_to_explore):
            if concept2_neighbor == concept1_neighbor:
                continue
            start_vec, end_vec = find_in_dataset([concept1_neighbor, concept2_neighbor], target_voc)
            inferred_res = drd_model[rel].predict(x={"retro_word_1": start_vec.reshape(1, tools.dimensionality),
                                                     "retro_word_2": end_vec.reshape(1, tools.dimensionality)})
            norm_res = normalizers[rel].transform(inferred_res)

            rel_values.append(norm_res)

    print("The strength of that assumption is:")
    print(np.mean(rel_values))
