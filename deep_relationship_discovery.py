import json
import multiprocessing
import os
from multiprocessing import Queue
from multiprocessing.pool import Pool
from random import shuffle

from keras import Input, Model
from keras.layers import Dense, Concatenate, BatchNormalization
from keras.optimizers import Adam
from keras.utils import plot_model
from tqdm import tqdm

from gan_tester import attention
import pandas as pd
import csv

relations = ["/r/RelatedTo", "/r/FormOf", "/r/IsA", "/r/PartOf", "/r/HasA", "/r/UsedFor", "/r/CapableOf",
             "/r/AtLocation",
             "/r/Causes", "/r/HasSubevent", "/r/HasFirstSubevent", "/r/HasLastSubevent", "/r/HasPrerequisite",
             "/r/HasProperty", "/r/MotivatedByGoal", "/r/ObstructedBy", "/r/Desires", "/r/CreatedBy", "/r/Synonym",
             "/r/Antonym", "/r/DistinctFrom", "/r/DerivedFrom", "/r/SymbolOf", "/r/DefinedAs", "/r/Entails",
             "/r/MannerOf",
             "/r/LocatedNear", "/r/HasContext", "/r/SimilarTo", "/r/EtymologicallyRelatedTo",
             "/r/EtymologicallyDerivedFrom", "/r/CausesDesire", "/r/MadeOf", "/r/ReceivesAction", "/r/InstanceOf",
             "/r/NotDesires", "/r/NotUsedFor", "/r/NotCapableOf", "/r/NotHasProperty"]


def create_model():
    # Input needs to be 2 word vectors
    wv1 = Input(shape=(300,), name="retro_word_1")
    wv2 = Input(shape=(300,), name="retro_word_2")
    expansion_size = 256
    # Expand and contract the 2 word vectors
    wv1_expansion_1 = Dense(expansion_size)(wv1)
    wv1_expansion_1 = BatchNormalization()(wv1_expansion_1)
    wv1_expansion_2 = Dense(int(expansion_size / 2))(wv1_expansion_1)
    wv1_expansion_2 = BatchNormalization()(wv1_expansion_2)
    wv1_expansion_3 = Dense(int(expansion_size / 4))(wv1_expansion_2)
    wv1_expansion_3 = BatchNormalization()(wv1_expansion_3)


    wv2_expansion_1 = Dense(expansion_size)(wv2)
    wv2_expansion_1 = BatchNormalization()(wv2_expansion_1)
    wv2_expansion_2 = Dense(int(expansion_size / 2))(wv2_expansion_1)
    wv2_expansion_2 = BatchNormalization()(wv2_expansion_2)
    wv2_expansion_3 = Dense(int(expansion_size / 4))(wv2_expansion_2)
    wv2_expansion_3 = BatchNormalization()(wv2_expansion_3)

    # Concatenate both expansions
    merge1 = Concatenate()([wv1_expansion_3, wv2_expansion_3])
    merge_expand = Dense(expansion_size)(merge1)
    merge_expand = BatchNormalization()(merge_expand)
    # Add atention layer
    merge_attention = attention(merge_expand)
    attention_expand = Dense(expansion_size)(merge_attention)
    attention_expand = BatchNormalization()(attention_expand)
    semi_final_layer = Dense(expansion_size)(attention_expand)
    # Output layer
    amount_of_relations = len(relations)
    # One big layer
    # final = Dense(amount_of_relations)(semi_final_layer)
    # Many tasks
    tl_neurons = 20
    losses = []
    model_dict = {}
    for rel in relations:
        task_layer = Dense(tl_neurons)(semi_final_layer)
        layer_name = rel.replace("/r/", "")
        loss = "mean_squared_error"
        losses.append(loss)

        drd = Model([wv1, wv2], Dense(1)(task_layer),name=layer_name)
        optimizer = Adam(lr=0.0002)
        drd.compile(optimizer=optimizer, loss=[loss])
        # drd.summary()
        # plot_model(drd)
        model_dict[layer_name] = drd
    return model_dict


q = Queue()


def initializer():
    global q
    pass


def trio(relation, start, end):
    tuple = (relation, start, end)
    q.put(tuple)


import numpy as np


def train_on_assertions(model, data, epoch_amount=50, batch_size=1):
    retroembeddings = "retrogan/retroembeddings.h5"
    retrofitted_embeddings = pd.read_hdf(retroembeddings, "mat", encoding='utf-8')

    def load_batch():
        l = len(data.index)
        iterable = list(range(0, l))
        shuffle(iterable)
        for ndx in tqdm(range(0, l, batch_size), ncols=30):
            ixs = iterable[ndx:min(ndx + batch_size, l)]
            stuff = data.iloc[ixs]
            data_dict = {
                "y": float(np.array(stuff[3])[0]),
                "output": str(np.array(stuff[0])[0]),
                "retro_word_1": np.array(retrofitted_embeddings.loc[stuff[1]]),
                "retro_word_2": np.array(retrofitted_embeddings.loc[stuff[2]])
            }
            yield data_dict

    for epoch_amount in range(epoch_amount):
        total_loss = 0
        iter = 0
        for data_dict in load_batch():
            # print(data_dict)
            total_loss+=model[data_dict["output"].replace("/r/", "")].train_on_batch(
                x={'retro_word_1': data_dict["retro_word_1"],
                   'retro_word_2': data_dict["retro_word_2"],
                   },
                y=np.array([data_dict["y"]]))
            iter+=1
            # print(data_dict["output"].replace("/r/", ""),"Loss:", loss)
        print("Avg loss",total_loss/iter)
        print("Saving...")
        save_folder = "drd"
        try:
            os.mkdir(save_folder)
        except Exception as e:
            print(e)
        for key in model.keys():
            model[key].save(save_folder+"/"+key+".model")
            # exit()


def create_data():
    if os.path.exists("valid_rels.hd5"):
        print("Using cache")
        return

    retroembeddings = "retrogan/retroembeddings.h5"
    assertionspath = "retrogan/conceptnet-assertions-5.6.0.csv"
    retrofitted_embeddings = pd.read_hdf(retroembeddings, "mat")
    valid_relations = []
    with open(assertionspath) as assertionsfile:
        assertions = csv.reader(assertionsfile, delimiter="\t")
        row_num = 0
        for assertion_row in assertions:
            row_num += 1
            if row_num % 100000 == 0: print(row_num)
            try:
                a = retrofitted_embeddings.loc[assertion_row[2]]
                b = retrofitted_embeddings.loc[assertion_row[3]]
                if not assertion_row[1] in relations:
                    continue
                info = json.loads(assertion_row[4])
                weight = info["weight"]
                valid_relations.append([assertion_row[1], assertion_row[2], assertion_row[3], weight])
            except Exception as e:
                pass
                # print(e)
            if len(valid_relations) % 10000 == 0:
                print(len(valid_relations))
    af = pd.DataFrame(data=valid_relations, index=range(len(valid_relations)))
    af.to_hdf("valid_rels.hd5", "mat")
    return af


def load_data(path):
    data = pd.read_hdf(path, "mat", encoding="utf-8")
    return data


if __name__ == '__main__':
    model = create_model()
    data = create_data()
    data = load_data("valid_rels.hd5")
    trained_model = train_on_assertions(model, data)

    # Output needs to be the relationship weights
