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

import pandas as pd
import csv
from keras.engine.saving import load_model
from conceptnet5.vectors import standardized_concept_uri

from retrogan_trainer_small import attention, ConstMultiplierLayer

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
    expansion_size = 512
    # Expand and contract the 2 word vectors
    wv1_expansion_1 = Dense(expansion_size)(wv1)
    wv1_expansion_1 = BatchNormalization()(wv1_expansion_1)
    wv1_expansion_2 = Dense(int(expansion_size / 2),activation='relu')(wv1_expansion_1)
    wv1_expansion_2 = BatchNormalization()(wv1_expansion_2)
    wv1_expansion_2 = attention(wv1_expansion_2)
    wv1_expansion_3 = Dense(int(expansion_size / 4),activation='relu')(wv1_expansion_2)
    wv1_expansion_3 = BatchNormalization()(wv1_expansion_3)


    wv2_expansion_1 = Dense(expansion_size)(wv2)
    wv2_expansion_1 = BatchNormalization()(wv2_expansion_1)
    wv2_expansion_2 = Dense(int(expansion_size / 2),activation='relu')(wv2_expansion_1)
    wv2_expansion_2 = BatchNormalization()(wv2_expansion_2)
    wv2_expansion_2 = attention(wv2_expansion_2)
    wv2_expansion_3 = Dense(int(expansion_size / 4),activation='relu')(wv2_expansion_2)
    wv2_expansion_3 = BatchNormalization()(wv2_expansion_3)

    # Concatenate both expansions
    merge1 = Concatenate()([wv1_expansion_3, wv2_expansion_3])
    merge_expand = Dense(expansion_size,activation='relu')(merge1)
    merge_expand = BatchNormalization()(merge_expand)
    # Add atention layer
    merge_attention = attention(merge_expand)
    attention_expand = Dense(expansion_size,activation='relu')(merge_attention)
    attention_expand = BatchNormalization()(attention_expand)
    semi_final_layer = Dense(expansion_size,activation='relu')(attention_expand)
    common_layers_model = Model([wv1, wv2],semi_final_layer,name="Common layers")
    # Output layer
    amount_of_relations = len(relations)
    # One big layer
    # final = Dense(amount_of_relations)(semi_final_layer)
    # Many tasks
    tl_neurons = 20
    losses = []
    model_dict = {}
    for rel in relations:
        task_layer = Dense(tl_neurons,activation='relu')(common_layers_model.output)
        task_layer = BatchNormalization()(task_layer)
        layer_name = rel.replace("/r/", "")
        loss = "mean_squared_error"
        losses.append(loss)

        drd = Model([wv1, wv2], Dense(1)(task_layer),name=layer_name)
        optimizer = Adam(lr=0.0002)
        drd.compile(optimizer=optimizer, loss=[loss])
        # drd.summary()
        # plot_model(drd)
        model_dict[layer_name] = drd
    model_dict["common"]=common_layers_model
    return model_dict


q = Queue()


def initializer():
    global q
    pass


def trio(relation, start, end):
    tuple = (relation, start, end)
    q.put(tuple)


import numpy as np


def train_on_assertions(model, data, epoch_amount=50, batch_size=1,save_folder = "drd"):
    retroembeddings = "retroembeddings.h5"
    retrofitted_embeddings = pd.read_hdf(retroembeddings, "mat", encoding='utf-8')

    def load_batch():
        l = len(data.index)
        iterable = list(range(0, l))
        shuffle(iterable)
        for ndx in tqdm(range(0, l, batch_size), ncols=30):
            try:
                ixs = iterable[ndx:min(ndx + batch_size, l)]
                stuff = data.iloc[ixs]
                data_dict = {
                    "y": float(np.array(stuff[3])[0]),
                    "output": str(np.array(stuff[0])[0]),
                    "retro_word_1": np.array(retrofitted_embeddings.loc[stuff[1].iloc[-1]]),
                    "retro_word_2": np.array(retrofitted_embeddings.loc[stuff[2].iloc[-1]])
                }
                yield data_dict
            except Exception as e:
                print(e)
                print("Continuing")

    for epoch_amount in range(epoch_amount):
        total_loss = 0
        iter = 0
        exclude = ["/r/RelatedTo", "/r/FormOf", "/r/IsA", "/r/PartOf", "/r/HasA", "/r/UsedFor", "/r/CapableOf",
             "/r/AtLocation",
             "/r/Causes", "/r/HasSubevent", "/r/HasFirstSubevent", "/r/HasLastSubevent", "/r/HasPrerequisite",
             "/r/HasProperty", "/r/MotivatedByGoal", "/r/ObstructedBy", "/r/Desires", "/r/CreatedBy", "/r/Synonym",
             "/r/Antonym", ]
        for data_dict in load_batch():
            if data_dict["output"] not in exclude:
                continue
            # print(data_dict)
            loss = model[data_dict["output"].replace("/r/", "")].train_on_batch(
                x={'retro_word_1': np.array([data_dict["retro_word_1"],data_dict["retro_word_2"]]),
                   'retro_word_2': np.array([data_dict["retro_word_2"],data_dict["retro_word_1"]])
                   },
                y=np.array([data_dict["y"],data_dict["y"]]))
            if loss>15:
                print("Loss",data_dict["output"].replace("/r/", ""),loss)
            total_loss+=loss
            iter+=1
            # print(data_dict["output"].replace("/r/", ""),"Loss:", loss)
        print("Avg loss",total_loss/iter)
        print("Saving...")

        try:
            os.mkdir(save_folder)
        except Exception as e:
            print(e)
        for key in model.keys():
            model[key].save(save_folder+"/"+key+".model")
            # exit()


def create_data(use_cache=True):
    if os.path.exists("valid_rels.hd5") and use_cache:
        print("Using cache")
        return pd.read_hdf("valid_rels.hd5","mat")

    retroembeddings = "retroembeddings.h5"
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


def test_model(model_dict):
    print("testing")
    retroembeddings = "retroembeddings.h5"
    retrofitted_embeddings = pd.read_hdf(retroembeddings, "mat")
    w1 = np.array(retrofitted_embeddings.loc[standardized_concept_uri("en","dog")]).reshape(1,300)
    w2 = np.array(retrofitted_embeddings.loc[standardized_concept_uri("en","animal")]).reshape(1,300)
    res = model_dict["IsA"].predict(x={"retro_word_1":w1,
                                     "retro_word_2":w2})
    print(res)

def load_model_ours(save_folder = "./drd",model_name="all"):
    model_dict = {}
    if model_name == 'all':
        for rel in relations:
            print("Loading",rel)
            layer_name = rel.replace("/r/", "")
            model_dict[layer_name] = load_model(save_folder+"/"+layer_name+".model",custom_objects={"ConstMultiplierLayer":ConstMultiplierLayer})
    else:
        layer_name = model_name.replace("/r/", "")
        model_dict[layer_name] = load_model(save_folder + "/" + layer_name + ".model",
                                            custom_objects={"ConstMultiplierLayer": ConstMultiplierLayer})
    model_dict["common"] = load_model(save_folder + "/" + "common" + ".model",
                                            custom_objects={"ConstMultiplierLayer": ConstMultiplierLayer})
    return model_dict


if __name__ == '__main__':
    model = create_model()
    # model = load_model_ours()
    data = create_data(use_cache=True)
    # data = load_data("valid_rels.hd5")
    train_on_assertions(model, data)
    model = load_model_ours(model_name="UsedFor")
    test_model(model)
    # Output needs to be the relationship weights
