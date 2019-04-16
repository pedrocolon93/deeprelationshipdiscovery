import csv
import json
import os
from multiprocessing import Queue
from random import shuffle

import gc
import pandas as pd
from conceptnet5.vectors import standardized_concept_uri
from keras import Input, Model
from keras.engine.saving import load_model
from keras.layers import Dense, Concatenate, BatchNormalization, Conv1D, Reshape, MaxPooling1D, Flatten
from keras.optimizers import Adam
from tqdm import tqdm

from retrogan_trainer import attention, ConstMultiplierLayer

relations = ["/r/PartOf","/r/IsA", "/r/HasA", "/r/UsedFor", "/r/CapableOf", "/r/Desires",
             "/r/AtLocation",
             "/r/Causes", "/r/HasSubevent", "/r/HasFirstSubevent", "/r/HasLastSubevent", "/r/HasPrerequisite",
             "/r/HasProperty", "/r/MotivatedByGoal", "/r/ObstructedBy", "/r/CreatedBy", "/r/Synonym",
             "/r/Antonym", "/r/DistinctFrom", "/r/DerivedFrom", "/r/SymbolOf", "/r/DefinedAs", "/r/Entails",
             "/r/MannerOf","/r/RelatedTo",
             "/r/LocatedNear", "/r/HasContext", "/r/FormOf","/r/SimilarTo", "/r/EtymologicallyRelatedTo",
             "/r/EtymologicallyDerivedFrom", "/r/CausesDesire", "/r/MadeOf", "/r/ReceivesAction", "/r/InstanceOf",
             "/r/NotDesires", "/r/NotUsedFor", "/r/NotCapableOf", "/r/NotHasProperty"]

def conv1d(layer_input, filters, f_size=6, strides=1, normalization=True):
    d = Conv1D(filters, f_size, strides=strides, activation="relu")(layer_input)
    return d

def create_model():
    # Input needs to be 2 word vectors
    wv1 = Input(shape=(300,), name="retro_word_1")
    wv2 = Input(shape=(300,), name="retro_word_2")
    expansion_size = 512
    filters = 8
    # Expand and contract the 2 word vectors
    wv1_expansion_1 = Dense(expansion_size*2)(wv1)
    wv1_expansion_1 = BatchNormalization()(wv1_expansion_1)
    # r_1 = Reshape((-1, 1))(wv1_expansion_1)
    # t1 = conv1d(r_1, filters, f_size=4)
    # f1 = MaxPooling1D(pool_size=4)(t1)
    # f1 = Flatten()(f1)
    # wv1_expansion_2 = attention(f1)
    wv1_expansion_2 = attention(wv1_expansion_1)
    wv1_expansion_3 = Dense(int(expansion_size / 4),activation='relu')(wv1_expansion_2)
    wv1_expansion_3 = BatchNormalization()(wv1_expansion_3)

    wv2_expansion_1 = Dense(expansion_size*2)(wv2)
    wv2_expansion_1 = BatchNormalization()(wv2_expansion_1)
    # r_2 = Reshape((-1, 1))(wv2_expansion_1)
    # t2 = conv1d(r_2, filters, f_size=4)
    # f2 = MaxPooling1D(pool_size=4)(t2)
    # f2 = Flatten()(f2)
    # wv2_expansion_2 = attention(f2)
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
    semi_final_layer = BatchNormalization()(semi_final_layer)
    # common_layers_model = Model([wv1, wv2],semi_final_layer,name="Common layers")
    # common_optimizer = Adam(lr=0.000002)
    # common_layers_model.compile()
    # Output layer
    # amount_of_relations = len(relations)
    # One big layer
    # final = Dense(amount_of_relations)(semi_final_layer)
    # Many tasks
    task_layer_neurons = 256
    losses = []
    model_dict = {}
    for rel in relations:
        task_layer = Dense(task_layer_neurons,activation='relu')(semi_final_layer)
        task_layer = BatchNormalization()(task_layer)
        task_layer = attention(task_layer)
        task_layer = Dense(task_layer_neurons,activation='relu')(task_layer)
        task_layer = BatchNormalization()(task_layer)

        layer_name = rel.replace("/r/", "")
        loss = "mean_squared_error"
        losses.append(loss)

        drd = Model([wv1, wv2], Dense(1)(task_layer),name=layer_name)
        optimizer = Adam(lr=0.0002)
        drd.compile(optimizer=optimizer, loss=[loss])
        drd.summary()
        # plot_model(drd)
        model_dict[layer_name] = drd
    # model_dict["common"]=common_layers_model
    # common_layers_model.summary()
    return model_dict


q = Queue()


def initializer():
    global q
    pass


def trio(relation, start, end):
    tuple = (relation, start, end)
    q.put(tuple)


import numpy as np


def train_on_assertions(model, data, epoch_amount=100, batch_size=32,save_folder = "drd"):

    retroembeddings = "trained_models/retroembeddings/2019-04-0813:03:02.430691/retroembeddings.h5"
    # retroembeddings = "retrogan/numberbatch.h5"
    retrofitted_embeddings = pd.read_hdf(retroembeddings, "mat", encoding='utf-8')
    training_data_dict = {}
    training_func_dict = {}
    def load_batch(output_name):
        print("Loading batch for",output_name)
        #TODO LOAD BATCH OF A CERTAIN TYPE
        if output_name not in training_data_dict.keys():
            training_data_dict[output] = []
            for i in tqdm(range(len(data))):
                stuff = data.iloc[i]
                if str(stuff[0])==output_name:
                    training_data_dict[output].append(i)
        print("\nDone\n")
        l = len(training_data_dict[output_name])
        iterable = list(range(0, l))
        shuffle(iterable)
        for ndx in tqdm(range(0, l, batch_size)):
            try:
                ixs = iterable[ndx:min(ndx + batch_size, l)]
                x_1 = []
                x_2 = []
                y = []
                for ix in ixs:
                    try:
                        stuff = data.iloc[training_data_dict[output][ix]]
                        l1 = np.array(retrofitted_embeddings.loc[stuff[1]])
                        l2 = np.array(retrofitted_embeddings.loc[stuff[2]])
                        x_1.append(l1)
                        x_2.append(l2)
                        y.append(float(stuff[3]))
                    except:
                        continue
                # print(np.array(x_1),np.array(x_2),np.array(y))
                yield np.array(x_1),np.array(x_2),np.array(y)
            except Exception as e:
                print(e)
                return False
        return True


    for epoch in range(epoch_amount):
        total_loss = 0
        iter = 0
        exclude = relations
        shuffle(exclude)
        for output in exclude:
            training_func_dict[output] = load_batch(output)
        window_loss = 0
        tasks_completed = {}
        for task in exclude:
            tasks_completed[task] = False
        while True:
            for output in exclude:
                try:
                    x_1,x_2,y = training_func_dict[output].__next__()
                    # print(x_1.shape)
                    # print(x_2.shape)
                    # print(y.shape)
                    loss = model[output.replace("/r/", "")].train_on_batch(x={'retro_word_1':x_1,'retro_word_2':x_2},y=y)
                    # loss_2 = model[output.replace("/r/", "")].train_on_batch(x={'retro_word_1':x_2,'retro_word_2':x_1},y=y)
                    total_loss+=loss
                    iter+=1
                    if loss>10:
                        print("Loss",output,loss)
                except Exception as e:
                    # print("Error in",output,str(e))
                    if 'the label' not in str(e):
                        tasks_completed[output] = True
            if False not in tasks_completed.values() or \
                    len([x for x in tasks_completed.values() if x])/len(tasks_completed.values())>0.6:
                for output in exclude:
                    training_func_dict[output] = load_batch(output)
                break
        print("Avg loss",total_loss/iter)
        print(str(epoch)+"/"+str(epoch_amount))
        print("Saving...")
        try:
            os.mkdir(save_folder)
        except Exception as e:
            print(e)
        for key in model.keys():
            model[key].save(save_folder+"/"+key+".model")
            # exit()
        print("Testing")
        model_name = "PartOf"
        test_model(model, model_name=model_name)

def create_data(use_cache=True):
    if os.path.exists("valid_rels.hd5") and use_cache:
        print("Using cache")
        return pd.read_hdf("valid_rels.hd5","mat")

    assertionspath = "retrogan/conceptnet-assertions-5.6.0.csv"
    valid_relations = []
    with open(assertionspath) as assertionsfile:
        assertions = csv.reader(assertionsfile, delimiter="\t")
        row_num = 0
        for assertion_row in assertions:
            row_num += 1
            if row_num % 100000 == 0: print(row_num)
            try:
                if "/c/en/" not in assertion_row[2] or "/c/en/" not in assertion_row[3] or \
                        assertion_row[1] not in relations:
                    continue
                info = json.loads(assertion_row[4])
                weight = info["weight"]
                c1_split = assertion_row[2].split("/")
                # print(c1_split)
                c1 = "/c/en/"+c1_split[3]
                c2_split = assertion_row[3].split("/")
                c2 = "/c/en/"+c2_split[3]
                valid_relations.append([assertion_row[1], c1,c2, weight])
            except Exception as e:
                print(e)
                # pass
                # print(e)
            if len(valid_relations) % 10000 == 0:
                print(len(valid_relations))
    af = pd.DataFrame(data=valid_relations, index=range(len(valid_relations)))
    af.to_hdf("valid_rels.hd5", "mat")
    return af


def load_data(path):
    data = pd.read_hdf(path, "mat", encoding="utf-8")
    return data


def test_model(model_dict,model_name):
    print("testing")
    global w1,w2
    res = model_dict[model_name].predict(x={"retro_word_1":w1,
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
        print("Loading models")
        model_dict[layer_name] = load_model(save_folder + "/" + layer_name + ".model",
                                            custom_objects={"ConstMultiplierLayer": ConstMultiplierLayer})
        print("Loading weights")
        model_dict[layer_name].load_weights(save_folder + "/" + layer_name + ".model")
    model_dict["common"] = load_model(save_folder + "/" + "common" + ".model",
                                            custom_objects={"ConstMultiplierLayer": ConstMultiplierLayer})
    return model_dict


if __name__ == '__main__':
    # save_folder =     "./trained_models/deepreldis/"+str(datetime.datetime.now())
    retroembeddings = "trained_models/retroembeddings/2019-04-0813:03:02.430691/retroembeddings.h5"
    retrofitted_embeddings = pd.read_hdf(retroembeddings, "mat")
    global w1,w2
    w1 = np.array(retrofitted_embeddings.loc[standardized_concept_uri("en","table")]).reshape(1,300)
    w2 = np.array(retrofitted_embeddings.loc[standardized_concept_uri("en","car")]).reshape(1,300)
    del retrofitted_embeddings
    # gc.collect()
    # print("Creating model...")
    # model = create_model()
    # print("Done\nLoading data")
    # # model = load_model_ours()
    # data = create_data(use_cache=False)
    # # data = load_data("valid_rels.hd5")
    # print("Done\nTraining")
    # train_on_assertions(model, data)
    # print("Done\n")
    model_name = "IsA"
    model = load_model_ours(save_folder="trained_models/deepreldis/2019-04-1613:43:00.000000",model_name=model_name)
    test_model(model,model_name=model_name)
    # Output needs to be the relationship weights
