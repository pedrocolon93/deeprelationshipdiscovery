import csv
import json
import os
import pickle
from random import shuffle

import gc
import numpy as np
import pandas as pd
import sklearn
from conceptnet5.vectors import standardized_concept_uri
from keras import Input, Model
from keras.engine.saving import load_model
from keras.layers import Dense, Concatenate, BatchNormalization, Conv1D, multiply, Dropout
from keras.optimizers import Adam
# from keras.utils import plot_model
from keras.utils import plot_model
from tqdm import tqdm

from retrogan_trainer import attention, ConstMultiplierLayer

relations = ["/r/PartOf", "/r/IsA", "/r/HasA", "/r/UsedFor", "/r/CapableOf", "/r/Desires",
             "/r/AtLocation"
             , "/r/HasSubevent", "/r/HasFirstSubevent", "/r/HasLastSubevent", "/r/HasPrerequisite",
             "/r/HasProperty", "/r/MotivatedByGoal", "/r/ObstructedBy", "/r/CreatedBy", "/r/Synonym",
             "/r/Causes", "/r/Antonym", "/r/DistinctFrom", "/r/DerivedFrom", "/r/SymbolOf", "/r/DefinedAs",
             "/r/SimilarTo", "/r/Entails"]
             # "/r/MannerOf", "/r/RelatedTo",
             # "/r/LocatedNear", "/r/HasContext", "/r/FormOf",  "/r/EtymologicallyRelatedTo",
             # "/r/EtymologicallyDerivedFrom", "/r/CausesDesire", "/r/MadeOf", "/r/ReceivesAction", "/r/InstanceOf",
             # "/r/NotDesires", "/r/NotUsedFor", "/r/NotCapableOf", "/r/NotHasProperty"]


def conv1d(layer_input, filters, f_size=6, strides=1, normalization=True):
    d = Conv1D(filters, f_size, strides=strides, activation="relu")(layer_input)
    return d


def create_model():
    # Input needs to be 2 word vectors
    wv1 = Input(shape=(300,), name="retro_word_1")
    wv2 = Input(shape=(300,), name="retro_word_2")
    expansion_size = 128
    filters = 8
    # Expand and contract the 2 word vectors
    wv1_expansion_1 = Dense(expansion_size * 2)(wv1)
    wv1_expansion_1 = BatchNormalization()(wv1_expansion_1)
    # r_1 = Reshape((-1, 1))(wv1_expansion_1)
    # t1 = conv1d(r_1, filters, f_size=4)
    # f1 = MaxPooling1D(pool_size=4)(t1)
    # f1 = Flatten()(f1)
    # wv1_expansion_2 = attention(f1)
    wv1_expansion_2 = attention(wv1_expansion_1)
    wv1_expansion_3 = Dense(int(expansion_size / 4), activation='relu')(wv1_expansion_2)
    wv1_expansion_3 = BatchNormalization()(wv1_expansion_3)

    wv2_expansion_1 = Dense(expansion_size * 2)(wv2)
    wv2_expansion_1 = BatchNormalization()(wv2_expansion_1)
    # r_2 = Reshape((-1, 1))(wv2_expansion_1)
    # t2 = conv1d(r_2, filters, f_size=4)
    # f2 = MaxPooling1D(pool_size=4)(t2)
    # f2 = Flatten()(f2)
    # wv2_expansion_2 = attention(f2)
    # wv2_expansion_2 = Dense(int(expansion_size / 2),activation='relu')(wv2_expansion_1)
    # wv2_expansion_2 = BatchNormalization()(wv2_expansion_1)
    wv2_expansion_2 = attention(wv2_expansion_1)
    wv2_expansion_3 = Dense(int(expansion_size / 4), activation='relu')(wv2_expansion_2)
    wv2_expansion_3 = BatchNormalization()(wv2_expansion_3)

    # Concatenate both expansions
    merge1 = Concatenate()([wv1_expansion_3, wv2_expansion_3])
    merge_expand = Dense(expansion_size, activation='relu')(merge1)
    merge_expand = BatchNormalization()(merge_expand)
    # Add atention layer
    merge_attention = attention(merge_expand)
    attention_expand = Dense(expansion_size, activation='relu')(merge_attention)
    attention_expand = BatchNormalization()(attention_expand)
    semi_final_layer = Dense(expansion_size, activation='relu')(attention_expand)
    semi_final_layer = BatchNormalization()(semi_final_layer)
    common_layers_model = Model([wv1, wv2], semi_final_layer, name="Common layers")
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
    # prob_model_dict = {}
    # FOR PICS
    model_outs = []
    for rel in relations:
        task_layer = Dense(task_layer_neurons, activation='relu')(semi_final_layer)
        task_layer = BatchNormalization()(task_layer)
        task_layer = attention(task_layer)
        task_layer = Dense(task_layer_neurons, activation='relu')(task_layer)
        task_layer = BatchNormalization()(task_layer)

        layer_name = rel.replace("/r/", "")
        loss = "mean_squared_error"
        losses.append(loss)

        out = Dense(1,name=layer_name)(task_layer)
        # probability = Dense(units=1, activation='sigmoid',name=layer_name+"_prob")(task_layer)
        # scaler = Dense(units=1)(Dropout(0.5)(task_layer))
        # scaled_out = Dense(1)(multiply([probability,scaler]))
        # out = multiply([scale, probability])
        # scale = ConstMultiplierLayer()(probability)

        model_outs.append(out)
        # drdp = Model([wv1, wv2], probability, name=layer_name + "probability")
        drd = Model([wv1, wv2], out, name=layer_name)
        optimizer = Adam(lr=0.0002)
        drd.compile(optimizer=optimizer, loss=[loss])
        # drdp.compile(optimizer=optimizer, loss=[loss])
        # drdp.summary()
        drd.summary()
        plot_model(drd,show_shapes=True)
        model_dict[layer_name] = drd
        # prob_model_dict[layer_name] = drdp

    # plot_model(Model([wv1,wv2],model_outs,name="Deep_Relationship_Discovery"),show_shapes=True,to_file="DRD.png")
    # model_dict["common"]=common_layers_model
    common_layers_model.summary()
    return model_dict#, prob_model_dict


def train_on_assertions(model, data, epoch_amount=100, batch_size=32, save_folder="drd",cutoff=0.6):
    retrofitted_embeddings = pd.read_hdf(retroembeddings, "mat", encoding='utf-8')
    training_data_dict = {}
    training_func_dict = {}
    print("Amount of data:",len(data))
    for i in tqdm(range(len(data))):
        stuff = data.iloc[i]
        if stuff[0] not in training_data_dict.keys():
            training_data_dict[stuff[0]] = []
        training_data_dict[stuff[0]].append(i)

    def load_batch(output_name):
        print("Loading batch for", output_name)
        # TODO LOAD BATCH OF A CERTAIN TYPE
        print("\nDone\n")
        l = len(training_data_dict[output_name])
        iterable = list(range(0, l))
        shuffle(iterable)
        for ndx in range(0, l, batch_size):
            try:
                ixs = iterable[ndx:min(ndx + batch_size, l)]
                x_1 = []
                x_2 = []
                y = []
                for ix in ixs:
                    try:
                        stuff = data.iloc[training_data_dict[output][ix]]
                        l1 = np.array(retrofitted_embeddings.loc[stuff[1]]).reshape(1,300)
                        l2 = np.array(retrofitted_embeddings.loc[stuff[2]]).reshape(1,300)
                        x_1.append(l1)
                        x_2.append(l2)
                        y.append(float(stuff[3]))
                    except Exception as e:
                        # print(e)
                        continue
                # print(np.array(x_1),np.array(x_2),np.array(y))
                yield np.array(x_1), np.array(x_2), np.array(y)
            except Exception as e:
                # print(e)
                return False
        return True

    for epoch in tqdm(range(epoch_amount)):
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
                    x_1, x_2, y = training_func_dict[output].__next__()
                    # print(x_1.shape)
                    x_1 = x_1.reshape(x_1.shape[0],x_1.shape[-1])
                    # print(x_1.shape)
                    x_2 = x_2.reshape(x_2.shape[0], x_2.shape[-1])
                    loss = model[output.replace("/r/", "")].train_on_batch(x={'retro_word_1': x_1, 'retro_word_2': x_2},
                                                                           y=y)
                    # loss_2 = model[output.replace("/r/", "")].train_on_batch(x={'retro_word_1':x_2,'retro_word_2':x_1},y=y)
                    total_loss += loss
                    iter += 1
                    if loss > 10:
                        print("Loss", output, loss)
                    if iter%100:
                        print(loss)
                except Exception as e:
                    # print("Error in", output, str(e))
                    if 'the label' not in str(e):
                        tasks_completed[output] = True
            if False not in tasks_completed.values() or \
                    len([x for x in tasks_completed.values() if x]) / len(tasks_completed.values()) > cutoff:
                for output in exclude:
                    training_func_dict[output] = load_batch(output)
                break
        print("Avg loss", total_loss / iter)
        print(str(epoch) + "/" + str(epoch_amount))
        print("Saving...")
        try:
            os.mkdir(save_folder)
        except Exception as e:
            print(e)
        for key in model.keys():
            model[key].save(save_folder + "/" + key + ".model")
        # for key in prob_model.keys():
        #     prob_model[key].save(save_folder + "/" + key + "probability.model")
            # exit()
        print("Testing")
        model_name = "PartOf"
        test_model(model, model_name=model_name)
        # test_model(prob_model, model_name=model_name)

def create_data(use_cache=True):
    if os.path.exists("tmp/valid_rels.hd5") and use_cache:
        print("Using cache")
        return pd.read_hdf("tmp/valid_rels.hd5", "mat")
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
                c1 = "/c/en/" + c1_split[3]
                c2_split = assertion_row[3].split("/")
                c2 = "/c/en/" + c2_split[3]
                valid_relations.append([assertion_row[1], c1, c2, weight])
            except Exception as e:
                # print(e)
                pass
                # print(e)
            if len(valid_relations) % 10000 == 0:
                print(len(valid_relations))
    af = pd.DataFrame(data=valid_relations, index=range(len(valid_relations)))
    print("Training data:")
    print(af)
    af.to_hdf("tmp/valid_rels.hd5", "mat")
    return af


def load_data(path):
    data = pd.read_hdf(path, "mat", encoding="utf-8")
    return data


def test_model(model_dict, model_name="all", normalizers=None):
    print("testing")
    global w1, w2,w3
    res1 = model_dict[model_name].predict(x={"retro_word_1": w1,
                                            "retro_word_2": w2})
    res2 = model_dict[model_name].predict(x={"retro_word_1": w1,
                                            "retro_word_2": w3})
    print(res1,res2)

    if normalizers is not None:
        norm_res = normalizers[model_name].transform(res1)
        print(norm_res)
        norm_res = normalizers[model_name].transform(res2)
        print(norm_res)


def normalize_outputs(model, save_folder="./drd", use_cache=True):
    save_path = save_folder + "/" + "normalization_dict.pickle"
    if use_cache:
        try:
            norm_dict = pickle.load(open(save_path, 'rb'))
            return norm_dict
        except Exception as e:
            print(e)
            print("Cache not found")

    # Set up our variables
    normalization_dict = {}
    x1 = []
    x2 = []
    y = []

    # Load our data
    retrofitted_embeddings = pd.read_hdf(retroembeddings, "mat")
    rand_range = [x for x in range(len(retrofitted_embeddings.index))]
    shuffle(rand_range)

    for i in range(len(retrofitted_embeddings.index)):
        x1.append(np.array(retrofitted_embeddings.iloc[rand_range[i]]).reshape(1,300))
        x2.append(np.array(retrofitted_embeddings.iloc[i]).reshape(1, 300))
    for rel in relations:
        rel_normer = sklearn.preprocessing.MinMaxScaler().fit(
            model[rel.replace("/r/", "")].predict(x={"retro_word_1": np.array(x1).reshape(len(x1), 300),
                                                     "retro_word_2": np.array(x2).reshape(len(x1), 300)}))
        normalization_dict[rel.replace("/r/", "")] = rel_normer
        print(rel_normer)
    pickle.dump(normalization_dict, open(save_path, "wb"))
    return normalization_dict


def load_model_ours(save_folder="./drd", model_name="all",probability_models=False):
    model_dict = {}
    if model_name == 'all':
        for rel in relations:
            print("Loading", rel)
            layer_name = rel.replace("/r/", "")
            if probability_models:
                print("Loading",save_folder + "/" + layer_name + "probability.model")
                model_dict[layer_name] = load_model(save_folder + "/" + layer_name + "probability.model",
                                                    custom_objects={"ConstMultiplierLayer": ConstMultiplierLayer})
                model_dict[layer_name].summary()
            else:
                model_dict[layer_name] = load_model(save_folder + "/" + layer_name + ".model",
                                                    custom_objects={"ConstMultiplierLayer": ConstMultiplierLayer})

    else:
        layer_name = model_name.replace("/r/", "")
        if not probability_models:
            print("Loading models")
            model_dict[layer_name] = load_model(save_folder + "/" + layer_name + ".model",
                                                custom_objects={"ConstMultiplierLayer": ConstMultiplierLayer})
            print("Loading weights")
            model_dict[layer_name].load_weights(save_folder + "/" + layer_name + ".model")
        else:
            print("Loading models")
            model_dict[layer_name] = load_model(save_folder + "/" + layer_name + "probability.model",
                                                custom_objects={"ConstMultiplierLayer": ConstMultiplierLayer})
            print("Loading weights")
            model_dict[layer_name].load_weights(save_folder + "/" + layer_name + "probability.model")

    # model_dict["common"] = load_model(save_folder + "/" + "common" + ".model",
    #                                         custom_objects={"ConstMultiplierLayer": ConstMultiplierLayer})
    return model_dict

# retroembeddings = "trained_models/retroembeddings/2019-04-0813:03:02.430691/retroembeddings.h5"
retroembeddings = "trained_models/retroembeddings/2019-05-15 11:47:52.802481/retroembeddings.h5"

if __name__ == '__main__':
    # # save_folder =     "./trained_models/deepreldis/"+str(datetime.datetime.now())
    retrofitted_embeddings = pd.read_hdf(retroembeddings, "mat")
    global w1, w2, w3
    w1 = np.array(retrofitted_embeddings.loc[standardized_concept_uri("en", "building")]).reshape(1, 300)
    w2 = np.array(retrofitted_embeddings.loc[standardized_concept_uri("en", "photography")]).reshape(1, 300)
    w3 = np.array(retrofitted_embeddings.loc[standardized_concept_uri("en", "surfing")]).reshape(1, 300)
    model_name = "UsedFor"
    # del retrofitted_embeddings
    # gc.collect()
    print("Creating model...")
    model = create_model()
    print("Done\nLoading data")
    # model = load_model_ours()
    data = create_data(use_cache=False)
    # # data = load_data("valid_rels.hd5")
    print("Done\nTraining")
    train_on_assertions(model, data)
    print("Done\n")
    # model = load_model_ours(save_folder="trained_models/deepreldis/2019-05-28",model_name=model_name)
    # model = load_model_ours(save_folder="trained_models/deepreldis/2019-04-25_2_sigmoid",model_name=model_name,probability_models=True)
    normalizers = normalize_outputs(model,save_folder="trained_models/deepreldis/2019-05-28")
    # normalizers = normalize_outputs(model,use_cache=False)
    test_model(model, normalizers=normalizers, model_name=model_name)
    # Output needs to be the relationship weights
