import csv
import datetime
import os
import pickle
import random
import pandas as pd
from random import shuffle

from tqdm import tqdm
import sklearn
import numpy as np
import tensorflow as tf
# from keras.utils import plot_model
from tensorflow_core.python.keras.saving.save import load_model
from tensorflow_core.python.keras.utils.vis_utils import plot_model

# from retrogan_trainer_attractrepel_working import *
from rcgan import RetroCycleGAN
from tools import generate_fastext_embedding

relations = ["/r/PartOf", "/r/IsA", "/r/HasA", "/r/UsedFor", "/r/CapableOf", "/r/Desires", #6
             "/r/AtLocation", "/r/HasSubevent", "/r/HasFirstSubevent", "/r/HasLastSubevent", "/r/HasPrerequisite", #5
             "/r/HasProperty", "/r/MotivatedByGoal", "/r/ObstructedBy", "/r/CreatedBy", "/r/Synonym", #5
             "/r/Causes", "/r/Antonym", "/r/DistinctFrom", "/r/DerivedFrom", "/r/SymbolOf", "/r/DefinedAs", #6
             "/r/SimilarTo", "/r/Entails", #2
             "/r/MannerOf", "/r/RelatedTo",#2
             "/r/LocatedNear", "/r/HasContext", "/r/FormOf",  "/r/EtymologicallyRelatedTo", #4
             "/r/EtymologicallyDerivedFrom", "/r/CausesDesire", "/r/MadeOf", "/r/ReceivesAction", "/r/InstanceOf", #5
             "/r/NotDesires", "/r/NotUsedFor", "/r/NotCapableOf", "/r/NotHasProperty","/r/NotIsA","/r/NotHasA"] # 4



# def conv1d(layer_input, filters, f_size=6, strides=1, normalization=True):
#     d = Conv1D(filters, f_size, strides=strides, activation="relu")(layer_input)
#     return d


def create_model():
    # Input needs to be 2 word vectors
    wv1 = tf.keras.layers.Input(shape=(300,), name="retro_word_1")
    wv2 = tf.keras.layers.Input(shape=(300,), name="retro_word_2")
    expansion_size = 1024
    intermediate_size = 512

    def create_word_input_abstraction(wv1):
        # Expand and contract the 2 word vectors
        wv1_expansion_1 = tf.keras.layers.Dense(expansion_size)(wv1)
        wv1_expansion_1 = tf.keras.layers.BatchNormalization()(wv1_expansion_1)
        wv1_expansion_1 = tf.keras.layers.Dropout(0.1)(wv1_expansion_1)
        # r_1 = Reshape((-1, 1))(wv1_expansion_1)
        # t1 = conv1d(r_1, filters, f_size=4)
        # f1 = MaxPooling1D(pool_size=4)(t1)
        # f1 = Flatten()(f1)
        # wv1_expansion_2 = attention(f1)
        # wv1_expansion_2 = attention(wv1_expansion_1)
        wv1_expansion_3 = tf.keras.layers.Dense(int(expansion_size / 4), activation='relu')(wv1_expansion_1)
        wv1_expansion_3 = tf.keras.layers.BatchNormalization()(wv1_expansion_3)
        return wv1_expansion_3

    wv1_expansion_3 = create_word_input_abstraction(wv1)
    wv2_expansion_3 = create_word_input_abstraction(wv2)

    # Concatenate both expansions
    merge1 = tf.keras.layers.Concatenate()([wv1_expansion_3, wv2_expansion_3])
    merge1 = tf.keras.layers.Dropout(0.1)(merge1)
    merge_expand = tf.keras.layers.Dense(intermediate_size, activation='relu')(merge1)
    merge_expand = tf.keras.layers.BatchNormalization()(merge_expand)
    # Add atention layer
    # merge_attention = attention(merge_expand)
    attention_expand = tf.keras.layers.Dense(intermediate_size, activation='relu')(merge_expand)
    attention_expand = tf.keras.layers.BatchNormalization()(attention_expand)
    attention_expand = tf.keras.layers.Dropout(0.1)(attention_expand)

    semi_final_layer = tf.keras.layers.Dense(intermediate_size, activation='relu')(attention_expand)
    semi_final_layer = tf.keras.layers.BatchNormalization()(semi_final_layer)
    common_layers_model = tf.keras.Model([wv1, wv2], semi_final_layer, name="Common layers")
    # common_optimizer = Adam(lr=0.000002)
    # common_layers_model.compile()
    # Output layer
    # amount_of_relations = len(relations)
    # One big layer
    # final = Dense(amount_of_relations)(semi_final_layer)
    # Many tasks
    task_layer_neurons = 512
    losses = []
    model_dict = {}
    callback_dict = {}
    # prob_model_dict = {}
    # FOR PICS
    model_outs = []
    for rel in relations:
        task_layer = tf.keras.layers.Dense(task_layer_neurons, activation='relu')(semi_final_layer)
        task_layer = tf.keras.layers.BatchNormalization()(task_layer)
        task_layer = tf.keras.layers.Dropout(0.1)(task_layer)

        # task_layer = attention(task_layer)
        task_layer = tf.keras.layers.Dense(task_layer_neurons, activation='relu')(task_layer)
        task_layer = tf.keras.layers.BatchNormalization()(task_layer)

        layer_name = rel.replace("/r/", "")
        # loss = "mean_squared_error"
        loss = "binary_crossentropy"
        losses.append(loss)

        out = tf.keras.layers.Dense(1,name=layer_name,activation='sigmoid')(task_layer)
        # probability = Dense(units=1, activation='sigmoid',name=layer_name+"_prob")(task_layer)
        # scaler = Dense(units=1)(Dropout(0.5)(task_layer))
        # scaled_out = Dense(1)(multiply([probability,scaler]))
        # out = multiply([scale, probability])
        # scale = ConstMultiplierLayer()(probability)

        model_outs.append(out)
        # drdp = Model([wv1, wv2], probability, name=layer_name + "probability")
        drd = tf.keras.Model([wv1, wv2], out, name=layer_name)
        optimizer = tf.keras.optimizers.Adam(lr=0.0002)
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


def train_on_assertions(model, data, epoch_amount=100, batch_size=32, save_folder="drd",cutoff=0.63):
    # retrofitted_embeddings = pd.read_hdf(retroembeddings, "mat", encoding='utf-8')
    training_data_dict = {}
    training_func_dict = {}
    print("Amount of data:",len(data))
    for i in tqdm(range(len(data))):
        stuff = data.iloc[i]
        rel = stuff[0]
        if rel not in training_data_dict.keys():
            training_data_dict[rel] = []
        training_data_dict[rel].append(i)

    def load_batch(output_name):
        print("Loading batch for", output_name)
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
                        l1 = np.array(retrofitted_embeddings.loc[stuff[1]]).reshape(1,300).astype(np.float32)
                        l2 = np.array(retrofitted_embeddings.loc[stuff[2]]).reshape(1,300).astype(np.float32)
                        x_1.append(l1)
                        x_2.append(l2)
                        # y.append(float(stuff[3]))
                        y.append(1)
                    except Exception as e:
                        print(e)
                        continue

                for idx in range(len(y)):
                    x1 = x_1[idx]
                    x2 = x_2[idx]
                    # print(idx)
                    ix1 = random.sample(range(0, len(y)), 1)
                    ix2 = random.sample(range(0, len(y)), 1)
                    # print(ix1)
                    # print(ix2)
                    x_1.append(x_1[ix1[0]])
                    x_2.append(x2)
                    y.append(0)
                    x_1.append(x1)
                    x_2.append(x_2[ix2[0]])
                    y.append(0)
                    idx+=1
                # print(np.array(x_1),np.array(x_2),np.array(y))
                print("Shuffling in the confounders")
                c = list(zip(x_1, x_2,y))
                random.shuffle(c)
                x_1,x_2,y = zip(*c)
                print("Done")
                yield np.array(x_1), np.array(x_2), np.array(y)
            except Exception as e:
                print(e)
                return False
        return True
    callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/")
    # callback.set_model(drd)
    # callback_dict[layer_name] = callback
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
        iter = 0
        while True:
            for output in exclude:
                try:
                    it = training_func_dict[output]
                    x_1, x_2, y = it.__next__()
                    # print(x_1.shape)
                    x_1 = x_1.reshape(x_1.shape[0],x_1.shape[-1])
                    # print(x_1.shape)
                    x_2 = x_2.reshape(x_2.shape[0], x_2.shape[-1])
                    try:
                        loss = model[output.replace("/r/", "")].train_on_batch(x={'retro_word_1': x_1, 'retro_word_2': x_2},
                                                                               y=y)
                    except:
                        loss = model[output.replace("/r/", "")].train_on_batch(x={'input_1': x_1, 'input_2': x_2},
                                                                               y=y)

                    callback.on_epoch_end(iter, {"loss":loss})

                    # loss_2 = model[output.replace("/r/", "")].train_on_batch(x={'retro_word_1':x_2,'retro_word_2':x_1},y=y)
                    total_loss += loss
                    iter += 1
                    # if loss > 10:
                    #     print("Loss", output, loss)
                    # if iter%100:
                    #     print(loss)
                except KeyError as e:
                    print("No data for", e)
                except StopIteration as e:
                    print("Done with",output)
                    tasks_completed[output] = True
                    print("Resetting the iterator")
                    training_func_dict[output] = load_batch(output)

                # except Exception as e:
                #     print(e)
                #     print("Error in", output, str(e))
                    # print(training_func_dict)
                    #
                    # if 'the label' not in str(e):
                    #     tasks_completed[output] = True
                    # else:
                    #     print(e)
            # print(len([x for x in tasks_completed.values() if x]),"/", len(tasks_completed.values()))
            if len([x for x in tasks_completed.values() if x]) / len(tasks_completed.values()) > cutoff:
                print(tasks_completed)
                break
            else:
                print(tasks_completed)
            iter += 1

        print("Avg loss", total_loss / iter)
        print(str(epoch) + "/" + str(epoch_amount))

        # for key in prob_model.keys():
        #     prob_model[key].save(save_folder + "/" + key + "probability.model")
            # exit()
        # print("Testing")
        # model_name = "PartOf"
        # test_model(model, model_name=model_name)
        if epoch%10==0:
            try:
                os.mkdir(save_folder)
            except Exception as e:
                print(e)
            for key in model.keys():
                model[key].save(save_folder + "/" + key + ".model")
        # test_model(prob_model, model_name=model_name)
    print("Saving...")
    try:
        os.mkdir(save_folder)
    except Exception as e:
        print(e)
    for key in model.keys():
        model[key].save(save_folder + "/" + key + ".model")
# def create_data(use_cache=True):
#     if os.path.exists("tmp/valid_rels.hd5") and use_cache:
#         print("Using cache")
#         return pd.read_hdf("tmp/valid_rels.hd5", "mat")
#     assertionspath = "retrogan/conceptnet-assertions-5.6.0.csv"
#     valid_relations = []
#     with open(assertionspath) as assertionsfile:
#         assertions = csv.reader(assertionsfile, delimiter="\t")
#         row_num = 0
#         for assertion_row in assertions:
#             row_num += 1
#             if row_num % 100000 == 0: print(row_num)
#             try:
#                 if "/c/en/" not in assertion_row[2] or "/c/en/" not in assertion_row[3] or \
#                         assertion_row[1] not in relations:
#                     continue
#                 info = json.loads(assertion_row[4])
#                 weight = info["weight"]
#                 c1_split = assertion_row[2].split("/")
#                 # print(c1_split)
#                 c1 = "/c/en/" + c1_split[3]
#                 c2_split = assertion_row[3].split("/")
#                 c2 = "/c/en/" + c2_split[3]
#                 valid_relations.append([assertion_row[1], c1, c2, weight])
#             except Exception as e:
#                 # print(e)
#                 pass
#                 # print(e)
#             if len(valid_relations) % 10000 == 0:
#                 print(len(valid_relations))
#

def create_sentence_embedding(c1, retrofitted_embeddings,rcgan,newlist):
    s = c1.split(" ")
    concept_vecs = []
    if len(s)>1:
        for word in s:
            try:
                concept_vecs.append(retrofitted_embeddings.loc[word])
            except:
                print("Creating emb for",word)
                added = False
                for tup in newlist:
                    if word == tup[0]:
                        concept_vecs.append(tup[1])
                        added = True
                        break
                if not added:
                    concept_vecs.append(pd.Series(rcgan.g_AB.predict(np.array(generate_fastext_embedding(word))
                                                           .reshape(1, 300)
                                                           ).reshape(300)))
    concept_vecs = np.array(concept_vecs)
    avg = np.mean(concept_vecs,axis=0)
    return (c1,pd.Series(avg))



def create_data2(use_cache=True):
    global retrofitted_embeddings
    rcgan = RetroCycleGAN(save_folder="test", batch_size=32, generator_lr=0.0001, discriminator_lr=0.001)
    rcgan.load_weights(preface="final", folder="trained_models/retrogans/ft_full_alldata_feb11")

    if os.path.exists("tmp/valid_rels.hd5") and use_cache:
        print("Using cache")
        a = pd.read_hdf("tmp/valid_rels.hd5", "mat")
        b = pd.read_hdf("tmp/updated_embeddings.hd5", "mat")
        b.drop_duplicates(inplace=True)
        return a, b
    assertionspath = "train600k.txt"
    valid_relations = []
    new = []
    with open(assertionspath) as assertionsfile:
        assertions = csv.reader(assertionsfile, delimiter="\t")
        row_num = 0
        skipped = 0
        for assertion_row in tqdm(assertions):
            # if row_num>100: break
            row_num += 1
            if row_num % 100000 == 0: print(row_num)
            try:
                rel = assertion_row[0]
                if "/r/" not in rel: rel = "/r/"+rel
                weight = float(assertion_row[3])
                # print(c1_split)
                c1 = assertion_row[1]
                c2 = assertion_row[2]

                if c1 not in retrofitted_embeddings.index or \
                        c2 not in retrofitted_embeddings.index or \
                        rel not in relations:
                    if rel not in relations:
                        print("Skipping relation",rel)
                        skipped+=1
                        continue
                    # try:
                    if len(c1.split(" "))>1:
                        #We have a sentence so create the embedding with the average.
                        # print("Not in index c1:",c1,len(c1.split(" ")))
                        a = create_sentence_embedding(c1,retrofitted_embeddings,rcgan,new)
                        # a = rcgan.g_AB.predict(np.array(generate_fastext_embedding(c1)).reshape(1, 300))
                        new.append(a)
                    elif c1 not in retrofitted_embeddings.index:
                        print("Not in index still and less than 1 c1:",c1,len(c1.split(" ")))
                        a = rcgan.g_AB.predict(np.array(generate_fastext_embedding(c1)).reshape(1, 300)).reshape(300)
                        new.append((c1,pd.Series(a)))

                    if len(c2.split(" "))>1:
                        # print("Not in index c2:",c2,len(c2.split(" ")))
                        a= create_sentence_embedding(c2,retrofitted_embeddings,rcgan,new)
                        new.append(a)
                    elif c2 not in retrofitted_embeddings.index:
                        print("Not in index still and less than 1 c2:",c2,len(c2.split(" ")))
                        a = rcgan.g_AB.predict(np.array(generate_fastext_embedding(c2)).reshape(1, 300)).reshape(300)
                        new.append((c2,pd.Series(a)))
                            # else:
                            #     a = np.array(generate_fastext_embedding(c1))
                            #     new.append(a)
                            #     print("Adding",c1)

                    # except Exception as e:
                    #     print(e)
                    #     skipped+=1
                    #     continue

                valid_relations.append([rel, c1, c2, weight])
            except Exception as e:
                print("An error ocurred in",row_num,assertion_row)
                print(e)
                pass
                # print(e)
            if len(valid_relations) % 10000 == 0:
                print(len(valid_relations), skipped)
        print(skipped)
    new_index = [x for x in retrofitted_embeddings.index]
    new_vals = [x for x in retrofitted_embeddings.values]
    for i in range(len(new)):
        new_index.append(new[i][0])
        new_vals.append(new[i][1])
    print("Updating embeddings")
    retrofitted_embeddings = pd.DataFrame(data=new_vals,index=new_index)
    print("Dropping dupes")
    retrofitted_embeddings.drop_duplicates(inplace=True)
    print("SAVING TO FILE")
    retrofitted_embeddings.to_hdf("tmp/updated_embeddings.hd5","mat")
    print("Generating the training data")
    af = pd.DataFrame(data=valid_relations, index=range(len(valid_relations)))
    # af = retrofitted_embeddings
    print("Training data:")
    print(af)
    af.to_hdf("tmp/valid_rels.hd5", "mat")
    return af, retrofitted_embeddings


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
                                                    )
                model_dict[layer_name].summary()
            else:
                model_dict[layer_name] = load_model(save_folder + "/" + layer_name + ".model",
                                                    )

    else:
        layer_name = model_name.replace("/r/", "")
        if not probability_models:
            print("Loading models")
            model_dict[layer_name] = load_model(save_folder + "/" + layer_name + ".model",
                                                )
            print("Loading weights")
            model_dict[layer_name].load_weights(save_folder + "/" + layer_name + ".model")
        else:
            print("Loading models")
            model_dict[layer_name] = load_model(save_folder + "/" + layer_name + "probability.model",
                                                )
            print("Loading weights")
            model_dict[layer_name].load_weights(save_folder + "/" + layer_name + "probability.model")

    # model_dict["common"] = load_model(save_folder + "/" + "common" + ".model",
    #                                         custom_objects={"ConstMultiplierLayer": ConstMultiplierLayer})
    return model_dict

# retroembeddings = "trained_models/retroembeddings/2019-04-0813:03:02.430691/retroembeddings.h5"
# retroembeddings = "trained_models/retroembeddings/2019-10-22 11:57:48.878874/retroembeddings.h5"
# retroembeddings = "trained_models/retroembeddings/2019-05-15 11:47:52.802481/retroembeddings.h5"
# retroembeddings = "retrogan/mini.h5"
# retroembeddings = "/Users/pedro/Downloads/numberbatch-en-19.08.txt"
# retroembeddings = "/Users/pedro/PycharmProjects/OOVconverter/ft_full_ar_vecs.txt"
retroembeddings = "/home/pedro/mltests/ft_full_ar_vecs.txt"
# retroembeddings = "/home/pedro/Downloads/numberbatch-en-19.08.txt"
# retrofitted_embeddings = None

def load_embeddings(path):
    global retrofitted_embeddings
    # if ".h5" in path or ".hd" in path:
    #     retrofitted_embeddings = pd.read_hdf(retroembeddings, "mat")
    # elif ".txt" in path or ".vec" in path:
    vecs = []
    indexes = []
    skip_first = True
    with open(path) as f:
        for idx, line in enumerate(tqdm(f)):
            # if idx>200:break
            if skip_first:
                skip_first = False
                continue

            indexes.append(line.split(" ")[0])
            vecs.append(np.array([float(x) for x in line.split(" ")[1:]]))
    retrofitted_embeddings = pd.DataFrame(index=indexes,data=vecs)
    print("Done")
# print("Loading embeddings")
load_embeddings(retroembeddings)

def test_model_2(model):
    global retrofitted_embeddings
    rcgan = RetroCycleGAN(save_folder="test", batch_size=32, generator_lr=0.0001, discriminator_lr=0.001)
    rcgan.load_weights(preface="final", folder="trained_models/retrogans/ft_full_alldata_feb11")
    assertionspath = "test.txt"
    x_1 = []
    x_2 = []
    y_true = []
    rels = []
    with open(assertionspath) as assertionsfile:
        assertions = csv.reader(assertionsfile, delimiter="\t")
        row_num = 0
        skipped = 0
        for assertion_row in tqdm(assertions):
            row_num += 1
            if row_num % 100000 == 0: print(row_num)
            rel = assertion_row[0]
            if "/r/" not in rel: rel = "/r/" + rel
            y = int(assertion_row[3])

            # print(c1_split)
            c1 = assertion_row[1]
            c2 = assertion_row[2]
            x1 = None
            x2 = None

            if c1 not in retrofitted_embeddings.index or c2 not in retrofitted_embeddings.index:
                if rel not in relations:
                    skipped += 1
                    continue
                try:
                    if len(c1.split(" ")) > 1:
                        # We have a sentence so create the embedding with the average.
                        a= create_sentence_embedding(c1, retrofitted_embeddings,rcgan,[])
                        x1 = a[1]
                    elif len(c1.split(" "))==0:
                        x1 = retrofitted_embeddings.loc[c1]
                    else:
                        x1 = retrofitted_embeddings.loc[c1]
                    if len(c2.split(" ")) > 1:
                        a = create_sentence_embedding(c2, retrofitted_embeddings,rcgan,[])
                        x2 = a[1]
                    elif len(c2.split(" "))==0:
                        x2 = retrofitted_embeddings.loc[c2]
                    else:
                        x2 = retrofitted_embeddings.loc[c2]

                except Exception as e:
                    print(e)
                    skipped += 1
                    continue

            else:
                try:
                    x1 = retrofitted_embeddings.loc[c1]
                except Exception as e:
                    print("Skipping x1")
                try:
                    x2 = retrofitted_embeddings.loc[c2]
                except Exception as e:
                    print("Skupping x2 ")
            if x1 is None or x2 is None:
                # print("Skipping",assertion_row,row_num)
                # skipped+=1
                # x_1.append(None)
                # x_2.append(None)
                # y_true.append(0)
                # rels.append(rel)
                continue
            else:
                x_1.append(x1)
                x_2.append(x2)
                y_true.append(y)
                rels.append(rel)
        print("Skipped:",skipped)

    print("Done")
    y_pred = []
    print(len(y_true),len(x_1),len(x_2),len(rels))
    for i in tqdm(range(len(x_1))):
        try:
            tmp1=np.array(x_1[i])
            t1 = tmp1.reshape(1, tmp1.shape[0])
            t1 = t1[0,:]
            t1 = t1.reshape((1,t1.shape[-1]))
            # print(tmp1)
            # print(x_1.shape)
            tmp2=np.array(x_2[i])
            # print(tmp2)
            t2 = tmp2.reshape(1, tmp2.shape[0])
            t2 = t2[0,:]
            t2 = t2.reshape((1,t2.shape[-1]))

            pred = model[rels[i].replace("/r/", "")].predict(x={'input_1': t1,
                                                                'input_2': t2})
            y_pred.append(pred)
        except Exception as e:
            y_pred.append(-1)
            print(e)
    print(y_pred)
    avg = np.average(y_pred)
    print(avg)
    final_y_pred = [0 if x<avg else 1 for x in y_pred]
    m = tf.keras.metrics.BinaryAccuracy()
    m.update_state(y_true, final_y_pred)
    print('Final result: ', m.result().numpy())  # Final result: 0.75
    return


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    # # save_folder =     "./trained_models/deepreldis/"+str(datetime.datetime.now())
    global w1, w2, w3, retrofitted_embeddings
    data, retrofitted_embeddings = create_data2(use_cache=True)

    w1 = np.array(retrofitted_embeddings.loc["building"]).reshape(1, 300)
    w2 = np.array(retrofitted_embeddings.loc["cat"]).reshape(1, 300)
    w3 = np.array(retrofitted_embeddings.loc["dog"]).reshape(1, 300)
    model_name = "UsedFor"
    # del retrofitted_embeddings
    # gc.collect()
    print("Creating model...")
    model = create_model()
    print("Done\nLoading data")
    # model = load_model_ours("/home/pedro/mltests/trained_models/deepreldis/2020-02-13 04:25:15.169007")
    # model = load_model_ours("trained_models/deepreldis/2020-02-12 07:49:00.552561")
    # # data = load_data("valid_rels.hd5")
    print("Done\nTraining")
    train_on_assertions(model, data,save_folder="trained_models/deepreldis/"+str(datetime.datetime.now())+"/",epoch_amount=100,batch_size=32)
    print("Done\n")
    # model = load_model_ours(save_folder="drd/",model_name="all")
    # model = load_model_ours(save_folder="trained_models/deepreldis/2019-04-25_2_sigmoid",model_name=model_name,probability_models=True)
    # normalizers = normalize_outputs(model,use_cache=False)
    # normalizers = normalize_outputs(model,use_cache=False)
    test_model_2(model)
    # test_model(model, normalizers=None, model_name=model_name)
    # Output needs to be the relationship weights
