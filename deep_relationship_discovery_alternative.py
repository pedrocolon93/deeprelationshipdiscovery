import csv
import datetime
import gc
import pickle
import random
from random import shuffle

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
# from keras.utils import plot_model
from tqdm import tqdm

from retrogan_trainer_attractrepel_working import *
from tools import generate_fastext_embedding

print("Removing!!!")
print("*" * 100)
shutil.rmtree("logs/", ignore_errors=True)
print("Done!")
print("*" * 100)

relations = ["/r/PartOf", "/r/IsA", "/r/HasA", "/r/UsedFor", "/r/CapableOf", "/r/Desires", #6
             "/r/AtLocation", "/r/HasSubevent", "/r/HasFirstSubevent", "/r/HasLastSubevent", "/r/HasPrerequisite", #5
             "/r/HasProperty", "/r/MotivatedByGoal", "/r/ObstructedBy", "/r/CreatedBy", "/r/Synonym", #5
             "/r/Causes", "/r/Antonym", "/r/DistinctFrom", "/r/DerivedFrom", "/r/SymbolOf", "/r/DefinedAs", #6
             "/r/SimilarTo", "/r/Entails", #2
             "/r/MannerOf", "/r/RelatedTo",#2
             "/r/LocatedNear", "/r/HasContext", "/r/FormOf",  "/r/EtymologicallyRelatedTo", #4
             "/r/EtymologicallyDerivedFrom", "/r/CausesDesire", "/r/MadeOf", "/r/ReceivesAction", "/r/InstanceOf", #5
             "/r/NotDesires", "/r/NotUsedFor", "/r/NotCapableOf", "/r/NotHasProperty","/r/NotIsA","/r/NotHasA",
             "/r/InheritsFrom","/r/HasPainIntensity","/r/DesireOf","/r/LocationOfAction","/r/NotMadeOf"] # 4
# print("Disabling eager exec.")
# disable_eager_execution()
# print("Loading RetroG")
rcgan_folder = "final_final_retrogan/0/"
fasttext_folder = "fasttext_model/cc.en.300.bin"
word_dict = {}

def create_sentence_embedding(c1):
    s = c1.split(" ")
    concept_vecs = []
    for word in s:
        try:
            concept_vecs.append(retrofitted_embeddings.loc[word])
        except:
            # print("Creating emb for", word)
            if word in word_dict.keys():
                concept_vecs.append(word_dict[word])
            else:
                concept_vecs.append(pd.Series(rcgan.g_AB.predict(np.array(generate_fastext_embedding(word))
                                                             .reshape(1, 300)
                                                             ).reshape(300)))
    concept_vecs = np.array(concept_vecs)
    avg = np.mean(concept_vecs, axis=0)
    return pd.Series(avg)

def get_embedding(param):
    global rcgan
    if param in word_dict.keys():
        return word_dict[param]
    s = param.split(" ")
    if len(s)>1:
        a = create_sentence_embedding(param)
        word_dict[param] = a
        return word_dict[param]
    else:
        a= rcgan.g_AB.predict(np.array(generate_fastext_embedding(param)).reshape(1, 300)).reshape(300)
        word_dict[param] = a
        return word_dict[param]

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
    # merge1 = Dropout(0.1)(merge1)
    merge_expand = tf.keras.layers.Dense(intermediate_size, activation='relu')(merge1)
    merge_expand = tf.keras.layers.BatchNormalization()(merge_expand)
    # Add atention layer
    # merge_attention = attention(merge_expand)
    attention_expand = tf.keras.layers.Dense(intermediate_size, activation='relu')(merge_expand)
    attention_expand = tf.keras.layers.BatchNormalization()(attention_expand)
    attention_expand = tf.keras.layers.Dropout(0.1)(attention_expand)

    semi_final_layer = tf.keras.layers.Dense(intermediate_size, activation='relu')(attention_expand)
    semi_final_layer = tf.keras.layers.BatchNormalization()(semi_final_layer)
    model_dict = {}


    # drdp.compile(optimizer=optimizer, loss=[loss])
    # drdp.summary()
    # drd.summary()
    # tf.keras.utils.plot_model(drd, show_shapes=True)
    for rel in relations:
        model_outs = []
        losses = []

        # loss = "mean_squared_error"
        loss = "binary_crossentropy"
        losses.append(loss)

        layer_name = rel.replace("/r/", "")
        out = tf.keras.layers.Dense(1, name=layer_name, activation='sigmoid')(semi_final_layer)
        model_outs.append(out)
        # drdp = Model([wv1, wv2], probability, name=layer_name + "probability")
        drd = tf.keras.Model([wv1, wv2], model_outs, name=layer_name)
        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        drd.compile(optimizer=optimizer, loss=losses)
        model_dict[layer_name] = drd

        # prob_model_dict[layer_name] = drdp
    # plot_model(Model([wv1,wv2],model_outs,name="Deep_Relationship_Discovery"),show_shapes=True,to_file="DRD.png")
    # model_dict["common"]=common_layers_model
    # common_layers_model.summary()

    return model_dict#, prob_model_dict

def train_on_assertions(model, data, epoch_amount=100, batch_size=32, save_folder="drd",cutoff=0.99):
    # retrofitted_embeddings = pd.read_hdf(retroembeddings, "mat", encoding='utf-8')
    global word_dict
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
        # print("Loading batch for", output_name)
        l = len(training_data_dict[output_name])
        iterable = list(range(0, l))
        shuffle(iterable)
        extra_concepts = []
        with open("ft_full_alldata/fullfasttext") as ft:
        # with open("ft_full_ar_vecs.txt") as ft:
            limit = 50000
            count = 0
            for line in ft:
                if count==limit:break
                extra_concepts.append(line.split(" ")[0].replace("en_",""))
                count+=1
        batches = []
        # print("Preloading")
        for ndx in range(0, l, batch_size):
            try:
                ixs = iterable[ndx:min(ndx + batch_size, l)]
                x_1 = []
                x_2 = []
                y = []
                y_strength = []
                for ix in ixs:
                    try:
                        stuff = data.iloc[training_data_dict[output][ix]]
                        l1 = np.array(get_embedding(stuff[1])).reshape(1,300)
                        l2 = np.array(get_embedding(stuff[2])).reshape(1,300)
                        x_1.append(l1)
                        x_2.append(l2)
                        # out_vecs = np.zeros(len(relations))
                        # out_vecs[relations.index(stuff[0])]=1
                        y.append(1)
                        y_strength.append(float(stuff[3]))
                    except Exception as e:
                        print(e)
                        raise Exception("Fuck")
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
                    y_strength.append(0)

                    x_1.append(x1)
                    x_2.append(x_2[ix2[0]])
                    y.append(0)
                    y_strength.append(0)
                    idx+=1
                # print(np.array(x_1),np.array(x_2),np.array(y))
                # print("Shuffling in the confounders")

                #Add extra data!!!
                # print("Adding extra data")
                l1 = np.array(get_embedding(extra_concepts[random.sample(range(0, len(extra_concepts)), 1)[0]])).reshape(1, 300)
                l2 = np.array(get_embedding(extra_concepts[random.sample(range(0, len(extra_concepts)), 1)[0]])).reshape(1, 300)
                x_1.append(l1)
                x_2.append(l2)
                y.append(0)
                y_strength.append(0)
                # print("Adding extra data done")

                c = list(zip(x_1, x_2,y,y_strength))
                random.shuffle(c)
                x_1,x_2,y,y_strength = zip(*c)
                # print("Done")
                yield np.array(x_1), np.array(x_2), np.array(y),np.array(y_strength)
                # batches.append((np.array(x_1), np.array(x_2), np.array(y),np.array(y_strength)))
            except Exception as e:
                print("Exception occurred!!!",e)
                return False
        # print("Preloading done")
        # for tup in batches:
        #     yield tup


        return True
    callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/")
    # callback.set_model(drd)
    # callback_dict[layer_name] = callback
    for epoch in tqdm(range(epoch_amount)):
        print("Trying to collect to free up memory!!!")
        gc.collect()
        print("Done")
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
        with tqdm() as pbar:
            with tqdm(total=len(tasks_completed.keys())) as pbar2:
                while True:
                    # break
                    for output in exclude:
                        try:
                            it = training_func_dict[output]
                            iter += 1
                            x_1, x_2, y, y_strength = it.__next__()
                            pbar.update(1)
                            # print(x_1.shape)
                            x_1 = x_1.reshape(x_1.shape[0],x_1.shape[-1])
                            # print(x_1.shape)
                            x_2 = x_2.reshape(x_2.shape[0], x_2.shape[-1])
                            try:
                                loss = model[output.replace("/r/", "")].train_on_batch(x={'retro_word_1': x_1, 'retro_word_2': x_2},
                                                                                       y=(y))
                            except Exception as e:
                                loss = model[output.replace("/r/", "")].train_on_batch(x={'input_1': x_1, 'input_2': x_2},
                                                                                       y=(y))

                            callback.on_epoch_end(iter, {"loss_bin":loss})

                            # loss_2 = model[output.replace("/r/", "")].train_on_batch(x={'retro_word_1':x_2,'retro_word_2':x_1},y=y)
                            total_loss += loss
                            # if loss > 10:
                            #     print("Loss", output, loss)
                            # if iter%100:
                            #     print(loss)
                        except KeyError as e:
                            pass
                            # print("No data for", e)
                        except StopIteration as e:
                            # print("Done with",output)
                            if not tasks_completed[output]: pbar2.update(1)
                            tasks_completed[output] = True
                            # print("Resetting the iterator")
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
                        pass
                        # print(tasks_completed)
                    iter += 1
        try:
            accuracy = test_model_3(model)
            print("Accuracy is",accuracy)
            callback.on_epoch_end(epoch, {"accuracy_test": float(accuracy)})
            print("Pushed to callback!!")
        except Exception as e:
            print(e)

        print("Avg loss", total_loss / iter)
        print(str(epoch) + "/" + str(epoch_amount))

        if len(word_dict.keys())>20000:
            try:
                print("Dumping the dictionary")
                del word_dict
                word_dict = {}
                gc.collect()
            except Exception as e:
                print("Problem cleaning word dictionary!")
                print(e)
        # for key in prob_model.keys():
        #     prob_model[key].save(save_folder + "/" + key + "probability.model")
            # exit()
        # print("Testing")
        # model_name = "PartOf"
        # test_model(model, model_name=model_name)
        print("Saving the model")
        try:
            os.mkdir(save_folder+"/checkpoint")
        except Exception as e:
            print(e)
        for key in model.keys():
            model[key].save(save_folder+"/checkpoint" + "/" + key + ".model")
        # test_model(prob_model, model_name=model_name)
    print("Saving...")
    try:
        os.mkdir(save_folder)
    except Exception as e:
        print(e)
    for key in model.keys():
        model[key].save(save_folder + "/" + key + ".model")


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
                model_dict[layer_name] = tf.keras.models.load_model(save_folder + "/" + layer_name + "probability.model",
                                                    )
                model_dict[layer_name].summary()
            else:
                model_dict[layer_name] = tf.keras.models.load_model(save_folder + "/" + layer_name + ".model",
                                                    )

    else:
        layer_name = model_name.replace("/r/", "")
        if not probability_models:
            print("Loading models")
            model_dict[layer_name] = tf.keras.models.load_model(save_folder + "/" + layer_name + ".model",
                                                )
            # print("Loading weights")
            # model_dict[layer_name].load_weights(save_folder + "/" + layer_name + ".model")
        else:
            print("Loading models")
            model_dict[layer_name] = tf.keras.models.load_model(save_folder + "/" + layer_name + "probability.model",
                                                )
            # print("Loading weights")
            # model_dict[layer_name].load_weights(save_folder + "/" + layer_name + "probability.model")

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
# load_embeddings(retroembeddings)



def create_data3(use_cache):
    global retrofitted_embeddings

    if os.path.exists("tmp/valid_rels.hd5") and use_cache:
        print("Using cache")
        a = pd.read_hdf("tmp/valid_rels.hd5", "mat")
        return a
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
                if "/r/" not in rel: rel = "/r/" + rel
                weight = float(assertion_row[3])
                # print(c1_split)
                c1 = assertion_row[1]
                c2 = assertion_row[2]

                if rel not in relations:
                    print("Skipping relation", rel)
                    skipped += 1
                    continue
                valid_relations.append([rel, c1, c2, weight])
            except Exception as e:
                print("An error ocurred in", row_num, assertion_row)
                print(e)
                pass
                # print(e)
            if len(valid_relations) % 10000 == 0:
                print(len(valid_relations), skipped)
        print(skipped)

    print("Generating the training data")
    af = pd.DataFrame(data=valid_relations, index=range(len(valid_relations)))
    # af = retrofitted_embeddings
    print("Training data:")
    print(af)
    af.to_hdf("tmp/valid_rels.hd5", "mat")
    return af


def test_model_3(model):
    global retrofitted_embeddings
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
            x_1.append(c1)
            x_2.append(c2)
            out_vecs = 1

            y_true.append(out_vecs)
            rels.append(rel)

    print("Done")
    y_pred = []
    print(len(y_true), len(x_1), len(x_2), len(rels))
    for i in tqdm(range(len(y_true))):
        try:
            t1 = np.array(get_embedding(x_1[i])).reshape(1, 300)
            t2 = np.array(get_embedding(x_2[i])).reshape(1, 300)
            pred = -1
            try:
                pred = model[rels[i].replace("/r/", "")].predict(x={"retro_word_1": t1,
                                                         "retro_word_2": t2})
            except:
                pred = model[rels[i].replace("/r/", "")].predict(x={'input_1': t1,
                                                                    'input_2': t2})
            y_pred.append(pred[0])
        except Exception as e:
            y_pred.append(-1)
            print(e)
    # print(y_pred)
    # avg = np.average(y_pred)
    # print(avg)
    # final_y_pred = [0 if x < avg else 1 for x in y_pred]
    m = tf.keras.metrics.BinaryAccuracy()
    m.update_state(y_true, y_pred)
    print('Final result: ', m.result().numpy())  # Final result: 0.75
    return m.result().numpy()


def load_things():
    rcgan = RetroCycleGAN(save_folder="test", batch_size=32, generator_lr=0.0001, discriminator_lr=0.001)
    rcgan.load_weights(preface="final", folder=rcgan_folder)
    print("Loading ft")
    generate_fastext_embedding("cat", ft_dir=fasttext_folder)
    print("Ready")
    return rcgan

if __name__ == '__main__':
    global rcgan
    rcgan = load_things()
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
    data = create_data3(use_cache=False)

    # w1 = np.array(retrofitted_embeddings.loc["building"]).reshape(1, 300)
    # w2 = np.array(retrofitted_embeddings.loc["cat"]).reshape(1, 300)
    # w3 = np.array(retrofitted_embeddings.loc["dog"]).reshape(1, 300)
    # model_name = "UsedFor"
    # del retrofitted_embeddings
    # gc.collect()
    print("Creating model...")
    model = create_model()
    print("Done\nLoading data")
    # model = load_model_ours("trained_models/deepreldis/2020-02-22 14:20:21.182244")
    # model = load_model_ours("trained_models/deepreldis/2020-04-06 15:05:51.103060/checkpoint")
    # # data = load_data("valid_rels.hd5")
    print("Done\nTraining")
    train_on_assertions(model, data,save_folder="trained_models/deepreldis/"+str(datetime.datetime.now())+"/",epoch_amount=100,batch_size=128)
    print("Done\n")
    # model = load_model_ours(save_folder="drd/",model_name="all")
    # model = load_model_ours(save_folder="trained_models/deepreldis/2019-04-25_2_sigmoid",model_name=model_name,probability_models=True)
    # normalizers = normalize_outputs(model,use_cache=False)
    # normalizers = normalize_outputs(model,use_cache=False)
    test_model_3(model)
    # Output needs to be the relationship weights
