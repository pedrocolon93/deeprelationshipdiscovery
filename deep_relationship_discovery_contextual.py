import csv
import datetime
import gc
import pickle
import random
from collections import Iterable
from random import shuffle
from tensorflow_addons.optimizers import AdamW
import wandb
from transformers import XLNetModel, XLNetTokenizer, TFXLNetModel, XLNetConfig
from wandb.keras import WandbCallback

wandb.init(project="deep-relationship-discovery", magic=True)
from wandb import magic

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
# from keras.utils import plot_model
from tqdm import tqdm

from retrogan_trainer_attractrepel_working import *
from tools import generate_fastext_embedding

relations = ["/r/PartOf", "/r/IsA", "/r/HasA", "/r/UsedFor", "/r/CapableOf", "/r/Desires",  # 6
             "/r/AtLocation", "/r/HasSubevent", "/r/HasFirstSubevent", "/r/HasLastSubevent", "/r/HasPrerequisite",  # 5
             "/r/HasProperty", "/r/MotivatedByGoal", "/r/ObstructedBy", "/r/CreatedBy", "/r/Synonym",  # 5
             "/r/Causes", "/r/Antonym", "/r/DistinctFrom", "/r/DerivedFrom", "/r/SymbolOf", "/r/DefinedAs",  # 6
             "/r/SimilarTo", "/r/Entails",  # 2
             "/r/MannerOf", "/r/RelatedTo",  # 2
             "/r/LocatedNear", "/r/HasContext", "/r/FormOf", "/r/EtymologicallyRelatedTo",  # 4
             "/r/EtymologicallyDerivedFrom", "/r/CausesDesire", "/r/MadeOf", "/r/ReceivesAction", "/r/InstanceOf",  # 5
             "/r/NotDesires", "/r/NotUsedFor", "/r/NotCapableOf", "/r/NotHasProperty", "/r/NotIsA", "/r/NotHasA",
             "/r/InheritsFrom", "/r/HasPainIntensity", "/r/DesireOf", "/r/LocationOfAction", "/r/NotMadeOf"]  # 4
# print("Disabling eager exec.")
# disable_eager_execution()
# print("Loading RetroG")
rcgan_folder = "trained_models/retrogans/ft_full_alldata_feb11"
fasttext_folder = "fasttext_model/cc.en.300.bin"
word_dict = {}


def create_model_kxlnet():
    # Input needs to be 2 word vectors
    contextual_size = 768
    wv1 = tf.keras.layers.Input(shape=(300,), name="retro_word_1")
    wv1_ctx = tf.keras.layers.Input(shape=(contextual_size), name="ctx_word_1")
    wv2 = tf.keras.layers.Input(shape=(300,), name="retro_word_2")
    wv2_ctx = tf.keras.layers.Input(shape=(contextual_size), name="ctx_word_2")
    sentence_context = tf.keras.layers.Input(shape=(contextual_size), name="assertion_summary")

    expansion_size = 1024
    intermediate_size = 512

    def create_word_input_abstraction(wv1):
        # Expand and contract the 2 word vectors
        wv1_expansion_1 = tf.keras.layers.Dense(expansion_size)(wv1)
        wv1_expansion_1 = tf.keras.layers.BatchNormalization()(wv1_expansion_1)
        wv1_expansion_1 = tf.keras.layers.Dropout(0.1)(wv1_expansion_1)
        return wv1_expansion_1

    wv1_expansion_3 = create_word_input_abstraction(wv1)
    wv1_ctx_expansion_3 = create_word_input_abstraction(wv1_ctx)
    wv2_expansion_3 = create_word_input_abstraction(wv2)
    wv2_ctx_expansion_3 = create_word_input_abstraction(wv2_ctx)

    wv1_ctx_plus_know = wv1_expansion_3 + wv1_ctx_expansion_3
    wv2_ctx_plus_know = wv2_expansion_3 + wv2_ctx_expansion_3

    # Concatenate both expansions
    merge1 = tf.keras.layers.Concatenate()([wv1_ctx_plus_know, wv2_ctx_plus_know,sentence_context])
    merge_expand = tf.keras.layers.Dense(2048, )(merge1)
    merge_expand = tf.keras.layers.BatchNormalization()(merge_expand)
    merge_expand = tf.keras.layers.Dropout(0.1)(merge_expand)

    merge_expand = tf.keras.layers.Dense(2048, )(merge_expand)
    merge_expand = tf.keras.layers.BatchNormalization()(merge_expand)
    semi_final_layer = tf.keras.layers.Dropout(0.1)(merge_expand)

    common_layers_model = tf.keras.Model([wv1, wv1_ctx, wv2, wv2_ctx, sentence_context],
                                         semi_final_layer,
                                         name="Common layers")

    task_layer_neurons = 512
    model_dict = {}
    for rel in relations:
        model_outs = []
        task_layer = tf.keras.layers.Dense(task_layer_neurons, activation='relu')(semi_final_layer)
        task_layer = tf.keras.layers.BatchNormalization()(task_layer)
        task_layer = tf.keras.layers.Dropout(0.1)(task_layer)

        # task_layer = attention(task_layer)
        task_layer = tf.keras.layers.Dense(task_layer_neurons, activation='relu')(task_layer)
        task_layer = tf.keras.layers.BatchNormalization()(task_layer)
        task_layer = tf.keras.layers.Dropout(0.1)(task_layer)

        layer_name = rel.replace("/r/", "")
        losses = []
        losses.append("binary_crossentropy")
        losses.append("mean_squared_error")

        prob_out = tf.keras.layers.Dense(task_layer_neurons, activation='relu')(task_layer)
        prob_out = tf.keras.layers.BatchNormalization()(prob_out)
        prob_out = tf.keras.layers.Dense(1, name=layer_name + "prob", activation='sigmoid')(prob_out)

        strength_out = tf.keras.layers.Dense(task_layer_neurons, activation='relu')(task_layer)
        strength_out = tf.keras.layers.BatchNormalization()(strength_out)
        strength_out = tf.keras.layers.Dense(1, name=layer_name + "strength", activation='sigmoid')(strength_out)

        drd = tf.keras.Model([wv1, wv1_ctx, wv2, wv2_ctx, sentence_context],
                             [prob_out, strength_out],
                             name=layer_name)

        optimizer = tf.keras.optimizers.Adam(lr=0.0002)
        drd.compile(optimizer=optimizer, loss=losses)
        drd.summary()
        model_dict[layer_name] = drd

    common_layers_model.summary()

    return model_dict


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


def get_ft_embedding(param):
    global rcgan
    if param in word_dict.keys():
        return word_dict[param]
    s = param.split(" ")
    if len(s) > 1:
        a = create_sentence_embedding(param)
        word_dict[param] = a
        return word_dict[param]
    else:
        a = rcgan.g_AB.predict(np.array(generate_fastext_embedding(param)).reshape(1, 300)).reshape(300)
        word_dict[param] = a
        return word_dict[param]


left_ctx_dict = dict()
right_ctx_dict = dict()
assertion_ctx = dict()

for rel in relations:
    left_ctx_dict[rel] = {}
    right_ctx_dict[rel] = {}
    assertion_ctx[rel] = {}


def decode_rel(relationship):
    rel_dict = {
        "/r/PartOf": "part of",
        "/r/IsA": "is a",
        "/r/HasA": "has a",
        "/r/UsedFor": "used for",
        "/r/CapableOf": "capable of",
        "/r/Desires": "desires",
        "/r/AtLocation": "at location",
        "/r/HasSubevent": "has subevent",
        "/r/HasFirstSubevent": "has first subevent",
        "/r/HasLastSubevent": "has last subevent",
        "/r/HasPrerequisite": "has prerequisite",
        "/r/HasProperty": "has property", "/r/MotivatedByGoal": "motivated by goal",
        "/r/ObstructedBy": "obstructed by", "/r/CreatedBy": "created by", "/r/Synonym": "synonym of",  # 5
        "/r/Causes": "causes", "/r/Antonym": "antonym of", "/r/DistinctFrom": "distinct from",
        "/r/DerivedFrom": "derived from", "/r/SymbolOf": "symbol of", "/r/DefinedAs": "defined as",  # 6
        "/r/SimilarTo": "similar to", "/r/Entails": "entails",  # 2
        "/r/MannerOf": "manner of", "/r/RelatedTo": "related to",  # 2
        "/r/LocatedNear": "located near", "/r/HasContext": "has context", "/r/FormOf": "form of",
        "/r/EtymologicallyRelatedTo": "etymologically related to",  # 4
        "/r/EtymologicallyDerivedFrom": "etymologically derived from",
        "/r/CausesDesire": "causes desire", "/r/MadeOf": "made of",
        "/r/ReceivesAction": "receives action", "/r/InstanceOf": "instance of",  # 5
        "/r/NotDesires": "not desires", "/r/NotUsedFor": "not used for", "/r/NotCapableOf": "not capable of",
        "/r/NotHasProperty": "not has property", "/r/NotIsA": "not is a", "/r/NotHasA": "not has a",
        "/r/InheritsFrom": "inherits from", "/r/HasPainIntensity": "has pain intensity",
        "/r/DesireOf": "desire of", "/r/LocationOfAction": "location of action", "/r/NotMadeOf": "not made of"
    }
    translated = rel_dict[relationship]
    return translated


def get_xlnet_embeddings(word1, word2, xlnet_model, xlnet_tokenizer, relationship):
    global left_ctx_dict, right_ctx_dict, assertion_ctx
    sentence = "The " + word1 + ", " + decode_rel(relationship) + ", " + word2 + "."
    enc_sentence = xlnet_tokenizer.encode(sentence, add_special_tokens=True, return_tensors="tf")
    enc_sentence_list = xlnet_tokenizer.encode(sentence, add_special_tokens=True)
    dec_sent = xlnet_tokenizer.decode(enc_sentence_list)
    start_index = -1
    split_index = xlnet_tokenizer.convert_tokens_to_ids([","])[0]
    end_index = xlnet_tokenizer.convert_tokens_to_ids(["."])[0]
    split_index_1 = -5
    split_index_2 = -4
    ass_ctx = ""
    for idx, element in enumerate(enc_sentence[0]):
        # print(element)
        # print(int(element))
        if idx == 0:
            start_index = idx
        if int(element) == end_index:
            end_index = idx
        if int(element) == split_index:
            if split_index_1 == -5:
                split_index_1 = idx
            elif split_index_2 == -4:
                split_index_2 = idx
    start_words = [x for x in enc_sentence.numpy()[0][range(start_index + 1, split_index_1)]]
    end_words = [x for x in enc_sentence.numpy()[0][range(split_index_2 + 1, end_index)]]
    if str(start_words) in left_ctx_dict[relationship].keys() and str(end_words) in right_ctx_dict[relationship].keys() and sentence in assertion_ctx[relationship].keys():
        return left_ctx_dict[relationship][str(start_words)], right_ctx_dict[relationship][str(end_words)],assertion_ctx[relationship][sentence]

    outputs = xlnet_model(enc_sentence)
    hidden_dims = outputs[1]
    hidden_dims_stacked = tf.stack(hidden_dims, axis=0)
    w1 = hidden_dims_stacked[:, :, start_index + 1:split_index_1, :]
    w2 = hidden_dims_stacked[:, :, split_index_2 + 1:end_index, :]
    ass_ctx = hidden_dims_stacked[:,:,len(enc_sentence[0])-1,:]
    w1_red = w1[9:13,:,:,:]
    w2_red = w2[9:13, :, :, :]
    ass_ctx = ass_ctx[9:13, :, :]
    w1 = tf.math.reduce_mean(w1_red, axis=0)
    w1 = tf.math.reduce_mean(w1, axis=1)
    w2 = tf.math.reduce_mean(w2_red, axis=0)
    w2 = tf.math.reduce_mean(w2, axis=1)
    ass_ctx = tf.math.reduce_mean(ass_ctx, axis=0)

    left_ctx_dict[relationship][str(start_words)] = w1.numpy()
    right_ctx_dict[relationship][str(end_words)] = w2.numpy()
    assertion_ctx[relationship][sentence] = ass_ctx.numpy()
    return w1.numpy(), w2.numpy(), ass_ctx.numpy()


def train_on_assertions(model, data, epoch_amount=100, batch_size=32, save_folder="drd", cutoff=0.99, logdir="./logs/",
                        name="default", xlnet_model=None, xlnet_tokenizer=None):
    # retrofitted_embeddings = pd.read_hdf(retroembeddings, "mat", encoding='utf-8')
    global word_dict, left_ctx_dict, right_ctx_dict
    training_data_dict = {}
    training_func_dict = {}
    print("Amount of data:", len(data))
    for i in tqdm(range(len(data))):
        stuff = data.iloc[i]
        rel = stuff[0]
        if rel not in training_data_dict.keys():
            training_data_dict[rel] = []
        training_data_dict[rel].append(i)

    # extra_concepts = []
    # with open("ft_full_alldata/fullfasttext") as ft:
    #     # with open("ft_full_ar_vecs.txt") as ft:
    #     limit = 50000
    #     count = 0
    #     print("Loading extra concepts")
    #     for line in tqdm(ft):
    #         if count == limit: break
    #         c = line.split(" ")[0].replace("en_", "")
    #         get_ft_embedding(c)
    #         extra_concepts.append(c)
    #         count += 1

    def load_batch(output_name):
        # print("Loading batch for", output_name)
        l = len(training_data_dict[output_name])
        iterable = list(range(0, l))
        shuffle(iterable)

        batches = []
        # print("Preloading")
        for ndx in range(0, l, batch_size):
            try:
                ixs = iterable[ndx:min(ndx + batch_size, l)]
                x_1 = []
                x_2 = []
                x_c_1 = []
                x_c_2 = []
                ctx_x = []
                y = []
                y_strength = []
                for ix in ixs:
                    try:
                        stuff = data.iloc[training_data_dict[output][ix]]
                        l1 = np.array(get_ft_embedding(stuff[1])).reshape(1, 300)
                        l2 = np.array(get_ft_embedding(stuff[2])).reshape(1, 300)
                        c1, c2,ctx = get_xlnet_embeddings(stuff[1], stuff[2], xlnet_model, xlnet_tokenizer, output_name)
                        x_1.append(l1)
                        x_c_1.append(c1)
                        x_2.append(l2)
                        x_c_2.append(c2)
                        ctx_x.append(ctx)
                        y.append(1)
                        y_strength.append(float(stuff[3]))
                    except Exception as e:
                        print(e)
                        raise Exception("Fuck")
                        continue
                orig_y = len(y)
                for idx in range(len(y)):
                    x1 = x_1[idx]
                    x2 = x_2[idx]
                    x1c = x_c_1[idx]
                    x2c = x_c_2[idx]

                    # print(idx)
                    ix1 = random.sample(range(0, orig_y), 1)
                    ix2 = random.sample(range(0, orig_y), 1)
                    stufforiginal = data.iloc[training_data_dict[output][idx]]
                    stuffx1 = data.iloc[training_data_dict[output][ix1[0]]]
                    stuffx2 = data.iloc[training_data_dict[output][ix2[0]]]
                    cx11, cx12, ctx1 = get_xlnet_embeddings(stuffx1[1], stufforiginal[2], xlnet_model, xlnet_tokenizer, output_name)
                    cx21, cx22, ctx2 = get_xlnet_embeddings(stufforiginal[1], stuffx2[2], xlnet_model, xlnet_tokenizer, output_name)

                    # print(ix1)
                    # print(ix2)
                    x_1.append(x_1[ix1[0]])
                    x_c_1.append(cx11)
                    x_2.append(x2)
                    x_c_2.append(x2c)
                    ctx_x.append(ctx1)
                    # yval = np.random.binomial(1, 0.2)
                    # yval = random.sample([0,1],1)[0]
                    yval = 0
                    y.append(yval)
                    y_strength.append(yval)
                    yval = 0
                    # yval = random.sample([0,1],1)[0]
                    # yval = np.random.binomial(1, 0.2)

                    x_1.append(x1)
                    x_c_1.append(x1c)
                    x_2.append(x_2[ix2[0]])
                    x_c_2.append(cx22)
                    ctx_x.append(ctx2)
                    y.append(yval)
                    y_strength.append(yval)
                    # idx+=1
                # print(np.array(x_1),np.array(x_2),np.array(y))
                # print("Shuffling in the confounders")

                # Add extra data!!!
                # print("Adding extra data")
                # extra_data_amount = 1
                # l1 = np.array(get_ft_embedding(extra_concepts[random.sample(range(0, len(extra_concepts)), extra_data_amount)[0]])).reshape(1, 300)
                # l2 = np.array(get_ft_embedding(extra_concepts[random.sample(range(0, len(extra_concepts)), extra_data_amount)[0]])).reshape(1, 300)
                # x_1.append(l1)
                # x_2.append(l2)
                # yval = random.sample([0, 1], 1)[0]
                #
                # y.append(yval)
                # y_strength.append(0)
                # print("Adding extra data done")

                c = list(zip(x_1, x_2, x_c_1, x_c_2, ctx_x ,y, y_strength))
                random.shuffle(c)
                x_1, x_2, x_c_1, x_c_2, ctx_x, y, y_strength = zip(*c)
                # print("Done")
                yield np.array(x_1), np.array(x_2), np.array(x_c_1), np.array(x_c_2), np.array(ctx_x), np.array(y), np.array(y_strength)
                # batches.append((np.array(x_1), np.array(x_2), np.array(y),np.array(y_strength)))
            except Exception as e:
                print("Exception occurred!!!", e)
                return False
        # print("Preloading done")
        # for tup in batches:
        #     yield tup

        return True

    callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    wandbcb = WandbCallback(
        monitor='accuracy_test',
        verbose=0,
        mode='auto',
        log_weights=True,
        save_model=True,
        log_evaluation=True,
        log_best_prefix='best_',
        log_batch_frequency=50)
    wandbcb.set_model(model[list(model.keys())[0]])
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
        with tqdm() as pbar:
            with tqdm(total=len(tasks_completed.keys())) as pbar2:
                while True:
                    # break
                    for output in exclude:
                        try:
                            it = training_func_dict[output]
                            iter += 1
                            x_1, x_2, x_c_1, x_c_2, ctx, y, y_strength = it.__next__()
                            pbar.update(1)
                            # print(x_1.shape)
                            x_1 = x_1.reshape(x_1.shape[0], x_1.shape[-1])
                            x_c_1 = x_c_1.reshape(x_c_1.shape[0], x_c_1.shape[-1])

                            # print(x_1.shape)
                            x_2 = x_2.reshape(x_2.shape[0], x_2.shape[-1])
                            x_c_2 = x_c_2.reshape(x_c_2.shape[0], x_c_2.shape[-1])
                            ctx = ctx.reshape(ctx.shape[0], ctx.shape[-1])
                            # try:
                            loss = model[output.replace("/r/", "")].train_on_batch(x={'retro_word_1': x_1,
                                                                                      'retro_word_2': x_2,
                                                                                      'ctx_word_1': x_c_1,
                                                                                      'ctx_word_2': x_c_2,
                                                                                      'assertion_summary':ctx},
                                                                                   y=(y, y_strength))
                            if isinstance(loss, Iterable):
                                callback.on_batch_end(iter, {"loss": loss[0], "loss_bin": loss[1], "loss_mse": loss[2]})
                                try:
                                    wandbcb.on_batch_end(iter,
                                                         {"loss": loss[0], "loss_bin": loss[1], "loss_mse": loss[2]})
                                except Exception as e:
                                    print(e)
                                    pass

                            else:
                                callback.on_batch_end(iter, {"loss_bin": loss})

                            # loss_2 = model[output.replace("/r/", "")].train_on_batch(x={'retro_word_1':x_2,'retro_word_2':x_1},y=y)
                            if isinstance(loss, Iterable):
                                total_loss += loss[0]
                            else:
                                total_loss += loss
                            # if loss > 10:
                            #     print("Loss", output, loss)
                            # if iter%100:
                            #     print(loss)
                            # print("No data for", e)
                        except StopIteration as e:
                            # print("Done with",output)
                            if not tasks_completed[output]: pbar2.update(1)
                            tasks_completed[output] = True
                            # print("Resetting the iterator")
                            training_func_dict[output] = load_batch(output)
                        except KeyError as e:
                            print("Key error", e)

                    if len([x for x in tasks_completed.values() if x]) / len(tasks_completed.values()) > cutoff:
                        print(tasks_completed)
                        break
                    else:
                        pass
                        # print(tasks_completed)
                    iter += 1
        try:
            accuracy = test_model_3(model, xlnet_model, xlnet_tokenizer)
            print(name, save_fol)
            print("Accuracy is", accuracy)
            callback.on_epoch_end(epoch, {"accuracy_test": float(accuracy)})
            wandbcb.on_epoch_end(epoch, {"accuracy_test": float(accuracy)})
            # wandb.log({"accuracy_test": float(accuracy)})
            print("Pushed to callback!!")
        except Exception as e:
            print(e)

        print("Avg loss", total_loss / iter)
        print(str(epoch) + "/" + str(epoch_amount))

        if len(word_dict.keys()) > 50000:
            try:
                print("Dumping the dictionary")
                del word_dict
                word_dict = {}
                gc.collect()
            except Exception as e:
                print("Problem cleaning word dictionary!")
                print(e)
        for rel in relations:
            if len(left_ctx_dict[rel].keys()) > 25000:
                try:
                    print("Dumping the dictionary, left")
                    del left_ctx_dict[rel]
                    left_ctx_dict[rel] = {}
                    gc.collect()
                except Exception as e:
                    print("Problem cleaning word dictionary!")
                    print(e)
            if len(right_ctx_dict[rel].keys()) > 25000:
                try:
                    print("Dumping the dictionary")
                    del right_ctx_dict[rel]
                    right_ctx_dict[rel] = {}
                    gc.collect()
                except Exception as e:
                    print("Problem cleaning word dictionary!")
                    print(e)

        if epoch % 10 == 0:
            try:
                os.mkdir(save_folder)
            except Exception as e:
                print(e)
            for key in model.keys():
                model[key].save(save_folder + "/" + key + ".model", include_optimizer=False)
            wandb.save(save_folder + "/""*.model")
        # test_model(prob_model, model_name=model_name)
    print("Saving...")
    try:
        os.mkdir(save_folder)
    except Exception as e:
        print(e)
    for key in model.keys():
        model[key].save(save_folder + "/" + key + ".model", include_optimizer=False)


def load_data(path):
    data = pd.read_hdf(path, "mat", encoding="utf-8")
    return data


def test_model(model_dict, model_name="all", normalizers=None):
    print("testing")
    global w1, w2, w3
    res1 = model_dict[model_name].predict(x={"retro_word_1": w1,
                                             "retro_word_2": w2})
    res2 = model_dict[model_name].predict(x={"retro_word_1": w1,
                                             "retro_word_2": w3})
    print(res1, res2)

    if normalizers is not None:
        norm_res = normalizers[model_name].transform(res1)
        print(norm_res)
        norm_res = normalizers[model_name].transform(res2)
        print(norm_res)


def load_model_ours(save_folder="./drd", model_name="all", probability_models=False):
    model_dict = {}
    if model_name == 'all':
        for rel in tqdm(relations):
            print("Loading", rel)
            layer_name = rel.replace("/r/", "")
            if probability_models:
                print("Loading", save_folder + "/" + layer_name + "probability.model")
                model_dict[layer_name] = tf.keras.models.load_model(
                    save_folder + "/" + layer_name + "probability.model",
                )
                model_dict[layer_name].summary()
            else:
                model_dict[layer_name] = tf.keras.models.load_model(save_folder + "/" + layer_name + ".model")
                losses = []
                losses.append("binary_crossentropy")
                losses.append("mean_squared_error")
                lr_schedule = tf.optimizers.schedules.ExponentialDecay(2e-4, 9000, 0.9)
                wd_schedule = tf.optimizers.schedules.ExponentialDecay(5e-5, 9000, 0.9)
                optimizer = AdamW(learning_rate=lr_schedule, weight_decay=lambda: None)
                optimizer.weight_decay = lambda: wd_schedule(optimizer.iterations)
                model_dict[layer_name].compile(optimizer=optimizer, loss=losses)
                print("Adding adamw")
    else:
        layer_name = model_name.replace("/r/", "")
        if not probability_models:
            print("Loading models")
            model_dict[layer_name] = tf.keras.models.load_model(save_folder + "/" + layer_name + ".model")
            loss = "binary_crossentropy"
            losses = []
            losses.append(loss)
            losses.append("mean_squared_error")
            lr_schedule = tf.optimizers.schedules.ExponentialDecay(2e-4, 9000, 0.9)
            wd_schedule = tf.optimizers.schedules.ExponentialDecay(5e-5, 9000, 0.9)
            optimizer = AdamW(learning_rate=lr_schedule, weight_decay=lambda: None)
            optimizer.weight_decay = lambda: wd_schedule(optimizer.iterations)
            model_dict[layer_name].compile(optimizer=optimizer, loss=losses)
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
    retrofitted_embeddings = pd.DataFrame(index=indexes, data=vecs)
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
            # if row_num > 100: break
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


def test_model_3(model, xlnet_model, xlnet_tokenizer):
    global retrofitted_embeddings
    # assertionspath = "test.txt"
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
            y_true.append(y)
            rels.append(rel)

    print("Done")
    y_pred = []
    print(len(y_true), len(x_1), len(x_2), len(rels))
    for i in tqdm(range(len(y_true))):
        try:
            t1 = np.array(get_ft_embedding(x_1[i])).reshape(1, 300)
            t2 = np.array(get_ft_embedding(x_2[i])).reshape(1, 300)
            ct1, ct2 = get_xlnet_embeddings(x_1[i], x_2[i], xlnet_model, xlnet_tokenizer, rels[i])
            pred = -1
            pred = model[rels[i].replace("/r/", "")].predict(x={"retro_word_1": t1,
                                                                "retro_word_2": t2,
                                                                "ctx_word_1": ct1,
                                                                "ctx_word_2": ct2})
            if isinstance(pred[0][0], Iterable):
                y_pred.append(pred[0][0][0])
            else:
                y_pred.append(pred[0][0])
        except Exception as e:
            y_pred.append(-1)
            print(e)
    # print(y_pred)
    avg = np.average(y_pred)
    print(avg)
    final_y_pred = [0 if x < avg else 1 for x in y_pred]
    results = []
    cutoffs = []
    for i in range(1, 100):
        cutoff = i / 100.0
        cutoffs.append(cutoff)
        m = tf.keras.metrics.BinaryAccuracy(threshold=cutoff)
        m.update_state(y_true, y_pred)
        results.append(m.result().numpy())
    print(results)
    maxacc = np.argmax(results)
    print("max is at", cutoffs[maxacc])
    m = tf.keras.metrics.BinaryAccuracy(threshold=cutoffs[maxacc])
    m.update_state(y_true, y_pred)
    print('Final result: ', m.result().numpy())  # Final result: 0.75
    import matplotlib
    import matplotlib.pyplot as plt

    # Data for plotting
    t = cutoffs
    s = results

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='cutoff', ylabel='accuracy',
           title='About as simple as it gets, folks')
    ax.grid()
    fig.savefig("test.png")
    plt.show()
    return m.result().numpy()


def load_things():
    rcgan = RetroCycleGAN(save_folder="test", batch_size=32, generator_lr=0.0001, discriminator_lr=0.001)
    rcgan.load_weights(preface="final", folder=rcgan_folder)
    print("Loading ft")
    generate_fastext_embedding("cat", ft_dir=fasttext_folder)
    print("Ready")
    return rcgan


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
    global rcgan

    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    config = XLNetConfig.from_pretrained("xlnet-base-cased")
    config.output_hidden_states = True
    model = TFXLNetModel.from_pretrained("xlnet-base-cased", config=config)
    rcgan = load_things()

    # # save_folder =     "./trained_models/deepreldis/"+str(datetime.datetime.now())
    global w1, w2, w3, retrofitted_embeddings
    data = create_data3(use_cache=False)

    print("Creating model...")
    # name = input("Model name")
    name = "thewalrus"
    save_fol = "trained_models/deepreldis/" + name + "_" + str(datetime.datetime.now()) + "/"
    c_drd = create_model_kxlnet()

    print("Done\nLoading data")
    # model = load_model_ours(save_fol)
    # model = load_model_ours("trained_models/deepreldis/2020-02-22 14:20:21.182244")
    # model = load_model_ours("trained_models/deepreldis/att_sota_2_2020-04-14 11:38:05.216330")
    # # data = load_data("valid_rels.hd5")
    print("Done\nTraining")
    # print("Saving to,",save_fol)
    print("Removing!!!")
    print("*" * 100)
    # shutil.rmtree(save_fol, ignore_errors=True)
    print("Done!")
    print("*" * 100)

    train_on_assertions(c_drd, data, save_folder=save_fol,
                        epoch_amount=100, batch_size=128, cutoff=0.99, logdir=save_fol, name=name,
                        xlnet_model=model, xlnet_tokenizer=tokenizer)
    print("Done\n")
    # model = load_model_ours(save_folder="drd/",model_name="all")
    # model = load_model_ours(save_folder="trained_models/deepreldis/2019-04-25_2_sigmoid",model_name=model_name,probability_models=True)
    # normalizers = normalize_outputs(model,use_cache=False)
    # normalizers = normalize_outputs(model,use_cache=False)
    test_model_3(model, model, tokenizer)
    # Output needs to be the relationship weights
