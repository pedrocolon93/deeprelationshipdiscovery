from __future__ import print_function, division

import datetime
import math
import os
import random
import shutil
from random import shuffle

import numpy as np
import pandas as pd

os.environ["TF_KERAS"] = "1"
# from keras_adabound import AdaBound
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.python.keras.layers import Input, Conv1D, Dense, BatchNormalization, \
    Dropout, LayerNormalization
from tensorflow.python.keras import Model
from tensorflow.python.framework.ops import disable_eager_execution
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# tf.debugging.set_log_device_placement(True)

import tools


# from wandb import magic


# seed(1)
#
# set_random_seed(2)
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.compat.v1.summary.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

dimensionality = 300
class RetroCycleGAN():
    def __init__(self, save_index="0", save_folder="./", generator_size=32,
                 discriminator_size=64, word_vector_dimensions=300,
                 discriminator_lr=0.0001, generator_lr=0.0001,
                 lambda_cycle=1, lambda_id_weight=0.01,
                 clip_value=0.5, optimizer="sgd", batch_size=32,dimensionality=300):

        self.save_folder = save_folder

        # Input shape
        self.word_vector_dimensions = word_vector_dimensions
        self.img_shape = (self.word_vector_dimensions,)  # , self.channels)
        self.save_index = save_index

        # Number of filters in the first layer of G and D
        self.gf = generator_size
        self.df = discriminator_size

        # Loss weights
        self.lambda_cycle = lambda_cycle  # Cycle-consistency loss
        self.lambda_id = lambda_id_weight * self.lambda_cycle  # Identity loss

        d_lr = discriminator_lr
        self.d_lr = d_lr
        g_lr = generator_lr
        self.g_lr = g_lr
        # cv = clip_value
        # cn = cn
        self.d_A = self.build_discriminator(name="word_vector_discriminator")
        self.d_B = self.build_discriminator(name="retrofitted_word_vector_discriminator")
        # Best combo sofar SGD, gaussian, dropout,5,0.5 mml(0,5,.5),3x1024gen, 2x1024, no normalization

        # return Adam(lr,amsgrad=True,decay=1e-8)

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AB = self.build_generator(name="to_retro_generator")

        # self.d_A.summary()
        # self.g_AB.summary()
        # plot_model(self.g_AB, show_shapes=True)
        self.g_BA = self.build_generator(name="from_retro_generator")

        # self.d_B.summary()
        # self.g_BA.summary()
        # Input images from both domains
        unfit_wv = Input(shape=self.img_shape, name="plain_word_vector")
        fit_wv = Input(shape=self.img_shape, name="retrofitted_word_vector")

        # Translate images to the other domain
        fake_B = self.g_AB(unfit_wv)
        fake_A = self.g_BA(fit_wv)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        print("Building recon model")
        # self.reconstr = Model(inputs=[unfit_wv,fit_wv],outputs=[reconstr_A,reconstr_B])
        print("Done")
        # Identity mapping of images
        unfit_wv_id = self.g_BA(unfit_wv)
        fit_wv_id = self.g_AB(fit_wv)

        # For the combined model we will only train the generators
        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)
        # Combined model trains generators to fool discriminators
        self.d_A.trainable = False
        self.d_B.trainable = False

        self.combined = Model(inputs=[unfit_wv, fit_wv],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       reconstr_A, reconstr_B,
                                       unfit_wv_id, fit_wv_id],
                              name="combinedmodel")

        log_path = './logs'
        callback = keras.callbacks.TensorBoard(log_dir=log_path)
        callback.set_model(self.combined)
        self.combined_callback = callback

        # plot_model(self.combined, to_file="RetroGAN.png", show_shapes=True)

    def compile_all(self, optimizer="sgd"):
        def max_margin_loss(y_true, y_pred):
            cost = 0
            sim_neg = 25
            sim_margin = 1
            for i in range(0, sim_neg):
                new_true = tf.random.shuffle(y_true)
                normalize_a = tf.nn.l2_normalize(y_true)
                normalize_b = tf.nn.l2_normalize(y_pred)
                normalize_c = tf.nn.l2_normalize(new_true)
                minimize = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
                maximize = tf.reduce_sum(tf.multiply(normalize_a, normalize_c))
                mg = sim_margin - minimize + maximize
                cost += tf.keras.backend.clip(mg, 0, 10000)
            return cost / (sim_neg * 1.0)

        def create_opt(lr=0.1):
            if optimizer == "sgd":
                return tf.optimizers.SGD(lr=0.00001, momentum=0.9, decay=1 / (1000 * 10000))
            elif optimizer == "adam":
                #lr_schedule = tf.optimizers.schedules.ExponentialDecay(lr, 25000, 0.98)
                # #lr_schedule =tf.optimizers.schedules.PolynomialDecay(lr, 7500, end_learning_rate=lr/100, power=1.0)
                #wd_schedule = tf.optimizers.schedules.ExponentialDecay(lr*1e-10, 25000, 0.98)
                # #wd_schedule =tf.optimizers.schedules.PolynomialDecay(lr/10, 7500, end_learning_rate=lr/1000, power=1.0)
                #
                #opt = AdamW(learning_rate=lr_schedule, weight_decay=lambda: None)
                #opt.weight_decay = lambda: wd_schedule(opt.iterations)
                #return opt
                opt = tf.optimizers.Adam(lr=lr, epsilon=1e-10)
                opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
                return opt
                # return opt
            elif optimizer == "adabound":
                return AdaBound(lr=1e-3, final_lr=0.1, gamma=0.0000001)
            else:
                raise KeyError("coULD NOT FIND THE OPTIMIZER")

        self.d_A.compile(loss='binary_crossentropy',
                         optimizer=create_opt(self.d_lr),
                         metrics=['accuracy'])
        self.d_B.compile(loss='binary_crossentropy',
                         optimizer=create_opt(self.d_lr),
                         metrics=['accuracy'])
        self.g_AB.compile(loss=max_margin_loss,
                          optimizer=create_opt(self.g_lr),
                          )
        self.g_BA.compile(loss=max_margin_loss,
                          optimizer=create_opt(self.g_lr),
                          )
        # self.reconstr.compile(loss=max_margin_loss,
        #                   optimizer=create_opt(self.g_lr),
        #                     loss_weights=[0.1,0.1]
        #                   )

        self.combined.compile(loss=['binary_crossentropy', 'binary_crossentropy',
                                    'mae', 'mae',
                                    max_margin_loss,max_margin_loss,
                                    'mae', 'mae'],
                              loss_weights=[1, 1,
                                            self.lambda_cycle*1, self.lambda_cycle*1,
                                            self.lambda_cycle*2, self.lambda_cycle*2,
                                            self.lambda_id, self.lambda_id],
                              # TODO ADD A CUSTOM LOSS THAT SIMPLY ADDS A
                              # GENERALIZATION CONSTRAINT ON THE MAE
                              optimizer=create_opt(self.g_lr))
        # self.combined.summary()

    def build_generator(self, name):
        """U-Net Generator"""

        def dense(layer_input, filters, f_size=6, normalization=True, dropout=True):
            """Layers used during downsampling"""
            # d = BatchNormalization()(layer_input)
            d = Dense(filters, activation="relu")(layer_input)
            # d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = BatchNormalization()(d)
            if dropout:
                d = Dropout(0.2)(d)
            return d

        def conv1d(layer_input, filters, f_size=6, strides=1, normalization=True):
            d = Conv1D(filters, f_size, strides=strides, activation="relu")(layer_input)

            return d

        # def deconv1d(layer_input, filters, f_size=6, strides=1, normalization=True):
        #     d = UpSampling1D(filters, f_size, strides=strides, activation="relu")(layer_input)
        #     return d

        # Image input
        inpt = Input(shape=self.img_shape)
        # Continue into fc layers
        encoder = dense(inpt, 2048, normalization=False)
        #intermediate = dense(encoder, 512, normalization=False)
        decoder = dense(encoder, 2048, normalization=False)#+encoder
        output = Dense(dimensionality)(decoder)
        return Model(inpt, output, name=name)

    def build_discriminator(self, name):

        def d_layer(layer_input, filters, f_size=7, normalization=True, dropout=True):
            """Discriminator layer"""
            d = Dense(filters, activation="relu")(layer_input)
            if normalization:
                d = BatchNormalization()(d)
            if dropout:
                d = Dropout(0.3)(d)
            # d = GaussianNoise(0.1)(d)
            return d

        inpt = Input(shape=self.img_shape)
        # d1 = tf.keras.layers.GaussianNoise(1)(inpt)
        d1 = d_layer(inpt, 2048, normalization=False, dropout=True)
        d1 = d_layer(d1, 2048, normalization=True, dropout=True)
        #d1 = d_layer(d1, 512, normalization=False, dropout=True)
        # d1 = d_layer(d1, 2048, normalization=False, dropout=True)
        # validity = Dense(1, activation="sigmoid")(d1)
        validity = Dense(1, activation="sigmoid",dtype='float32')(d1)
        return Model(inpt, validity, name=name)

    def load_weights(self, preface="", folder=None):
        if folder is None:
            folder = self.save_folder
        try:
            self.g_AB.reset_states()
            self.g_BA.reset_states()
            self.combined.reset_states()
            self.d_B.reset_states()
            self.d_A.reset_states()
            self.d_A.load_weights(os.path.join(folder, preface + "fromretrodis.h5"))
            self.d_B.load_weights(os.path.join(folder, preface + "toretrodis.h5"))
            self.g_AB.load_weights(os.path.join(folder, preface + "toretrogen.h5"))
            self.g_BA.load_weights(os.path.join(folder, preface + "fromretrogen.h5"))
            self.combined.load_weights(os.path.join(folder, preface + "combined_model.h5"))

        except Exception as e:
            print(e)

    def train(self, epochs, dataset, batch_size=1, pretraining_epochs=150, rc=None):
        start_time = datetime.datetime.now()
        res = []
        X_train = Y_train = None
        X_train, Y_train = tools.load_all_words_dataset_3(dataset,
                                                          save_folder="adversarial_paper_data/",
                                                          threshold=0.90,
                                                          cache=False,
                                                          remove_constraint=rc)
        print("SHapes",X_train.shape,Y_train.shape)
        # import wandb
        # wandb.init(project="retrogan")
        #
        # from wandb.keras import WandbCallback
        # wandbcb = WandbCallback(
        #     log_best_prefix="best_",
        #     monitor='simverb',
        #     verbose=0,
        #     mode='auto',
        #     save_weights_only=True,
        #     log_weights=True,
        #     save_model=True,
        #     log_gradients=False,
        #     log_evaluation=True,
        #     log_batch_frequency=50)
        # wandbcb.set_model(self.g_AB)

        print(X_train)
        print(Y_train)
        print("*"*100)
        # print(np.isnan(X_train).any())
        # print(np.isnan(Y_train).any())
        print("Done")

        def load_batch(batch_size=32, always_random=False):
            def _int_load():
                iterable = list(Y_train.index)
                shuffle(iterable)
                batches = []
                print("Prefetching batches")
                for ndx in tqdm(range(0, len(iterable), batch_size)):
                    try:
                        ixs = iterable[ndx:min(ndx + batch_size, len(iterable))]
                        if always_random:
                            ixs = list(np.array(iterable)[random.sample(range(0, len(iterable)), batch_size)])
                        imgs_A = X_train.loc[ixs]
                        imgs_B = Y_train.loc[ixs]
                        if np.isnan(imgs_A).any().any() or np.isnan(imgs_B).any().any():#np.isnan(imgs_B).any():
                            # print(ixs)
                            continue

                        batches.append((imgs_A, imgs_B))
                    except Exception as e:
                        print("Skipping batch")
                        # print(e)
                return batches
            # while True:
            #     try:
            batches = _int_load()
                    # break
                # except:
                #     batches = None
                
            print("Begginging iteration")
            for i in tqdm(range(0, len(batches)), ncols=30):
                imgs_A, imgs_B = batches[i]
                yield imgs_A.values, imgs_B.values

        def load_random_batch(batch_size=32, batch_amount=1000000):
            iterable = list(Y_train.index)
            # shuffle(iterable)
            ixs = list(np.array(iterable)[random.sample(range(0, len(iterable)), batch_size)])
            imgs_A = X_train.loc[ixs]
            imgs_B = Y_train.loc[ixs]
            def test_nan(a,b):
                return np.isnan(a).any().any() or np.isnan(b).any().any()
            while True:
                if(test_nan(imgs_A,imgs_B)):
                    ixs = list(np.array(iterable)[random.sample(range(0, len(iterable)), batch_size)])
                    imgs_A = X_train.loc[ixs]
                    imgs_B = Y_train.loc[ixs]
                else:
                    break
            return imgs_A, imgs_B

        def exp_decay(epoch):
            initial_lrate = 0.1
            k = 0.1
            lrate = initial_lrate * math.exp(-k * epoch)
            return lrate

        # noise = np.random.normal(size=(1, dimensionality), scale=0.001)
        # noise = np.tile(noise,(batch_size,1))
        dis_train_amount = 3.0

        self.compile_all("adam")
        # ds = tf.data.Dataset.from_generator(load_batch,(tf.float32,tf.float32),args=(batch_size,))
        # ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        def train_(training_epochs, always_random=False):
            for epoch in range(training_epochs):
                noise = np.random.normal(size=(batch_size, dimensionality),scale=0.01)
                for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(batch_size, always_random=always_random)):
                # for batch_i, (imgs_A, imgs_B) in enumerate(ds):
                    #try:
                    # if epoch % 2 == 0:
                    #     # print("Adding noise")
                    #     imgs_A = np.add(noise[0:imgs_A.shape[0], :], imgs_A)
                    #     imgs_B = np.add(noise[0:imgs_B.shape[0], :], imgs_B)

                    fake_B = self.g_AB.predict(imgs_A)
                    fake_A = self.g_BA.predict(imgs_B)
                    # Train the discriminators (original images = real / translated = Fake)
                    dA_loss = None
                    dB_loss = None
                    valid = np.ones((imgs_A.shape[0],))  # *noisy_entries_num,) )
                    fake = np.zeros((imgs_A.shape[0],))  # *noisy_entries_num,) )

                    for i in range(int(dis_train_amount)):
                        nimgs_A = imgs_A
                        nimgs_B = imgs_B

                        dA_loss_real = self.d_A.train_on_batch(nimgs_A, valid)
                        dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                        if dA_loss is None:
                            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
                        else:
                            dA_loss += 0.5 * np.add(dA_loss_real, dA_loss_fake)

                        dB_loss_real = self.d_B.train_on_batch(nimgs_B, valid)
                        dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                        if dB_loss is None:
                            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
                        else:
                            dB_loss += 0.5 * np.add(dB_loss_real, dB_loss_fake)
                    d_loss = 0.5 * np.add(dA_loss / dis_train_amount, dB_loss / dis_train_amount)
                    # Total disciminator loss
                    # ------------------
                    #  Train Generators
                    # ------------------
                    # Train the generators
                    # rand_a, rand_b = load_random_batch(batch_size=batch_size)
                    mm_a_loss = self.g_AB.train_on_batch(imgs_A, imgs_B)
                    mm_b_loss = self.g_BA.train_on_batch(imgs_B, imgs_A)
                    # self.d_A.trainable = False
                    # self.d_B.trainable = False

                    g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                          [valid, valid,
                                                           imgs_A, imgs_B,
                                                           imgs_A, imgs_B,
                                                           imgs_A, imgs_B])
                    # self.d_A.trainable = True
                    # self.d_B.trainable = True

                    def named_logs(model, logs):
                        result = {}
                        for l in zip(model.metrics_names, logs):
                            result[l[0]] = l[1]
                        return result

                    r = named_logs(self.combined, g_loss)
                    r.update({
                        'mma': mm_a_loss,
                        'mmb': mm_b_loss,
                    })

                    elapsed_time = datetime.datetime.now() - start_time
                    if batch_i % 50 == 0:
                        print(
                            "\n[Epoch %d/%d] [Batch %d] [D loss: %f, acc: %3d%%] "
                            "[G loss: %05f, adv: %05f, recon: %05f, recon_mm: %05f,id: %05f][mma:%05f,mmb:%05f]time: %s " \
                            % (epoch, training_epochs,
                               batch_i,
                               d_loss[0], 100* d_loss[1],
                               g_loss[0],
                               np.mean(g_loss[1:3]),
                               np.mean(g_loss[3:5]),
                               np.mean(g_loss[5:7]),
                               np.mean(g_loss[7:8]),
                               mm_a_loss,
                               mm_b_loss,
                               elapsed_time))
                        # wandbcb.on_batch_end(batch_i, r)
                        # wandb.log({"batch_num":batch_i,"epoch_num":epoch})
                        # self.combined_callback.on_batch_end(batch_i, r)

                    # except Exception as e:
                    #     print("There was a problem")
                    #     print("*"*100)
                    #     print(e)
                    #     print("*"*100)

                # try:
                # if epoch % 5 == 0:
                #     self.save_model(name=str(epoch))
                print("\n")
                sl = tools.test_sem(rcgan.g_AB, dataset, dataset_location="testing/SimLex-999.txt",
                                    fast_text_location="fasttext_model/cc.en.300.bin",prefix="en_")[0]
                sv = tools.test_sem(rcgan.g_AB, dataset, dataset_location="testing/SimVerb-3500.txt",
                                    fast_text_location="fasttext_model/cc.en.300.bin",prefix="en_")[0]
                if epoch%4==0:
                    self.save_model(name="checkpoint_"+str(epoch))

                res.append((sl, sv))
                # if epoch%10==0:
                #     testwords = ["human", "cat"]
                #     print("The test word vectors are:", testwords)
                    # ft version
                    # vals = np.array(
                    #     rcgan.g_AB.predict(np.array(X_train.values).reshape((-1, dimensionality)),
                    #                        batch_size=64)
                    # )

                    # testds = pd.DataFrame(data=vals, index=X_train.index)
                    # tools.datasets.update({"mine": [dataset["original"], dataset["retrofitted"]]})
                    # fastext_words = tools.find_in_fasttext(testwords, dataset="mine", prefix="en_")
                    # for idx, word in enumerate(testwords):
                    #     print(word)
                    #     retro_representation = rcgan.g_AB.predict(fastext_words[idx].reshape(1, dimensionality))
                    #     print(tools.find_closest_in_dataset(retro_representation, testds))

                #self.combined_callback.on_epoch_end(epoch, {"simlex": sl, "simverb": sv})
                #wandbcb.on_epoch_end(epoch, {"simlex": sl, "simverb": sv})

                print(res)
                print("\n")

        print("Actual training")
        train_(pretraining_epochs)
        print("SGD Fine tuning")
        self.compile_all("sgd")
        train_(epochs, always_random=True)
        print("Final performance")
        sl = tools.test_sem(rcgan.g_AB, dataset, dataset_location="testing/SimLex-999.txt",
                            fast_text_location="fasttext_model/cc.en.300.bin",prefix="en_")[0]
        sv = tools.test_sem(rcgan.g_AB, dataset, dataset_location="testing/SimVerb-3500.txt",
                            fast_text_location="fasttext_model/cc.en.300.bin",prefix="en_")[0]
        res.append((sl, sv))

        self.save_model(name="final")
        return res

    def save_model(self, name=""):
        self.d_A.save(os.path.join(self.save_folder, name + "fromretrodis.h5"), include_optimizer=False)
        self.d_B.save(os.path.join(self.save_folder, name + "toretrodis.h5"), include_optimizer=False)
        self.g_AB.save(os.path.join(self.save_folder, name + "toretrogen.h5"), include_optimizer=False)
        self.g_BA.save(os.path.join(self.save_folder, name + "fromretrogen.h5"), include_optimizer=False)
        self.combined.save(os.path.join(self.save_folder, name + "combined_model.h5"), include_optimizer=False)
        # wandb.save(os.path.join(self.save_folder, name + "toretrogen.h5"))


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


    ##input("press enter")
    print("Removing!!!")
    print("*" * 100)
    shutil.rmtree("logs/", ignore_errors=True)
    print("Done!")
    print("*" * 100)
    disable_eager_execution()
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)
    # print('Compute dtype: %s' % policy.compute_dtype)
    # print('Variable dtype: %s' % policy.variable_dtype)
    # with tf.device('/GPU:0'):
    tools.dimensionality = dimensionality
    postfix = "retrogan"
    test_ds = [
        #{
        #    "original":"original_ft_cc_nb.hdf",
        #    "retrofitted":"retrofitted_ft_cc_nb.hdf",
        #    "directory":"../conceptnetretrogan/",
        #    "rc":None
        #}
        #{
        #    "original":"completefastext.txt.hdf",
        #    "retrofitted":"fullfasttext.hdf",
        #    "directory":"ft_full_alldata/",
        #    "rc":None
        #},
        # {
        #     "original":"completefastext.txt.hdf",
        #     "retrofitted":"disjointfasttext.hdf",
        #     "directory":"ft_disjoint_alldata/",
        #     "rc":"adversarial_paper_data/simlexsimverb.words"
        # },
        # {
        #     "original":"completeglove.txt.hdf",
        #     "retrofitted":"fullglove.hdf",
        #     "directory":"glove_full_alldata/",
        #     "rc":None
        # },
        # {
        #     "original":"completeglove.txt.hdf",
        #     "retrofitted":"disjointglove.hdf",
        #     "directory":"glove_disjoint_alldata/",
        #     "rc":"adversarial_paper_data/simlexsimverb.words"
        # },
        # {
        #     "original": "completeglove.txt.hdf",
        #     "retrofitted": "disjointglove.hdf",
        #     "directory": "glove_disjoint_paperdata/",
        #     "rc": "adversarial_paper_data/simlexsimverb.words"
        # },
       # {
       #     "original": "completeglove.txt.hdf",
       #     "retrofitted": "fullglove.hdf",
       #     "directory": "glove_full_paperdata/",
       #     "rc": None
       # },
        # {
        #     "original": "completefastext.txt.hdf",
        #     "retrofitted": "disjointfasttext.hdf",
        #     "directory": "ft_disjoint_paperdata/",
        #     "rc": "adversarial_paper_data/simlexsimverb.words"
        # },
        #{
        #    "original": "completefastext.txt.hdf",
        #    "retrofitted": "fullfasttext.hdf",
        #    "directory": "ft_full_paperdata/",
        #    "rc": None
        #},
        # {
        #     "original": "cskg_ft_embs.h5",
        #     "retrofitted": "cskg_ft_ar_embs.h5",
        #     "directory": "cskg_atomic_data2/",
        #     "rc": None
        # },
        # {
        #     "original": "fasttext_seen.hdf",
        #     "retrofitted": "fasttext_seen_attractrepelretrofitted.hdf",
        #     "directory": "Data/ft_full/",
        #     "rc": None
        # },
        # {
        #     "original": "fasttext_seen.hdf",
        #     "retrofitted": "fasttext_seen_ook_attractrepelretrofitted.hdf",
        #     "directory": "Data/ft_ook/",
        #     "rc": None
        # },
        # {
        #     "original": "glove_seen.hdf",
        #     "retrofitted": "glove_seen_attractrepelretrofitted.hdf",
        #     "directory": "Data/glove_full/",
        #     "rc": None
        # },
        # {
        #     "original": "glove_seen.hdf",
        #     "retrofitted": "glove_seen_ook_attractrepelretrofitted.hdf",
        #     "directory": "Data/glove_ook/",
        #     "rc": None
        # }
        # {
        #     "original": "ft_nb_seen.h5",
        #     "retrofitted": "nb_retrofitted_attractrepelretrofitted.h5",
        #     "directory": "Data/nb_full/",
        #     "rc": None
        # },
        {
            "original": "ft_nb_seen.h5",
            "retrofitted": "nb_retrofitted_ook_attractrepel.h5",
            "directory": "Data/nb_ook/",
            "rc": None
        },
    ]
    print("Testing")
    print(test_ds)
    print("Checking that everything exists")
    for ds in test_ds:
        a = os.path.exists(os.path.join(ds["directory"], ds["original"]))
        b = os.path.exists(os.path.join(ds["directory"], ds["retrofitted"]))
        print(a, b)
        if not a or \
                not b:
            raise FileNotFoundError("Files in " + str(ds) + "\ndo not exist")
    models = []
    results = []
    final_save_folder = "./final_retrogan"
    os.makedirs(final_save_folder, exist_ok=True)

    for idx, ds in enumerate(test_ds):
        save_folder = "models/trained_retrogan/" + ds["directory"]+"500modarch"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        print("Training")
        print(ds)
        tools.directory = ds["directory"]
        bs = 32
        rcgan = RetroCycleGAN(save_folder=save_folder, batch_size=bs,
                              generator_lr=0.00005, discriminator_lr=0.0001)
        #rcgan.load_weights(preface="final", folder="/media/pedro/ssd_ext/OOVconverter/models/trained_retrogan/Data")
        sl = tools.test_sem(rcgan.g_AB, ds, dataset_location="testing/SimLex-999.txt",
                            fast_text_location="fasttext_model/cc.en.300.bin",prefix="en_")[0]
        #continue
        models.append(rcgan)
        ds_res = rcgan.train(pretraining_epochs=500, epochs=0, batch_size=bs, dataset=ds, rc=ds["rc"])
        results.append(ds_res)
        print("*" * 100)
        print(ds, results[-1])
        print("*" * 100)
        print("Saving")
        model_save_folder = os.path.join(final_save_folder, str(idx))
        os.makedirs(model_save_folder, exist_ok=True)
        with open(os.path.join(model_save_folder, "config"), "w") as f:
            f.write(str(ds))
        with open(os.path.join(model_save_folder, "results"), "w") as f:
            f.write(str(results[-1]))
        models[-1].save_folder = model_save_folder
        models[-1].save_model("final")
        print("Done")
