import datetime
import os
import random
from random import shuffle

import numpy as np
import tensorflow as tf
import wandb
from tensorflow import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout
from tqdm import tqdm

import tools


# random.seed(10)
# np.random.seed(10)
# tf.random.set_seed(10)


class RetroCycleGAN:
    def __init__(self, save_index="0", save_folder="./", generator_size=32,
                 discriminator_size=64, word_vector_dimensions=300,
                 discriminator_lr=0.0001, generator_lr=0.0001,
                 lambda_cycle=1, lambda_id_weight=0.01, one_way_mm=True,
                 cycle_mm=True,
                 cycle_dis=True,
                 id_loss=True,
                 cycle_mm_w=2,
                 cycle_loss=True):
        self.cycle_mm = cycle_mm
        self.cycle_dis = cycle_dis
        self.cycle_mae = cycle_loss
        self.id_loss = id_loss
        self.one_way_mm = one_way_mm
        self.cycle_mm_w = cycle_mm_w if self.cycle_mm else 0
        self.save_folder = save_folder

        # Input shape
        self.word_vector_dimensions = word_vector_dimensions
        self.embeddings_dimensionality = (self.word_vector_dimensions,)  # , self.channels)
        self.save_index = save_index

        # Number of filters in the first layer of G and D
        self.gf = generator_size
        self.df = discriminator_size

        # Loss weights
        self.lambda_cycle = lambda_cycle  if self.cycle_mae else 0# Cycle-consistency loss
        self.lambda_id = lambda_id_weight if self.id_loss else 0  # Identity loss

        d_lr = discriminator_lr
        self.d_lr = d_lr
        g_lr = generator_lr
        self.g_lr = g_lr
        # cv = clip_value
        # cn = cn
        self.d_A = self.build_discriminator(name="word_vector_discriminator")
        self.d_B = self.build_discriminator(name="retrofitted_word_vector_discriminator")
        self.d_ABBA = self.build_c_discriminator(name="cycle_cond_discriminator_unfit")
        self.d_BAAB = self.build_c_discriminator(name="cycle_cond_discriminator_fit")
        # Best combo sofar SGD, gaussian, dropout,5,0.5 mml(0,5,.5),3x1024gen, 2x1024, no normalization

        # return Adam(lr,amsgrad=True,decay=1e-8)

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AB = self.build_generator(name="to_retro_generator")
        # for layer in self.g_AB.layers:
        #     a = layer.get_weights()
        # print(a)

        # self.d_A.summary()
        # self.g_AB.summary()
        # plot_model(self.g_AB, show_shapes=True)
        self.g_BA = self.build_generator(name="from_retro_generator")

        # self.d_B.summary()
        # self.g_BA.summary()
        # Input images from both domains
        unfit_wv = Input(shape=self.embeddings_dimensionality, name="plain_word_vector")
        fit_wv = Input(shape=self.embeddings_dimensionality, name="retrofitted_word_vector")
        #

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
        # self.d_ABBA.trainable = False
        # self.d_BAAB.trainable = False

        self.combined = Model(inputs=[unfit_wv, fit_wv],  # Model that does A->B->A (left), B->A->B (right)
                              outputs=[valid_A, valid_B,  # for the bce calculation
                                       reconstr_A, reconstr_B,  # for the mae calculation
                                       reconstr_A, reconstr_B,  # for the max margin calculation
                                       unfit_wv_id, fit_wv_id,
                                       # dAc_r, dBc_r,  # for the conditional discriminator margin calculation
                                       # dAc_fake, dBc_fake  # for the conditional discriminator margin calculation
                                       ],  # for the id loss calculation
                              name="combinedmodel")

        log_path = './logs'
        callback = keras.callbacks.TensorBoard(log_dir=log_path)
        callback.set_model(self.combined)
        self.combined_callback = callback

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
                # print(mg)
                cost += tf.keras.backend.clip(mg, 0, 1000)
            return cost / (sim_neg * 1.0)

        def create_opt(lr=0.1):
            if optimizer == "adam":
                opt = tf.optimizers.Adam(lr=lr, epsilon=1e-10)
                return opt
            else:
                raise KeyError("coULD NOT FIND THE OPTIMIZER")
        # self.d_A.trainable = True
        # self.d_B.trainable = True

        self.d_A.compile(loss='binary_crossentropy',
                         optimizer=create_opt(self.d_lr),
                         metrics=['accuracy'])
        self.d_ABBA.compile(loss='binary_crossentropy',
                            optimizer=create_opt(self.d_lr),
                            metrics=['accuracy'])
        self.d_BAAB.compile(loss='binary_crossentropy',
                            optimizer=create_opt(self.d_lr),
                            metrics=['accuracy'])

        self.d_B.compile(loss='binary_crossentropy',
                         optimizer=create_opt(self.d_lr),
                         metrics=['accuracy'])
        # self.d_A.trainable = False
        # self.d_B.trainable = False

        self.g_AB.compile(loss=max_margin_loss,
                          optimizer=create_opt(self.g_lr),
                          )
        self.g_BA.compile(loss=max_margin_loss,
                          optimizer=create_opt(self.g_lr),
                          )

        self.combined.compile(loss=['binary_crossentropy', 'binary_crossentropy',
                                    'mae', 'mae',
                                    max_margin_loss, max_margin_loss,
                                    'mae', 'mae',
                                    ],
                              loss_weights=[1, 1,
                                            self.lambda_cycle * 1, self.lambda_cycle * 1,
                                            self.cycle_mm_w, self.cycle_mm_w,
                                            self.lambda_id, self.lambda_id,
                                            # self.lambda_cycle * 1, self.lambda_cycle * 1,
                                            # self.lambda_cycle * 1, self.lambda_cycle * 1
                                            ],
                              optimizer=create_opt(self.g_lr))
        # self.combined.summary()
        self.g_AB.summary()
        self.d_A.summary()
        self.combined.summary()

    def build_generator(self, name, hidden_dim=2048):
        """U-Net Generator"""

        def dense(layer_input, hidden_dim, normalization=True, dropout=True, dropout_percentage=0.2):
            d = Dense(hidden_dim, activation="relu")(layer_input)
            if normalization:
                d = BatchNormalization()(d)
            if dropout:
                d = Dropout(dropout_percentage)(d)
            return d

        # Image input
        inpt = Input(shape=self.embeddings_dimensionality)
        encoder = dense(inpt, hidden_dim, normalization=False, dropout=True, dropout_percentage=0.2)

        decoder = dense(encoder, hidden_dim, normalization=False, dropout=True, dropout_percentage=0.2)  # +encoder
        output = Dense(self.word_vector_dimensions)(decoder)
        return Model(inpt, output, name=name)

    def build_discriminator(self, name, hidden_dim=2048):

        def d_layer(layer_input, hidden_dim, normalization=True, dropout=True, dropout_percentage=0.3):
            """Discriminator layer"""
            d = Dense(hidden_dim, activation="relu")(layer_input)
            if normalization:
                d = BatchNormalization()(d)
            if dropout:
                d = Dropout(dropout_percentage)(d)
            return d

        inpt = Input(shape=self.embeddings_dimensionality)
        d1 = d_layer(inpt, hidden_dim, normalization=False, dropout=True, dropout_percentage=0.3)
        d1 = d_layer(d1, hidden_dim, normalization=True, dropout=True, dropout_percentage=0.3)
        validity = Dense(1, activation="sigmoid", dtype='float32')(d1)
        return Model(inpt, validity, name=name)

    def build_c_discriminator(self, name, hidden_dim=2048):

        def d_layer(layer_input, hidden_dim, normalization=True, dropout=True, dropout_percentage=0.3):
            """Discriminator layer"""
            d = Dense(hidden_dim, activation="relu")(layer_input)
            if normalization:
                d = BatchNormalization()(d)
            if dropout:
                d = Dropout(dropout_percentage)(d)
            return d

        inpt = Input(shape=600)
        d1 = d_layer(inpt, hidden_dim, normalization=False, dropout=True, dropout_percentage=0.3)
        d1 = d_layer(d1, hidden_dim, normalization=True, dropout=True, dropout_percentage=0.3)
        validity = Dense(1, activation="sigmoid", dtype='float32')(d1)
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

    def train(self, epochs, dataset, save_folder, name, batch_size=1, cache=False, epochs_per_checkpoint=4,
              dis_train_amount=3):
        wandb.init(project="retrogan", dir=save_folder)
        wandb.run.name = name
        # wandb.watch(self.g_AB,criterion="simlex")
        wandb.run.save()
        self.name = name
        start_time = datetime.datetime.now()
        res = []
        X_train, Y_train = tools.load_all_words_dataset_final(dataset["original"], dataset["retrofitted"],
                                                              save_folder=save_folder, cache=cache)
        print("Shapes of training data:",
              X_train.shape,
              Y_train.shape)
        print(X_train)
        print(Y_train)
        print("*" * 100)

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
                        if np.isnan(imgs_A).any().any() or np.isnan(imgs_B).any().any():  # np.isnan(imgs_B).any():
                            # print(ixs)
                            continue

                        batches.append((imgs_A, imgs_B))
                    except Exception as e:
                        print("Skipping batch")
                        # print(e)
                return batches

            batches = _int_load()

            print("Beginning iteration")
            for i in tqdm(range(0, len(batches)), ncols=30):
                imgs_A, imgs_B = batches[i]
                yield np.array(imgs_A.values, dtype=np.float32), np.array(imgs_B.values, dtype=np.float32)

        # def load_random_batch(batch_size=32, batch_amount=1000000):
        #     iterable = list(Y_train.index)
        #     # shuffle(iterable)
        #     ixs = list(np.array(iterable)[random.sample(range(0, len(iterable)), batch_size)])
        #     imgs_A = X_train.loc[ixs]
        #     imgs_B = Y_train.loc[ixs]
        #     def test_nan(a,b):
        #         return np.isnan(a).any().any() or np.isnan(b).any().any()
        #     while True:
        #         if(test_nan(imgs_A,imgs_B)):
        #             ixs = list(np.array(iterable)[random.sample(range(0, len(iterable)), batch_size)])
        #             imgs_A = X_train.loc[ixs]
        #             imgs_B = Y_train.loc[ixs]
        #         else:
        #             break
        #     return imgs_A, imgs_B
        #
        # def exp_decay(epoch):
        #     initial_lrate = 0.1
        #     k = 0.1
        #     lrate = initial_lrate * math.exp(-k * epoch)
        #     return lrate

        # noise = np.random.normal(size=(1, dimensionality), scale=0.001)
        # noise = np.tile(noise,(batch_size,1))
        dis_train_amount = dis_train_amount

        self.compile_all("adam")

        # ds = tf.data.Dataset.from_generator(load_batch,(tf.float32,tf.float32),args=(batch_size,))
        # ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        def train_(training_epochs, always_random=False):
            global_step = 0
            for epoch in range(training_epochs):
                # noise = np.random.normal(size=(batch_size, dimensionality), scale=0.01)
                for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(batch_size, always_random=always_random)):
                    global_step += 1
                    # for batch_i, (imgs_A, imgs_B) in enumerate(ds):
                    # try:
                    # if epoch % 2 == 0:
                    #     # print("Adding noise")
                    #     imgs_A = np.add(noise[0:imgs_A.shape[0], :], imgs_A)
                    #     imgs_B = np.add(noise[0:imgs_B.shape[0], :], imgs_B)
                    # imgs_A = tf.cast(imgs_A, tf.float32)
                    # imgs_B = tf.cast(imgs_B, tf.float32)

                    fake_B = self.g_AB.predict(imgs_A)
                    fake_A = self.g_BA.predict(imgs_B)
                    fake_ABBA = self.g_BA.predict(fake_B)
                    fake_BAAB = self.g_AB.predict(fake_A)
                    # Train the discriminators (original images = real / translated = Fake)
                    dA_loss = None
                    dB_loss = None
                    valid = np.ones((imgs_A.shape[0],))  # *noisy_entries_num,) )
                    fake = np.zeros((imgs_A.shape[0],))  # *noisy_entries_num,) )
                    # self.d_A.trainable = True
                    # self.d_B.trainable = True

                    for _ in range(int(dis_train_amount)):
                        # da = self.d_A.evaluate(imgs_A)
                        dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                        # daf = self.d_A(fake_A)
                        dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                        if dA_loss is None:
                            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
                        else:
                            dA_loss += 0.5 * np.add(dA_loss_real, dA_loss_fake)
                        dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                        dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                        if dB_loss is None:
                            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
                        else:
                            dB_loss += 0.5 * np.add(dB_loss_real, dB_loss_fake)
                    d_loss = (1.0 / dis_train_amount) * 0.5 * np.add(dA_loss, dB_loss)
                    # self.d_A.trainable = False
                    # self.d_B.trainable = False

                    def CycleCondLoss(d_ground, d_approx):
                        l = tf.math.log(d_ground) + tf.math.log(1 - d_approx)
                        return -1 * tf.reduce_mean(l)

                    # train cycle discriminators
                    d_cycle_dis = 0
                    g_cycle_dis = 0
                    if self.cycle_dis:
                        with tf.GradientTape() as tape:
                            dA = self.d_ABBA(tf.concat([fake_B, imgs_A], 1))
                            dA_r = self.d_ABBA(tf.concat([fake_B, fake_ABBA], 1))
                            la = CycleCondLoss(dA, dA_r)
                            tga = tape.gradient(la, self.d_ABBA.trainable_variables)
                            self.d_ABBA.optimizer.apply_gradients(zip(tga, self.d_ABBA.trainable_variables))
                            d_cycle_dis += la

                        with tf.GradientTape() as tape:
                            dB = self.d_BAAB(tf.concat([fake_A, imgs_B], 1))
                            dB_r = self.d_BAAB(tf.concat([fake_A, fake_BAAB], 1))
                            lb = CycleCondLoss(dB, dB_r)
                            tgb = tape.gradient(lb, self.d_BAAB.trainable_variables)
                            self.d_BAAB.optimizer.apply_gradients(zip(tgb, self.d_BAAB.trainable_variables))
                            d_cycle_dis += lb
                        with tf.GradientTape() as tape:
                            fake_B = self.g_AB(imgs_A)
                            fake_A = self.g_BA(imgs_B)
                            fake_ABBA = self.g_BA(fake_B)
                            fake_BAAB = self.g_AB(fake_A)
                            dB = self.d_BAAB(tf.concat([fake_A, imgs_B], 1))
                            dB_r = self.d_BAAB(tf.concat([fake_A, fake_BAAB], 1))

                            dA = self.d_ABBA(tf.concat([fake_B, imgs_A], 1))
                            dA_r = self.d_ABBA(tf.concat([fake_B, fake_ABBA], 1))
                            la = CycleCondLoss(dA, dA_r)
                            lb = CycleCondLoss(dB, dB_r)

                            tga = tape.gradient((la + lb) / 2.0, self.combined.trainable_variables)
                            self.combined.optimizer.apply_gradients(zip(tga, self.combined.trainable_variables))
                            g_cycle_dis += (la + lb) / 2.0

                    # Calculate the max margin loss for A->B, B->A
                    mm_b_loss = 0
                    mm_a_loss = 0
                    if self.one_way_mm:
                        mm_a_loss = self.g_AB.train_on_batch(imgs_A, imgs_B)
                        mm_b_loss = self.g_BA.train_on_batch(imgs_B, imgs_A)
                    # Calculate the cycle A->B->A, B->A->B with max margin, and mae
                    # Train cycle dis

                    g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                          [valid, valid,
                                                           imgs_A, imgs_B,
                                                           imgs_A, imgs_B,
                                                           imgs_A, imgs_B,
                                                           # valid,valid,
                                                           # valid,valid
                                                           ])

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
                    if batch_i % 50 == 0 and batch_i != 0:
                        print(
                            "\n[Epoch %d/%d] [Batch %d] [D loss: %f, acc: %3d%%] "
                            "[G loss: %05f, adv: %05f, recon: %05f, recon_mm: %05f,id: %05f][mma:%05f,mmb:%05f]time: %s " \
                            % (epoch, training_epochs,
                               batch_i,
                               d_loss[0], 100 * d_loss[1],
                               g_loss[0],
                               np.mean(g_loss[1:3]),
                               np.mean(g_loss[3:5]),
                               np.mean(g_loss[5:7]),
                               np.mean(g_loss[7:8]),
                               mm_a_loss,
                               mm_b_loss,
                               elapsed_time))

                        scalars = {
                            "epoch": epoch,
                            # "batch": batch_i,
                            "global_step": global_step,
                            "discriminator_loss": d_loss[0],
                            "discriminator_acc": d_loss[1],
                            "combined_loss": g_loss[0]+g_cycle_dis+d_cycle_dis,
                            "loss": g_loss[0] + d_loss[0],
                            "cycle_da": g_loss[1],
                            "cycle_db": g_loss[2],
                            "cycle_dis": d_cycle_dis,
                            "cycle_gen_condis":g_cycle_dis,
                            "MM_ABBA_CYCLE": g_loss[5],
                            "MM_BAAB_CYCLE": g_loss[6],
                            "abba_mae": g_loss[3],
                            "baab_mae": g_loss[4],
                            "idloss_ab": g_loss[7],
                            "idloss_ba": g_loss[8],
                            "mm_ab_loss": mm_a_loss,
                            "mm_ba_loss": mm_b_loss,
                        }
                        wandb.log(scalars, step=global_step)

                        # wandbcb.on_batch_end(batch_i, r)
                        # wandb.log({"batch_num":batch_i,"epoch_num":epoch})
                        # self.combined_callback.on_batch_end(batch_i, r)

                print("\n")
                sl, sv,c = self.test(dataset)
                if epoch % epochs_per_checkpoint == 0 and epoch != 0:
                    self.save_model(name="checkpoint")

                res.append((sl, sv, c))
                wandb.log({"simlex": sl, "simverb": sv, "card":c,"epoch": epoch})

                # self.combined_callback.on_epoch_end(epoch, {"simlex": sl, "simverb": sv})
                # wandbcb.on_epoch_end(epoch, {"simlex": sl, "simverb": sv})

                print(res)
                print("\n")

        print("Actual training")
        train_(epochs)
        print("Final performance")
        sl, sv,c = self.test(dataset)
        res.append((sl, sv,c))

        self.save_model(name="final")
        return res

    def test(self, dataset, simlex="testing/SimLex-999.txt", simverb="testing/SimVerb-3500.txt",card="testing/card660.tsv",
             fasttext="fasttext_model/cc.en.300.bin",
             prefix="en_"):
        sl = tools.test_sem(self.g_AB, dataset, dataset_location=simlex,
                            fast_text_location=fasttext, prefix=prefix,pt=False)[0]
        sv = tools.test_sem(self.g_AB, dataset, dataset_location=simverb,
                            fast_text_location=fasttext, prefix=prefix,pt=False)[0]
        c = tools.test_sem(self.g_AB, dataset, dataset_location=card,
                            fast_text_location=fasttext, prefix=prefix,pt=False)[0]
        return sl, sv,c

    def save_model(self, name=""):
        self.d_A.save(os.path.join(self.save_folder, name + "fromretrodis.h5"), include_optimizer=False)
        self.d_B.save(os.path.join(self.save_folder, name + "toretrodis.h5"), include_optimizer=False)
        self.g_AB.save(os.path.join(self.save_folder, name + "toretrogen.h5"), include_optimizer=False)
        self.g_BA.save(os.path.join(self.save_folder, name + "fromretrogen.h5"), include_optimizer=False)
        self.combined.save(os.path.join(self.save_folder, name + "combined_model.h5"), include_optimizer=False)
        # wandb.save(os.path.join(self.save_folder, name + "toretrogen.h5"))
