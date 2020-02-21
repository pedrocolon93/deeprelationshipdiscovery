from __future__ import print_function, division

import datetime
import math
import os
import random
import shutil
from random import shuffle

import numpy as np
# from tensorflow_core.python.keras import backend as K
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torch import randperm, clamp
from torch.optim import Adam

os.environ["TF_KERAS"] = "1"
from tqdm import tqdm

import tools

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
                 discriminator_lr=0.0001, generator_lr=0.0005,
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
        self.d_A = self.Discriminator(name="word_vector_discriminator")
        self.d_B = self.Discriminator(name="retrofitted_word_vector_discriminator")

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AB = self.Generator(name="to_retro_generator")
        self.g_BA = self.Generator(name="from_retro_generator")

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
        # Identity mapping of images
        unfit_wv_id = self.g_BA(unfit_wv)
        fit_wv_id = self.g_AB(fit_wv)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        mm_a = RetroCycleGAN.MaxMargin_Loss()
        mm_b = RetroCycleGAN.MaxMargin_Loss()
        id_ab = nn.L1Loss()
        id_ab = nn.L1Loss()
        cycle_ab = nn.L1Loss()
        cycle_ba = nn.L1Loss()
        adversarial_ab = nn.BCELoss()
        adversarial_ba = nn.BCELoss()

        mm_a_opt = Adam(lr=generator_lr)
        mm_b_opt = Adam(lr=generator_lr)
        adv_a_opt = Adam(lr=discriminator_lr)
        adv_b_opt = Adam(lr=discriminator_lr)
        id_a_opt = Adam(lr=generator_lr)
        id_b_opt = Adam(lr=generator_lr)
        cycle_opt = Adam(lr=generator_lr)


        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[unfit_wv, fit_wv],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       unfit_wv_id, fit_wv_id],
                              name="combinedmodel")

        log_path = './logs'
        callback = keras.callbacks.TensorBoard(log_dir=log_path)
        callback.set_model(self.combined)
        self.combined_callback = callback

        # plot_model(self.combined, to_file="RetroGAN.png", show_shapes=True)

    class MaxMargin_Loss(nn.Module):

        def __init__(self, sim_neg=25,batch_size=32,cuda=True,sim_margin=1.0):
            super(RetroCycleGAN.MaxMargin_Loss).__init__()
            self.sim_margin = sim_margin
            self.cuda=cuda
            self.sim_neg = sim_neg
            self.batch_size=batch_size

        def forward(self, y_pred, y_true):
            cost = 0.
            for i in range(0, self.params.sim_neg):
                new_true = randperm(self.params.batch_size)
                new_true = new_true.cuda() if self.params.cuda else new_true
                new_true = y_true[new_true]
                mg = self.params.sim_margin - F.cosine_similarity(y_true, y_pred) + F.cosine_similarity(new_true,
                                                                                                        y_pred)
                cost += clamp(mg, min=0)
            return cost.mean()


    class Generator(nn.Module):
        def __init__(self,name,input_image_shape=300,hidden_layers=2, hidden_dimension=2048, output_image_shape=300,norm=False,dropout=0.2):
            super(RetroCycleGAN.Generator, self).__init__()
            self.name = name
            self.norm = norm
            self.dropout_prob = dropout
            self.input = nn.Linear(input_image_shape,hidden_dimension)
            self.intermediate_layers = [nn.Linear(hidden_dimension,hidden_dimension) for x in range(hidden_layers)]
            self.batch_norm = [nn.BatchNorm1d(hidden_dimension) for x in range(hidden_layers)]
            self.dropout = [nn.Dropout(self.dropout_prob) for x in range(hidden_layers)]
            self.output = nn.Linear(hidden_dimension,output_image_shape)

        def forward(self, input):  # Input is a 1D tensor
            out = self.input(input)
            for i in range(len(self.intermediate_layers)):
                out = self.intermediate_layers[i](out)
                out = F.relu(out)
                if self.norm:
                    out = self.batch_norm[i](out)
                if self.dropout:
                    out = self.dropout[i](out)
            return self.output(out)

    class Discriminator(nn.Module):
        def __init__(self,name,input_image_shape=300,hidden_layers=2, hidden_dimension=2048, output_image_shape=1,norm=True,dropout=0.3):
            super(RetroCycleGAN.Discriminator, self).__init__()
            self.name = name
            self.norm = norm
            self.dropout_prob = dropout
            self.input = nn.Linear(input_image_shape,hidden_dimension)
            self.intermediate_layers = [nn.Linear(hidden_dimension,hidden_dimension) for x in range(hidden_layers)]
            self.batch_norm = [nn.BatchNorm1d(hidden_dimension) for x in range(hidden_layers)]
            self.dropout = [nn.Dropout(self.dropout_prob) for x in range(hidden_layers)]
            self.output = nn.Linear(hidden_dimension,output_image_shape)

        def forward(self, input):  # Input is a 1D tensor
            out = self.input(input)
            norm_skip = [0]
            for i in range(len(self.intermediate_layers)):
                out = self.intermediate_layers[i](out)
                out = F.relu(out)
                if self.norm:
                    if i not in norm_skip:
                        out = self.batch_norm[i](out)
                if self.dropout:
                    out = self.dropout[i](out)
            return F.sigmoid(self.output(out))


    def train(self, epochs, dataset, batch_size=1, pretraining_epochs=150, rc=None):
        start_time = datetime.datetime.now()
        res = []
        X_train = Y_train = None
        X_train, Y_train = tools.load_all_words_dataset_3(dataset,
                                                          save_folder="adversarial_paper_data/",
                                                          threshold=0.90,
                                                          cache=False,
                                                          remove_constraint=rc)
        print("Done")

        def load_batch(batch_size=32, always_random=False):
            iterable = list(X_train.index)
            shuffle(iterable)
            batches = []
            print("Prefetching batches")
            for ndx in tqdm(range(0, len(iterable), batch_size), ncols=30):
                ixs = iterable[ndx:min(ndx + batch_size, len(iterable))]
                if always_random:
                    ixs = list(np.array(iterable)[random.sample(range(0, len(iterable)), batch_size)])
                imgs_A = X_train.loc[ixs]
                imgs_B = Y_train.loc[ixs]
                batches.append((imgs_A, imgs_B))
            print("Begginging iteration")
            # ds = tf.data.Dataset.from_tensor_slices(batches)

            for i in tqdm(range(0, len(batches)), ncols=30):
                imgs_A, imgs_B = batches[i]
                yield imgs_A, imgs_B

        def load_random_batch(batch_size=32, batch_amount=1000000):
            iterable = list(X_train.index)
            # shuffle(iterable)
            ixs = random.sample(range(0, len(iterable)), batch_size)
            fixs = np.array(iterable)[ixs]
            imgs_A = X_train.loc[fixs]
            imgs_B = Y_train.loc[fixs]
            return imgs_A, imgs_B

        def exp_decay(epoch):
            initial_lrate = 0.1
            k = 0.1
            lrate = initial_lrate * math.exp(-k * epoch)
            return lrate

        # noise = np.random.normal(size=(batch_size, dimensionality), scale=0.01)
        dis_train_amount = 2.0

        self.compile_all("adam")

        def train_(training_epochs, always_random=False):
            for epoch in range(training_epochs):
                # dataset1 = tf.data.Dataset.from_generator(load_batch,
                #                                           (tf.float32, tf.float32),
                #                                           args=(batch_size, True)
                #                                           )
                # for count_batch in dataset1.take(5).interleave(4).prefetch(3):
                #     print(count_batch)
                # exit(1)
                for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(batch_size, always_random=always_random)):
                    try:
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
                            # if i % 2 == 0:
                            #     print("Adding noise")
                            #     nimgs_A = np.add(noise[0:imgs_A.shape[0], :], imgs_A)
                            #     nimgs_B = np.add(noise[0:imgs_B.shape[0], :], imgs_B)

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
                        rand_a, rand_b = load_random_batch(batch_size=32)
                        mm_a_loss = self.g_AB.train_on_batch(rand_a, rand_b)
                        mm_b_loss = self.g_BA.train_on_batch(rand_b, rand_a)

                        g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                              [valid, valid,
                                                               imgs_A, imgs_B,
                                                               imgs_A, imgs_B])

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
                        # if r["loss"] >10:
                        #     self.combined.compile(loss=['binary_crossentropy', 'binary_crossentropy',
                        #                                 'mae', 'mae',
                        #                                 'mae', 'mae'],
                        #                           loss_weights=[1, 1,
                        #                                         0,0,
                        #                                         0,0],
                        #                           # TODO ADD A CUSTOM LOSS THAT SIMPLY ADDS A
                        #                           # GENERALIZATION CONSTRAINT ON THE MAE
                        #                           optimizer=tf.optimizers.Adam(lr=0.0001,epsilon=1e-9))
                        #     self.combined_callback.on_epoch_end(epoch,{"nerfed_losses":1})

                        self.combined_callback.on_epoch_end(batch_i, r)
                        # profiler_result = profiler.stop()
                        # profiler.save("./logs", profiler_result)
                        elapsed_time = datetime.datetime.now() - start_time
                        if batch_i % 50 == 0:
                            print(
                                "\n[Epoch %d/%d] [Batch %d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f][mma:%05f,mmb:%05f]time: %s " \
                                % (epoch, training_epochs,
                                   batch_i,
                                   d_loss[0], 100 * d_loss[1],
                                   g_loss[0],
                                   np.mean(g_loss[1:3]),
                                   np.mean(g_loss[3:5]),
                                   np.mean(g_loss[5:6]),
                                   mm_a_loss,
                                   mm_b_loss,
                                   elapsed_time))
                    except Exception as e:
                        print("There was a problem")
                        print("*"*100)
                        print(e)
                        print("*"*100)

                # try:
                # if epoch % 5 == 0:
                #     self.save_model(name=str(epoch))
                self.save_model(name="checkpoint")
                print("\n")
                sl = tools.test_sem(rcgan.g_AB, dataset, dataset_location="testing/SimLex-999.txt",
                                    fast_text_location="fasttext_model/cc.en.300.bin")[0]
                sv = tools.test_sem(rcgan.g_AB, dataset, dataset_location="testing/SimVerb-3500.txt",
                                    fast_text_location="fasttext_model/cc.en.300.bin")[0]
                res.append((sl, sv))

                testwords = ["human", "cat"]
                print("The test word vectors are:", testwords)
                # ft version
                vals = np.array(
                    rcgan.g_AB.predict(np.array(X_train.values).reshape((-1, dimensionality)),
                                       batch_size=64)
                )

                testds = pd.DataFrame(data=vals, index=X_train.index)
                tools.datasets.update({"mine": [dataset["original"], dataset["retrofitted"]]})
                fastext_words = tools.find_in_fasttext(testwords, dataset="mine", prefix="en_")
                for idx, word in enumerate(testwords):
                    print(word)
                    retro_representation = rcgan.g_AB.predict(fastext_words[idx].reshape(1, dimensionality))
                    print(tools.find_closest_in_dataset(retro_representation, testds))

                self.combined_callback.on_epoch_end(epoch, {"simlex": sl, "simverb": sv})
                if epoch ==124:
                    print("Dropping the learning rate")
                    g1_current_learning_rate =K.eval(self.g_AB.optimizer.lr)/10.0
                    g2_current_learning_rate = K.eval(self.g_BA.optimizer.lr)/10.0
                    comb_current_learning_rate = K.eval(self.combined.optimizer.lr)/10.0
                    d_current_learning_rate =K.eval(self.d_A.optimizer.lr)/10.0
                    d2_current_learning_rate = K.eval(self.d_B.optimizer.lr)/10.0
                    K.set_value(self.d_A.optimizer.lr, d_current_learning_rate)  # set new lr
                    K.set_value(self.d_B.optimizer.lr, d2_current_learning_rate)  # set new lr
                    K.set_value(self.g_AB.optimizer.lr, g1_current_learning_rate)  # set new lr
                    K.set_value(self.g_BA.optimizer.lr, g2_current_learning_rate)  # set new lr
                    K.set_value(self.combined.optimizer.lr, comb_current_learning_rate)  # set new lr

                print(res)
                print("\n")
                # except Exception as e:
                #     print(e)

        print("Actual training")
        train_(pretraining_epochs)
        print("SGD Fine tuning")
        self.compile_all("sgd")
        train_(epochs, always_random=True)
        print("Final performance")
        sl = tools.test_sem(rcgan.g_AB, dataset, dataset_location="testing/SimLex-999.txt",
                            fast_text_location="fasttext_model/cc.en.300.bin")[0]
        sv = tools.test_sem(rcgan.g_AB, dataset, dataset_location="testing/SimVerb-3500.txt",
                            fast_text_location="fasttext_model/cc.en.300.bin")[0]
        res.append((sl, sv))

        self.save_model(name="final")
        return res

    def save_model(self, name=""):
        self.d_A.save(os.path.join(self.save_folder, name + "fromretrodis.h5"), include_optimizer=False)
        self.d_B.save(os.path.join(self.save_folder, name + "toretrodis.h5"), include_optimizer=False)
        self.g_AB.save(os.path.join(self.save_folder, name + "toretrogen.h5"), include_optimizer=False)
        self.g_BA.save(os.path.join(self.save_folder, name + "fromretrogen.h5"), include_optimizer=False)
        self.combined.save(os.path.join(self.save_folder, name + "combined_model.h5"), include_optimizer=False)


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
    print("Removing!!!")
    print("*" * 100)
    shutil.rmtree("logs/", ignore_errors=True)
    print("Done!")
    print("*" * 100)
    disable_eager_execution()
    # with tf.device('/GPU:0'):
    tools.dimensionality = dimensionality
    postfix = "ftar"
    test_ds = [
        # {
        #     "original":"completefastext.txt.hdf",
        #     "retrofitted":"fullfasttext.hdf",
        #     "directory":"ft_full_alldata/",
        #     "rc":None
        # },
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
        {
            "original": "completefastext.txt.hdf",
            "retrofitted": "fullfasttext.hdf",
            "directory": "ft_full_paperdata/",
            "rc": None
        }
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
        save_folder = "models/trained_retrogan/" + ds["directory"]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        print("Training")
        print(ds)
        tools.directory = ds["directory"]
        rcgan = RetroCycleGAN(save_folder=save_folder, batch_size=32, generator_lr=0.0001, discriminator_lr=0.001)

        # rcgan.load_weights(preface="final", folder="/media/pedro/ssd_ext/mltests/models/trained_retrogan/2020-01-27 00:34:26.680643ftar")
        sl = tools.test_sem(rcgan.g_AB, ds, dataset_location="testing/SimLex-999.txt",
                            fast_text_location="fasttext_model/cc.en.300.bin")[0]
        models.append(rcgan)
        ds_res = rcgan.train(pretraining_epochs=250, epochs=0, batch_size=32, dataset=ds, rc=ds["rc"])
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
