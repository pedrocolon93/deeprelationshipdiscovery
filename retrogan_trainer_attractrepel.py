from __future__ import print_function, division

import datetime
import math
import os
import random
from random import shuffle

import numpy as np
# from tensorflow_core.python.keras import backend as K
from numpy.random import seed
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.python.eager import *
from tensorflow.python.keras.layers import Input, Layer, Conv1D, Dense, multiply, add, BatchNormalization, \
    MaxPooling1D, Flatten, Dropout
from tensorflow.python.keras import Model
from tensorflow_core.python.framework.random_seed import set_random_seed
from tensorflow_core.python.keras.optimizer_v2.adam import Adam
from tensorflow_core.python.keras.utils.vis_utils import plot_model
from tqdm import tqdm
import tensorflow as tf

# tf.debugging.set_log_device_placement(True)

import tools

seed(1)

set_random_seed(2)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.compat.v1.summary.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

class ConstMultiplierLayer(Layer):
    def __init__(self, **kwargs):
        super(ConstMultiplierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.k = self.add_weight(
            name='k',
            shape=(),
            initializer='zeros',
            dtype='float32',
            trainable=True,
        )
        super(ConstMultiplierLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        return tf.math.multiply(self.k, x)

    def compute_output_shape(self, input_shape):
        return input_shape


def attention(layer_input,amount=None):
    # ATTENTION PART STARTS HERE
    if amount is not None:
        attention_probs = Dense(amount[1], activation='softmax')(layer_input)
    else:
        sh = tf.keras.backend.int_shape(layer_input)
        attention_probs = Dense(sh[1], activation='softmax')(layer_input)
    attention_mul = multiply([layer_input, attention_probs]
                             )
    attention_scale = ConstMultiplierLayer()(attention_mul)
    attention = add([layer_input, attention_scale])
    # ATTENTION PART FINISHES HERE
    return attention


class RetroCycleGAN():
    def __init__(self, save_index="0", save_folder="./", generator_size=32,
                 discriminator_size=64, word_vector_dimensions=300,
                 discriminator_lr=0.0004, generator_lr=0.0001,
                 lambda_cycle=3, lambda_id_weight=0.4,
                 clip_value=0.5, cn=4, batch_size=32):
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
        cv = clip_value
        cn = cn
        self.d_A = self.build_discriminator(name="word_vector_discriminator")
        self.d_B = self.build_discriminator(name="retrofitted_word_vector_discriminator")

        def create_opt(lr):
            return Adam(lr, clipvalue=cv, clipnorm=cn)

        self.d_A.compile(loss='mse',
                         optimizer=create_opt(d_lr),
                         metrics=['accuracy'])
        self.d_B.compile(loss='mse',
                         optimizer=create_opt(d_lr),
                         metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AB = self.build_generator(name="to_retro_generator")
        self.d_A.summary()
        self.g_AB.summary()
        plot_model(self.g_AB, show_shapes=True)
        self.g_BA = self.build_generator(name="from_retro_generator")
        self.d_B.summary()
        self.g_BA.summary()
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

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[unfit_wv, fit_wv],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       unfit_wv_id, fit_wv_id],
                              name="combinedmodel")
        def max_margin_loss(y_true,y_pred):
            cost = 0
            sim_neg = 5
            sim_margin = 0.6
            for i in range(0, sim_neg):
                new_true = tf.random.shuffle(y_true)
                normalize_a = tf.nn.l2_normalize(y_true)
                normalize_b = tf.nn.l2_normalize(y_pred)
                normalize_c = tf.nn.l2_normalize(new_true)
                minimize = tf.reduce_sum(tf.multiply(normalize_a,normalize_b))
                maximize = tf.reduce_sum(tf.multiply(normalize_c, normalize_b))
                mg = sim_margin - minimize + maximize
                cost += mg
                # cost += tf.clip_by_value(mg, min=0)
            return tf.keras.backend.mean(cost)

        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    max_margin_loss, max_margin_loss],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              # TODO ADD A CUSTOM LOSS THAT SIMPLY ADDS A
                              # GENERALIZATION CONSTRAINT ON THE MAE
                              optimizer=create_opt(g_lr))

        log_path = './logs'
        callback = keras.callbacks.TensorBoard(log_dir=log_path)
        callback.set_model(self.combined)
        self.combined_callback = callback


        plot_model(self.combined, to_file="RetroGAN.png", show_shapes=True)

    def build_generator(self, name):
        """U-Net Generator"""

        def dense(layer_input, filters, f_size=6, normalization=True):
            """Layers used during downsampling"""
            d = Dense(filters, activation="relu")(layer_input)
            if normalization:
                d = BatchNormalization()(d)
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
        d0 = dense(inpt, self.gf * 8, normalization=True)
        # d0 = dense(d0, self.gf * 8, normalization=True)
        # d0 = dense(d0, self.gf * 8, normalization=True)

        r = tf.compat.v2.keras.layers.Reshape((d0.get_shape()[-1], 1))(d0)
        # # Downscaling
        # t1 = conv1d(r,self.gf*8,f_size=6)
        t2 = conv1d(r, 512, f_size=8)
        t2 = conv1d(r, 128, f_size=4)
        # # t3 = conv1d(t2, self.gf, f_size=4)
        f = MaxPooling1D()(t2)
        f = Flatten()(f)
        # f = tf.compat.v1.keras.layers.Reshape((-1))(f)
        # # att
        # # if tf.keras.backend.int_shape(f)[1] is None:
        # #
        # #     shape_flatten = np.prod(shape_before_flatten)  # value of shape in the non-batch dimension
        # attn = attention(f)
        # # else:
        # attn = attention(d0)

        # VAE Like layer
        # latent_dim = 128
        # z_mean = Dense(latent_dim,
        #                     name='z_mean')(d2)
        # z_log_var = Dense(latent_dim,
        #                        name='z_log_var')(d2)
        #
        # # use reparameterization trick to push the sampling out as input
        # # note that "output_shape" isn't necessary with the TensorFlow backend
        # z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # Last 2 fc layers
        # Maybe attention
        # r = Reshape((-1, 1))(attn)
        # # Downscaling
        # t4 = conv1d(r, self.gf, f_size=4)
        # t5 = conv1d(t4, self.gf *2, f_size=4)
        # f = Flatten()(t5)
        # f = UpSampling1D(size=2)(t5)
        d4 = dense(f, self.gf * 8)
        output_img = Dense(dimensionality)(d4)
        return Model(inpt, output_img, name=name)

    def build_discriminator(self, name):

        def d_layer(layer_input, filters, f_size=7, normalization=True,dropout=True):
            """Discriminator layer"""
            d = Dense(filters, activation="relu")(layer_input)
            # d = LeakyReLU(alpha=0.0002)(d)
            if normalization:
                d = BatchNormalization()(d)
            if dropout:
                d = Dropout(0.5)(d)
            return d

        inpt = Input(shape=self.img_shape)
        d1 = d_layer(inpt, self.df * 16, normalization=False)
        d1 = d_layer(d1, self.df * 16)
        d1 = d_layer(d1, self.df * 16)
        d2 = d_layer(d1, self.df * 8)
        # d2 = attention(d2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 2,dropout=False)
        validity = Dense(1)(d4)
        return Model(inpt, validity, name=name)

    def load_weights(self):
        try:
            self.g_AB.reset_states()
            self.g_BA.reset_states()
            self.combined.reset_states()
            self.d_B.reset_states()
            self.d_A.reset_states()
            self.d_A.load_weights(os.path.join(self.save_folder, "fromretrodis.h5"))
            self.d_B.load_weights(os.path.join(self.save_folder, "toretrodis.h5"))
            self.g_AB.load_weights(os.path.join(self.save_folder, "toretro.h5"))
            self.g_BA.load_weights(os.path.join(self.save_folder, "fromretro.h5"))
            self.combined.load_weights(os.path.join(self.save_folder, "combined_model.h5"))

        except Exception as e:
            print(e)

    def train(self, epochs, dataset, batch_size=1, sample_interval=50, noisy_entries_num=5, batches=900,
              add_noise=False):
        # testwords = ["human", "dog", "cat", "potato", "fat"]
        # fastext_version = find_in_dataset(testwords, dataset=dataset["directory"] + dataset["original"])
        # retro_version = find_in_dataset(testwords, dataset=dataset["directory"] + dataset["retrofitted"])

        start_time = datetime.datetime.now()
        # self.load_weights(extension="0")
        # self.load_weights()
        # for idx, word in enumerate(testwords):
        #   print(word)
        #  retro_representation = rcgan.g_AB.predict(fastext_version[idx].reshape(1, dimensionality))
        # print(sklearn.metrics.mean_absolute_error(retro_version[idx],
        # retro_representation.reshape((dimensionality,))))
        # Adversarial loss ground truths
        # fake = np.random.uniform(0.0,0.1,size=(batch_size,))

        X_train = Y_train = X_test = Y_test = None

        seed = 32
        X_train, Y_train = tools.load_noisiest_words_dataset_2(dataset,
                                                               save_folder="fasttext_model/",
                                                               threshold=0.90,
                                                               cache=False)
        print("Done")

        # X_train, Y_train, X_test, Y_test = load_training_input_3(seed=seed,test_split=0.001,dataset=dataset)

        def load_batch(batch_size=2):
            l = X_train.shape[0]
            iterable = list(range(0, l))
            shuffle(iterable)
            for ndx in tqdm(range(0, l, batch_size), ncols=30):
                ixs = iterable[ndx:min(ndx + batch_size, l)]
                imgs_A = X_train[ixs]
                imgs_B = Y_train[ixs]
                yield imgs_A, imgs_B
        # def load_batch(batch_size=2,batch_amount=1000000):
        #     l = X_train.shape[0]
        #     iterable = list(range(0, l))
        #     shuffle(iterable)
        #     # for ndx in tqdm(range(0, l, batch_size), ncols=30):
        #     for i in tqdm(range(batch_amount)):
        #         ixs = random.sample(range(0,len(iterable)), batch_size)
        #         imgs_A = X_train[ixs]
        #         imgs_B = Y_train[ixs]
        #         yield imgs_A, imgs_B

        def step_decay(initial_lrate, epoch):
            drop = 0.5
            epochs_drop = 10.0
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lrate

        for epoch in range(epochs + 1):
            noise = np.random.normal(size=(batch_size, dimensionality), scale=0.01)
            # calculate learning rate:
            # g_current_learning_rate = step_decay(self.g_lr, epoch)
            # d_current_learning_rate = step_decay(self.d_lr, epoch)
            # K.set_value(self.d_A.optimizer.lr, d_current_learning_rate)  # set new lr
            # K.set_value(self.d_B.optimizer.lr, d_current_learning_rate)  # set new lr
            # K.set_value(self.combined.optimizer.lr, g_current_learning_rate)  # set new lr

            for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(batch_size)):
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # idx = np.random.randint(0, X_train.shape[0], batch_size)
                # noise = X_train[idx]
                # imgs = Y_train[idx]

                # if batch_i%2==0 and add_noise:
                #     normnoise = np.random.normal(size=(batch_size,300))
                #     noise = np.add(noise,normnoise)
                #     imgs = np.add(imgs,normnoise)

                # imgs_A = noise
                # imgs_B =imgs
                # Translate images to opposite domain
                if batch_i % 2 == 0:
                    print("Adding noise")
                    imgs_A = np.add(noise[0:imgs_A.shape[0], :], imgs_A)

                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                valid = np.ones((imgs_A.shape[0],), )  # *noisy_entries_num,) )
                # valid = np.random.uniform(0.9,1.0,size=(batch_size,))
                fake = np.zeros((imgs_A.shape[0],))  # *noisy_entries_num,) )

                # Train the discriminators (original images = real / translated = Fake)
                if batch_i % 4 == 0:
                    print("Training discriminators")
                    dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                    dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                    dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                    dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                    dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                    dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
                else:
                    dA_loss = [1.0,1]
                    dB_loss = [1.0,1]
                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                      [valid, valid,
                                                       imgs_A, imgs_B,
                                                       imgs_A, imgs_B])

                def named_logs(model, logs):
                    result = {}
                    for l in zip(model.metrics_names, logs):
                        result[l[0]] = l[1]
                    return result

                self.combined_callback.on_epoch_end(batch_i, named_logs(self.combined, g_loss))
                # profiler_result = profiler.stop()
                # profiler.save("./logs", profiler_result)
                elapsed_time = datetime.datetime.now() - start_time

                # if np.isnan(dA_loss_fake).any() or np.isnan(dA_loss_real).any() or np.isnan(
                #         dB_loss_fake).any() or np.isnan(dB_loss_real).any() or np.isnan(g_loss).any():
                #     print(np.isnan(dA_loss_fake).any(), np.isnan(dA_loss_real).any(), np.isnan(dB_loss_fake).any(),
                #           np.isnan(dB_loss_real).any(), np.isnan(g_loss).any())
                #     print("Problem")
                #     raise ArithmeticError("Problem with loss calculation")
                if batch_i % 500 == 0:
                    self.save_model()
                    print("\n")
                    tools.test_sem(rcgan.g_AB, dataset_location="testing/SimLex-999.txt",
                                   fast_text_location="fasttext_model/cc.en.300.bin")
                    print("\n")

                if batch_i % 100 == 0:
                    print(
                        "\n[Epoch %d/%d] [Batch %d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                        % (epoch, epochs,
                           batch_i,
                           d_loss[0], 100 * d_loss[1],
                           g_loss[0],
                           np.mean(g_loss[1:3]),
                           np.mean(g_loss[3:5]),
                           np.mean(g_loss[5:6]),
                           elapsed_time))
            try:
                self.save_model()
                tools.test_sem(rcgan.g_AB, dataset_location="testing/SimLex-999.txt",
                               fast_text_location="fasttext_model/cc.en.300.bin")

            except Exception as e:
                print(e)

        self.save_model()
        return X_train, Y_train, X_train, Y_train
        # return X_train, Y_train, X_test, Y_test

    def save_model(self):
        self.d_A.save(os.path.join(self.save_folder, "fromretrodis.h5"), include_optimizer=False)
        self.d_B.save(os.path.join(self.save_folder, "toretrodis.h5"), include_optimizer=False)
        self.g_AB.save(os.path.join(self.save_folder, "toretrogen.h5"), include_optimizer=False)
        self.g_BA.save(os.path.join(self.save_folder, "fromretrogen.h5"), include_optimizer=False)
        self.combined.save(os.path.join(self.save_folder, "combined_model.h5"), include_optimizer=False)


if __name__ == '__main__':
    profiler.start_profiler_server(6009)
    # with tf.device('/GPU:0'):
    global dimensionality
    dimensionality = 300
    tools.dimensionality = dimensionality
    # config.log_device_placement = True
    # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    # sess = tf.Session(config=config)
    # set_session(sess)  # set this TensorFlow session as the default session for Keras
    postfix = "ftar"
    save_folder = "fasttext_model/trained_retrogan/" + str(datetime.datetime.now()) + postfix
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    rcgan = RetroCycleGAN(save_folder=save_folder,batch_size=32)
    ds = {"original": "unfitted.hd5clean",
          "retrofitted": "attract_repel.hd5clean",
          "directory": "./fasttext_model/"}
    # X_train, Y_train, X_test, Y_test = rcgan.train(epochs=50, batch_size=32, sample_interval=100,
    #                                                dataset=ds)

    # rcgan.g_AB.load_weights(save_folder+"/toretrogen.h5")
    X_train, Y_train, X_test, Y_test = rcgan.train(epochs=10, batch_size=32, sample_interval=100,
                                                   dataset=ds)
        # rcgan.g_AB.load_weights("fasttext_model/trained_retrogan/2019-05-14 22:55:42.280715ft/toretrogen.h5")


        #
        # for idx,word in enumerate(testwords):
        #     print(word)
        #     retro_representation = rcgan.g_AB.predict(fastext_version[idx].reshape(1, dimensionality))
        #     tools.find_closest_in_dataset(retro_representation,n_top=20,dataset=ds["directory"]+ds["retrofitted"])
        #     print(sklearn.metrics.mean_absolute_error(retro_version[idx], retro_representation.reshape((dimensionality,))))