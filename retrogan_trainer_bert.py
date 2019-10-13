from __future__ import print_function, division

import datetime
import os
from random import shuffle

import numpy as np
import sklearn
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.engine import Layer
from keras.engine.saving import load_model
from keras.layers import BatchNormalization, add, multiply, Conv1D, Reshape, Flatten, UpSampling1D, MaxPooling1D, \
    Concatenate
from keras.layers import Input, Dense
from keras.losses import mean_absolute_error
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from numpy.random import seed
from tqdm import tqdm

import tools
from tools import load_noisiest_words_dataset, find_in_dataset
from bert_serving.client import BertClient

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ConstMultiplierLayer(Layer):
    def __init__(self, **kwargs):
        super(ConstMultiplierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.k = self.add_weight(
            name='k',
            shape=(),
            initializer='random_uniform',
            dtype='float32',
            trainable=True,
        )
        super(ConstMultiplierLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        return K.tf.multiply(self.k, x)

    def compute_output_shape(self, input_shape):
        return input_shape


def attention(layer_input):
    # ATTENTION PART STARTS HERE
    attention_probs = Dense(layer_input._keras_shape[1], activation='softmax')(layer_input)
    print(attention_probs)
    attention_mul = multiply([layer_input, attention_probs])
    attention_scale = ConstMultiplierLayer()(attention_mul)
    attention = add([layer_input, attention_scale])
    # ATTENTION PART FINISHES HERE
    return attention

class RetroCycleGAN():
    def __init__(self, save_index="0", save_folder="./", generator_size=32,
                 discriminator_size=64, word_vector_dimensions=300,
                 discriminator_lr=0.0004, generator_lr=0.0001,
                 lambda_cycle=3, lambda_id_weight=0.4,
                 clip_value=0.5, cn=4):
        self.save_folder = save_folder
        # Input shape
        self.word_vector_dimensions = word_vector_dimensions
        self.vector_shape = (self.word_vector_dimensions,)  # , self.channels)
        self.context_vector_shape = (768,)
        self.save_index = save_index

        # Number of filters in the first layer of G and D
        self.gf = generator_size
        self.df = discriminator_size

        # Loss weights
        self.lambda_cycle = lambda_cycle  # Cycle-consistency loss
        self.lambda_id = lambda_id_weight * self.lambda_cycle  # Identity loss

        d_lr = discriminator_lr
        g_lr = generator_lr
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

        self.build_bert_input()

        # Build the generators
        self.g_AB = self.build_generator(name="to_retro_generator")
        self.g_AB.summary()
        plot_model(self.g_AB, show_shapes=True)
        self.g_BA = self.build_generator(name="from_retro_generator")
        self.g_BA.summary()

        # Input images from both domains
        unfit_wv = Input(shape=self.vector_shape, name="plain_word_vector")
        fit_wv = Input(shape=self.vector_shape, name="retrofitted_word_vector")
        bert_in = Input(shape=(768,),name="bert_feeder")
        # Translate images to the other domain
        fake_B = self.g_AB([unfit_wv,bert_in])
        fake_A = self.g_BA([fit_wv,bert_in])
        # Translate images back to original domain
        reconstr_A = self.g_BA([fake_B,bert_in])
        reconstr_B = self.g_AB([fake_A,bert_in])
        # Identity mapping of images
        unfit_wv_id = self.g_BA([unfit_wv,bert_in])
        fit_wv_id = self.g_AB([fit_wv,bert_in])

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        def mae_plus_constant_loss(y_true, y_pred):
            constant = 0.0001
            return mean_absolute_error(y_true,y_pred)+constant

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[unfit_wv, fit_wv,bert_in],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       unfit_wv_id, fit_wv_id],
                              name="combinedmodel")
        self.combined.compile(loss=['mse', 'mse',
                                    mae_plus_constant_loss, mae_plus_constant_loss,
                                    'mae', 'mae'],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id
                                            ],

                              optimizer=create_opt(g_lr))
        plot_model(self.combined, to_file="RetroGAN.png", show_shapes=True)

    def build_generator(self, name):
        def dense(layer_input, nodes, normalization=True):
            """Layers used during downsampling"""
            d = Dense(nodes, activation="relu")(layer_input)
            if normalization:
                d = BatchNormalization()(d)
            return d

        def conv1d(layer_input, filters, f_size=6, strides=1, normalization=True):
            d = Conv1D(filters, f_size, strides=strides, activation="relu")(layer_input)
            return d

        def deconv1d(layer_input, filters, f_size=6, strides=1, normalization=True):
            d = UpSampling1D(filters, f_size, strides=strides, activation="relu")(layer_input)
            return d

        # Image input
        inpt = Input(shape=self.vector_shape)
        # Continue into fc layers
        d0 = dense(inpt, self.gf * 8, normalization=False)
        # d1 = dense(d0,self.gf * 8,normalization=True)
        # d2 = dense(d1, self.gf * 8, normalization=True)
        # d3 = dense(d2, self.gf * 8, normalization=True)
        r = Reshape((-1, 1))(d0)
        # # Downscaling
        # t1 = conv1d(r,self.gf*8,f_size=6)
        # t2 = conv1d(r, self.gf * 4, f_size=8)
        t3 = conv1d(r, self.gf, f_size=4)
        f = MaxPooling1D(pool_size=4)(t3)
        f = Flatten()(f)
        f = Concatenate()([f,self.bert_out])
        # f = Concatenate()([d3,self.bert_out])
        attn = attention(f)  # TODO LOOK AT MULTIPLE ATTENTION POOLS
        attn1 = attention(f)  # TODO LOOK AT MULTIPLE ATTENTION POOLS
        attn2 = attention(f)  # TODO LOOK AT MULTIPLE ATTENTION POOLS
        f = Concatenate()([attn,attn1,attn2])
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
        d4 = dense(f, self.gf * 8, normalization=False)
        output_img = Dense(dimensionality)(d4)
        return Model([inpt,self.bert_in], output_img, name=name)

    def build_discriminator(self, name):
        def d_layer(layer_input, filters, normalization=True):
            """Discriminator layer"""
            d = Dense(filters, activation="relu")(layer_input)
            if normalization:
                d = BatchNormalization()(d)
            return d

        inpt = Input(shape=self.vector_shape)
        d1 = d_layer(inpt, self.df * 32, normalization=False)
        d1 = d_layer(d1, self.df * 16)
        d2 = d_layer(d1, self.df * 8)
        d2_a1 = attention(d2)
        d2_a2 = attention(d2)
        d2_a3 = attention(d2)
        c = Concatenate()([d2_a1,d2_a2,d2_a3])
        d3 = d_layer(c, self.df * 4)
        d4 = d_layer(d3, self.df * 2)
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
        #
        start_time = datetime.datetime.now()
        # for idx, word in enumerate(testwords):
        #     print(word)
        #     retro_representation = rcgan.g_AB.predict(fastext_version[idx].reshape(1, dimensionality))
        #     print(sklearn.metrics.mean_absolute_error(retro_version[idx],
        #                                               retro_representation.reshape((dimensionality,))))

        X_train = Y_train = X_test = Y_test = None
        X_train, Y_train, indexes = load_noisiest_words_dataset(dataset,
                                                       save_folder="fasttext_model/",
                                                       threshold=0.90,
                                                       cache=False,
                                                       return_idx=True)
        print("Done")
        def load_batch(batch_size=2):
            l = X_train.shape[0]
            iterable = list(range(0, l))
            shuffle(iterable)
            for ndx in tqdm(range(0, l, batch_size), ncols=30):
                print("Yielding")
                ixs = iterable[ndx:min(ndx + batch_size, l)]
                original_domain_vecs = X_train[ixs]
                retrofitted_domain_vecs = Y_train[ixs]
                word_idxs = indexes[ixs]
                yield original_domain_vecs, retrofitted_domain_vecs, word_idxs
        for epoch in range(epochs + 1):
            noise = np.random.normal(size=(batch_size, dimensionality), scale=0.01)
            for batch_i, (imgs_A, imgs_B,word_ixs) in enumerate(load_batch(batch_size)):
                print("In batch",batch_i)
                # ----------------------
                #  Train Discriminators
                # ----------------------
                #Add a bit of noise on even iterations.
                if epoch % 2 == 0:
                    imgs_A = np.add(noise[0:imgs_A.shape[0], :], imgs_A)
                bc = BertClient()
                # print(word_ixs)
                bert_embs = bc.encode([x.replace("/c/en/","") for x in word_ixs])

                fake_B = self.g_AB.predict([imgs_A,bert_embs])
                fake_A = self.g_BA.predict([imgs_B,bert_embs])

                valid = np.ones((imgs_A.shape[0],), )
                fake = np.zeros((imgs_A.shape[0],))

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B,bert_embs],
                                                      [valid, valid,
                                                       imgs_A, imgs_B,
                                                       imgs_A, imgs_B])
                elapsed_time = datetime.datetime.now() - start_time

                if np.isnan(dA_loss_fake).any() or np.isnan(dA_loss_real).any() or np.isnan(
                        dB_loss_fake).any() or np.isnan(dB_loss_real).any() or np.isnan(g_loss).any():
                    print(np.isnan(dA_loss_fake).any(), np.isnan(dA_loss_real).any(), np.isnan(dB_loss_fake).any(),
                          np.isnan(dB_loss_real).any(), np.isnan(g_loss).any())
                    print("Problem")
                    raise ArithmeticError("Problem with loss calculation")

                # Plot the progress
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
                if batch_i % 100 == 0:
                    self.save_model()
            try:
                self.save_model()
                # Check at end of epoch!
                for idx, word in enumerate(testwords):
                    print(word)
                    retro_representation = rcgan.g_AB.predict(fastext_version[idx].reshape(1, dimensionality))
                    print(sklearn.metrics.mean_absolute_error(retro_version[idx],
                                                              retro_representation.reshape((dimensionality,))))
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

    def build_bert_input(self):
        bert_dim = (768,)
        context = Input(shape=bert_dim,name="context_vector")
        fc_1 = Dense(512,activation="relu")(context)
        fc_2 = Dense(512, activation="relu")(fc_1)
        self.bert_in = context
        self.bert_out = fc_2



if __name__ == '__main__':
    config = tf.ConfigProto()
    global dimensionality
    dimensionality = 300
    tools.dimensionality = dimensionality
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
    postfix = "ft_bert"
    save_folder = "fasttext_model/trained_retrogan/" + str(datetime.datetime.now()) + postfix
    # save_folder = "fasttext_model/trained_retrogan/2019-07-21 23:12:49.367429ft"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    rcgan = RetroCycleGAN(save_folder=save_folder,generator_size=32,discriminator_size=16)
    ds = {"original": "unfitted.hd5clean",
          "retrofitted": "fitted-debias.hd5clean",
          "directory": "./fasttext_model/"}

    X_train, Y_train, X_test, Y_test = rcgan.train(epochs=50, batch_size=32, sample_interval=100,
                                                   dataset=ds)

    testwords = ["human", "dog", "cat", "potato", "fat"]
    fastext_version = find_in_dataset(testwords, dataset=ds["directory"] + ds["original"])
    print(fastext_version)
    retro_version = find_in_dataset(testwords, dataset=ds["directory"] + ds["retrofitted"])
    print(retro_version)
    trained_model_path = os.path.join(save_folder, "toretrogen.h5")
    to_retro_converter = load_model(trained_model_path,
                                    custom_objects={"ConstMultiplierLayer":ConstMultiplierLayer},
                                    compile=False)
    to_retro_converter.compile(optimizer=Adam(),loss=['mae'])
    to_retro_converter.load_weights(trained_model_path)
    for idx, word in enumerate(testwords):
        print(word)
        retro_representation = to_retro_converter.predict(fastext_version[idx].reshape(1, dimensionality))
        res = tools.find_closest_in_dataset(retro_representation, n_top=20, dataset=ds["directory"] + ds["retrofitted"])
        print(res)
        print(sklearn.metrics.mean_absolute_error(retro_version[idx], retro_representation.reshape((dimensionality,))))
