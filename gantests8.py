from __future__ import print_function, division

import datetime
import gc
import os
import pickle
import sklearn

import numpy as np
from keras.layers import BatchNormalization, UpSampling2D, Conv2D, Reshape, Flatten, UpSampling1D, Conv1D
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, Nadam, Adadelta, RMSprop, SGD
from sklearn.metrics import mean_squared_error

from tools import load_training_input_2, find_word, find_closest, find_in_fasttext, find_in_numberbatch, \
    load_training_input_3


class RetroCycleGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 1
        self.img_cols = 300
        self.channels = 1
        self.img_shape = (self.img_cols,)#, self.channels)

        # Configure data loader


        # Calculate output shape of D (PatchGAN)
        # patch = int(self.img_rows / 2**4)
        # self.disc_patch = (patch, patch, 1)

        # # Number of filters in the first layer of G and D
        # self.gf = 16
        # self.df = 32
        #
        # # Loss weights
        # self.lambda_cycle = 0.1000                   # Cycle-consistency loss
        # self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss
        #
        # optimizer = Adam(0.0002,amsgrad=True)
        # # optimizer = Adam()
        # # optimizer = Nadam()
        # # optimizer = SGD(lr=0.000001,nesterov=True,momentum=0.8,decay=0.1e-8)
        # # optimizer = Adadelta()
        # # optimizer = RMSprop(lr=0.0001)
        # # Build and compile the discriminators
        # self.d_A = self.build_discriminator()
        # self.d_B = self.build_discriminator()
        # self.d_A.compile(loss='mse',
        #     optimizer=optimizer,
        #     metrics=['accuracy'])
        # self.d_B.compile(loss='mse',
        #     optimizer=optimizer,
        #     metrics=['accuracy'])
        #
        # #-------------------------
        # # Construct Computational
        # #   Graph of Generators
        # #-------------------------
        #
        # # Build the generators
        # self.g_AB = self.build_generator()
        # self.g_AB.summary()
        # self.g_BA = self.build_generator()
        # self.g_BA.summary()
        # # Input images from both domains
        # img_A = Input(shape=self.img_shape)
        # img_B = Input(shape=self.img_shape)
        #
        # # Translate images to the other domain
        # fake_B = self.g_AB(img_A)
        # fake_A = self.g_BA(img_B)
        # # Translate images back to original domain
        # reconstr_A = self.g_BA(fake_B)
        # reconstr_B = self.g_AB(fake_A)
        # # Identity mapping of images
        # img_A_id = self.g_BA(img_A)
        # img_B_id = self.g_AB(img_B)
        #
        # # For the combined model we will only train the generators
        # self.d_A.trainable = False
        # self.d_B.trainable = False
        #
        # # Discriminators determines validity of translated images
        # valid_A = self.d_A(fake_A)
        # valid_B = self.d_B(fake_B)
        #
        # # Combined model trains generators to fool discriminators
        # self.combined = Model(inputs=[img_A, img_B],
        #                       outputs=[ valid_A, valid_B,
        #                                 reconstr_A, reconstr_B,
        #                                 img_A_id, img_B_id ])
        # self.combined.compile(loss=['mse', 'mse',
        #                             'mae', 'mae',
        #                             'mae', 'mae'],
        #                     loss_weights=[  1, 1,
        #                                     self.lambda_cycle, self.lambda_cycle,
        #                                     self.lambda_id, self.lambda_id ],
        #                     optimizer=optimizer)

        # Calculate output shape of D (PatchGAN)
        # patch = int(self.img_rows / 2 ** 4)
        # self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002,decay=5e-7,amsgrad=True)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.d_B.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Objectives
        # + Adversarial: Fool domain discriminators
        # + Translation: Minimize MAE between e.g. fake B and true B
        # + Cycle-consistency: Minimize MAE between reconstructed images and original
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       fake_B, fake_A,
                                       reconstr_A, reconstr_B])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              optimizer=optimizer)
    # def build_generator(self):
    #     """U-Net Generator"""
    #
    #     def dense(layer_input, layer_size):
    #         """Layers used during downsampling"""
    #         d = Dense(layer_size)(layer_input)
    #         d = LeakyReLU(alpha=0.2)(d)
    #         dropout_rate = 0.5
    #         if dropout_rate:
    #             d = Dropout(dropout_rate)(d)
    #         # d = BatchNormalization()(d)
    #
    #         return d
    #
    #     def undense(layer_input, skip_input, layer_size,  dropout_rate=0.5):
    #         """Layers used during upsampling"""
    #         # u = UpSampling2D(size=2)(layer_input)
    #         # u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    #         u = Dense(layer_size)(layer_input)
    #         u = LeakyReLU(alpha=0.2)(u)
    #         if dropout_rate:
    #             u = Dropout(dropout_rate)(u)
    #         # u = BatchNormalization()(u)
    #         # u = InstanceNormalization()(u)
    #         u = Concatenate()([u, skip_input])
    #         return u
    #
    #     # Image input
    #     d0 = Input(shape=self.img_shape)
    #
    #     # Downsampling
    #     d1 = dense(d0, self.gf*8)
    #     d2 = dense(d1, self.gf*4)
    #     d3 = dense(d2, self.gf*2)
    #
    #     # Between mapping
    #     d4 = dense(d3, self.gf*2)
    #     d4 = dense(d4, self.gf*2)
    #     d4 = dense(d4, self.gf*2)
    #
    #     # Upsampling
    #     u1 = dense(d4, self.gf*2)
    #     u2 = dense(u1, self.gf*4)
    #     u3 = dense(u2, self.gf*8)
    #
    #     output_img = Dense(self.img_cols)(u3)
    #     return Model(d0, output_img)
    #
    # def build_discriminator(self):
    #
    #     def d_layer(layer_input, layer_size,  normalization=True):
    #         """Discriminator layer"""
    #         d = Dense(layer_size)(layer_input)
    #         d = LeakyReLU(alpha=0.002)(d)
    #         if normalization:
    #             d = BatchNormalization()(d)
    #         # d = Dropout(0.75)(d)
    #
    #         return d
    #
    #     img = Input(shape=self.img_shape)
    #
    #     d1 = d_layer(img, self.df*8, normalization=False)
    #     d2 = d_layer(d1, self.df*6)
    #     d3 = d_layer(d2, self.df*4)
    #     d3 = d_layer(d3, self.df*4)
    #     d3 = d_layer(d3, self.df*4)
    #     d4 = d_layer(d3, self.df)
    #
    #     validity = Dense(1,activation='sigmoid')(d4)
    #
    #     return Model(img, validity)

    # def build_generator(self):
    #     """U-Net Generator"""
    #
    #     def conv2d(layer_input, filters, f_size=6):
    #         """Layers used during downsampling"""
    #         d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    #         d = LeakyReLU(alpha=0.002)(d)
    #         d = BatchNormalization()(d)
    #         return d
    #
    #     def deconv2d(layer_input, skip_input, filters, f_size=7, dropout_rate=0.3):
    #         """Layers used during upsampling"""
    #         u = UpSampling2D(size=2)(layer_input)
    #         u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    #         if dropout_rate:
    #             u = Dropout(dropout_rate)(u)
    #         u = BatchNormalization()(u)
    #         u = Concatenate()([u, skip_input])
    #         return u
    #
    #     # Image input
    #     inpt = Input(shape=self.img_shape)
    #     d0 = Dense(512)(inpt)
    #     d0 = Reshape(target_shape=(16,32,1))(d0)
    #     # Downsampling
    #     d1 = conv2d(d0, self.gf)
    #     d2 = conv2d(d1, self.gf*2)
    #     d3 = conv2d(d2, self.gf*4)
    #     d4 = conv2d(d3, self.gf*8)
    #
    #     # Upsampling
    #     u1 = deconv2d(d4, d3, self.gf*4)
    #     u2 = deconv2d(u1, d2, self.gf*2)
    #     u3 = deconv2d(u2, d1, self.gf)
    #
    #     u4 = UpSampling2D(size=2)(u3)
    #     output_img = Flatten()(u4)
    #     output_img = Dense(300)(output_img)
    #     # output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
    #
    #     return Model(inpt, output_img)
    #
    # def build_discriminator(self):
    #
    #     def d_layer(layer_input, filters, f_size=7, normalization=True):
    #         """Discriminator layer"""
    #         d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    #         d = LeakyReLU(alpha=0.002)(d)
    #         if normalization:
    #             d = BatchNormalization()(d)
    #         d = Dropout(0.75)(d)
    #
    #         return d
    #
    #     inpt = Input(shape=self.img_shape)
    #     img = Dense(512)(inpt)
    #     img = Reshape(target_shape=(16,32,1))(img)
    #     d1 = d_layer(img, self.df, normalization=False)
    #     d2 = d_layer(d1, self.df*2)
    #     d3 = d_layer(d2, self.df*4)
    #     d4 = d_layer(d3, self.df*8)
    #
    #     validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
    #     validity = Flatten()(validity)
    #     validity = Dense(1)(validity)
    #     return Model(inpt, validity)


    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, normalize=True):
            """Layers used during downsampling"""
            # d = Conv1D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = Dense(filters)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalize:
                d = BatchNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling1D(size=2)(layer_input)
            # u = UpSampling2D(size=2)(layer_input)
            # u = Conv1D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            u = Dense(filters)(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, normalize=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling1D(size=2)(u6)
        # output_img = Conv1D(self.img_cols, kernel_size=4, strides=1,
        #                     padding='same', activation='tanh')(u7)
        output_img = Dense(self.img_cols)(u7)
        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            # d = Conv1D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = Dense(filters)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = BatchNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)
        validity = Dense(1)(d4)
        # validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    # def train(self, epochs, batch_size=1, sample_interval=50,noisy_entries_num=5,batches=900):
    #
    #     start_time = datetime.datetime.now()
    #
    #     # Adversarial loss ground truths
    #     valid = np.ones((batch_size,))#*noisy_entries_num,) )
    #     fake = np.zeros((batch_size,))#*noisy_entries_num,) )
    #
    #     X_train = Y_train = X_test = Y_test = None
    #
    #     regen = False
    #     normalize = False
    #     seed = 32
    #     file = "data.pickle"
    #     if not os.path.exists(file) or regen:
    #         X_train, Y_train, X_test, Y_test = load_training_input_2(normalize=normalize, seed=seed,test_split=0.1)
    #         pickle.dump((X_train, Y_train, X_test, Y_test), open(file, "wb"))
    #     else:
    #         X_train, Y_train, X_test, Y_test = pickle.load(open("data.pickle", 'rb'))
    #     n_batches = batches
    #     def load_batch(batch_size,train_test = True,n_batches = 900):
    #         for i in range(n_batches):
    #             if train_test:
    #                 idx = np.random.randint(0, X_train.shape[0], batch_size)
    #                 imgs_A = X_train[idx]
    #                 imgs_B = Y_train[idx]
    #                 yield imgs_A, imgs_B
    #             else:
    #                 idx = np.random.randint(0, X_test.shape[0], batch_size)
    #                 imgs_A = X_test[idx]
    #                 imgs_B = Y_test[idx]
    #                 yield imgs_A, imgs_B
    #     for epoch in range(epochs+1):
    #         for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(batch_size,n_batches=batches)):
    #
    #             # ----------------------
    #             #  Train Discriminators
    #             # ----------------------
    #
    #             self.latent_dim = 300
    #             idx = np.random.randint(0, X_train.shape[0], batch_size)
    #             noise = X_train[idx]
    #             imgs = Y_train[idx]
    #             if batch_i%2==0 or batch_i%3==0:
    #                 normnoise = np.random.normal(size=(batch_size,300))
    #                 noise = np.add(noise,normnoise)
    #                 imgs = np.add(imgs,normnoise)
    #
    #             imgs_A = noise
    #             imgs_B =imgs
    #
    #
    #             # Translate images to opposite domain
    #             fake_B = self.g_AB.predict(imgs_A)
    #             fake_A = self.g_BA.predict(imgs_B)
    #             if batch_i%5==0:
    #                 dA_loss_real = self.d_A.train_on_batch(imgs_A, fake)
    #                 dA_loss_fake = self.d_A.train_on_batch(fake_A, valid)
    #                 dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
    #
    #                 dB_loss_real = self.d_B.train_on_batch(imgs_B, fake)
    #                 dB_loss_fake = self.d_B.train_on_batch(fake_B, valid)
    #                 dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
    #             else:
    #             # Train the discriminators (original images = real / translated = Fake)
    #                 dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
    #                 dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
    #                 dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
    #
    #                 dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
    #                 dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
    #                 dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
    #
    #             # Total disciminator loss
    #             d_loss = 0.5 * np.add(dA_loss, dB_loss)
    #
    #             # ------------------
    #             #  Train Generators
    #             # ------------------
    #
    #             # Train the generators
    #             g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
    #                                                     [valid, valid,
    #                                                     imgs_A, imgs_B,
    #                                                     imgs_A, imgs_B])
    #
    #             elapsed_time = datetime.datetime.now() - start_time
    #
    #             # Plot the progress
    #             print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
    #                                                                     % ( epoch, epochs,
    #                                                                         batch_i, n_batches,
    #                                                                         d_loss[0], 100*d_loss[1],
    #                                                                         g_loss[0],
    #                                                                         np.mean(g_loss[1:3]),
    #                                                                         np.mean(g_loss[3:5]),
    #                                                                         np.mean(g_loss[5:6]),
    #                                                                         elapsed_time))
    #         self.g_AB.save("toretro")
    #         self.g_BA.save("fromretro")
    #         self.combined.save("combined_model")
    #
    #         # Check at end of epoch!
    #         testwords = ["human", "dog", "cat", "potato", "fat"]
    #         fastext_version = find_in_fasttext(testwords)
    #         retro_version = find_in_numberbatch(testwords)
    #         for idx, word in enumerate(testwords):
    #             print(word)
    #             retro_representation = rcgan.g_AB.predict(fastext_version[idx].reshape(1, 300))
    #             print(sklearn.metrics.mean_absolute_error(retro_version[idx],
    #                                                       retro_representation.reshape((300,))))
    def train(self, epochs, batch_size=128, sample_interval=50,batches=1000):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,))
        fake = np.zeros((batch_size,))

        # for epoch in range(epochs):
        #
        #     for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
        X_train = Y_train = X_test = Y_test = None

        regen = True
        normalize = False
        seed = 32
        file = "data.pickle"
        if not os.path.exists(file) or regen:
            X_train, Y_train, X_test, Y_test = load_training_input_3(seed=seed,test_split=0.01)
            pickle.dump((X_train, Y_train, X_test, Y_test), open(file, "wb"))
        else:
            X_train, Y_train, X_test, Y_test = pickle.load(open("data.pickle", 'rb'))
        n_batches = batches
        def load_batch(batch_size,train_test = True,n_batches = 900):
            for i in range(n_batches):
                if train_test:
                    idx = np.random.randint(0, X_train.shape[0], batch_size)
                    imgs_A = X_train[idx]
                    imgs_B = Y_train[idx]
                    yield imgs_A, imgs_B
                else:
                    idx = np.random.randint(0, X_test.shape[0], batch_size)
                    imgs_A = X_test[idx]
                    imgs_B = Y_test[idx]
                    yield imgs_A, imgs_B
        for epoch in range(epochs+1):
            for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(batch_size,n_batches=batches)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

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
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, \
                                                                         imgs_B, imgs_A, \
                                                                         imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[%d] [%d/%d] time: %s, [d_loss: %f, g_loss: %f]" % (epoch, batch_i,
                                                                           batches,
                                                                           elapsed_time,
                                                                           d_loss[0], g_loss[0]))

                # If at save interval => save generated image samples
            self.g_AB.save("toretro")
            self.g_BA.save("fromretro")
            self.combined.save("combined_model")

            # Check at end of epoch!
            testwords = ["human", "dog", "cat", "potato", "fat"]
            fastext_version = find_in_fasttext(testwords)
            retro_version = find_in_numberbatch(testwords)
            for idx, word in enumerate(testwords):
                print(word)
                retro_representation = rcgan.g_AB.predict(fastext_version[idx].reshape(1, 300))
                print(sklearn.metrics.mean_absolute_error(retro_version[idx],
                                                          retro_representation.reshape((300,))))

        self.g_AB.save("toretro")
        self.g_BA.save("fromretro")
        self.combined.save("combined_model")


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session



if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    rcgan = RetroCycleGAN()
    rcgan.train(epochs=100, batch_size=128, sample_interval=100,batches = 1000)
    # rcgan.combined.load_weights("combined_model")
    # exit()
    # rcgan.g_AB.load_weights("toretro")
    data = pickle.load(open('training_testing.data', 'rb'))
    testwords = ["human","dog","cat","potato","fat"]
    fastext_version = find_in_fasttext(testwords)
    retro_version = find_in_numberbatch(testwords)
    for idx,word in enumerate(testwords):
        print(word)
        retro_representation = rcgan.g_AB.predict(fastext_version[idx].reshape(1, 300))
        print(sklearn.metrics.mean_absolute_error(retro_version[idx], retro_representation.reshape((300,))))
        # find_closest(retro_representation)
    # word_count = 1
    # for i in range(10, 20, word_count):
    #     find_word(data["X_test"][i, :], retro=False)
    #     gc.collect()
    #     # input_rep = []
    #     # rep_amount = 10
    #     # for i in range(rep_amount):
    #     #     input_rep.append(input_vae.encoder.predict(data["X_test"][i, :].reshape(1, 300))[2])
    #     # input_representation = np.mean(input_rep, axis=0)
    #     # rcgan.g_AB = load_model("toretro")
    #     retro_representation = rcgan.g_AB.predict(data["X_test"][i, :].reshape(1, 300))
    #     # rcgan.g_BA = load_model("fromretro")
    #     # output_rep = []
    #     # for i in range(rep_amount):
    #     #     output_rep.append(output_vae.decoder.predict(retro_representation))
    #     #     # output_rep.append(rcgan.g_BA.predict(retro_representation))
    #     # # rcgan.combined=load_model("combined_model")
    #     # output_ = np.mean(output_rep, axis=0)
    #     find_closest(retro_representation)
    #     gc.collect()


