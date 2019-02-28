from __future__ import print_function, division

import datetime
import gc
import os
import pickle
import sklearn

import numpy as np
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, Nadam, Adadelta, RMSprop, SGD
from sklearn.metrics import mean_squared_error

from tools import load_training_input_2, find_word, find_closest, find_in_fasttext, find_in_numberbatch


class RetroPixGAN():
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

        # Number of filters in the first layer of G and D
        # self.gf = 32
        # self.df = 64

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002,decay=1e-7,amsgrad=True)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100000],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Dense(filters)(layer_input)
            d = LeakyReLU(alpha=0.0002)(d)
            # if bn:
            #     d = BatchNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            # u = UpSampling2D(size=2)(layer_input)
            u = Dense(filters, activation='relu')(layer_input)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            # u = BatchNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf * 8)
        # d3 = conv2d(d2, self.gf * 8)
        # d4 = conv2d(d3, self.gf * 8)
        # d5 = conv2d(d4, self.gf * 8)
        # d6 = conv2d(d5, self.gf * 4)
        # d7 = conv2d(d6, self.gf * 2)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf * 2)
        u2 = deconv2d(u1, d5, self.gf * 4)
        u3 = deconv2d(u2, d4, self.gf * 8)
        u4 = deconv2d(u3, d3, self.gf * 8)
        u5 = deconv2d(u4, d2, self.gf * 8)
        u6 = deconv2d(u5, d1, self.gf)

        # u7 = UpSampling2D(size=2)(u6)
        output_img = Dense(300)(u6)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Dense(filters)(layer_input)
            d = LeakyReLU(alpha=0.0002)(d)
            # if bn:
            #     d = BatchNormalization()(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 8)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 2)

        validity = Dense(1)(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50, batches=200):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) )
        # valid = np.random.normal(size=(batch_size,),loc=1,scale=0.1)
        fake = np.zeros((batch_size,) )
        # fake = np.random.normal(size=(batch_size,),loc=0,scale=0.1)
        X_train = Y_train = X_test = Y_test = None

        regen = False
        normalize = False
        seed = 32
        file = "data.pickle"
        if not os.path.exists(file) or regen:
            X_train, Y_train, X_test, Y_test = load_training_input_2(normalize=normalize, seed=seed, test_split=0.1)
            pickle.dump((X_train, Y_train, X_test, Y_test), open(file, "wb"))
        else:
            X_train, Y_train, X_test, Y_test = pickle.load(open("data.pickle", 'rb'))
            # data = {
            #     'X_train':X_train,
            #            'Y_train':Y_train, 'X_test':X_test, 'Y_test':Y_test
            # }
            # pickle.dump(data,open('training_testing.data','wb'))
        n_batches = batches

        def load_batch(batch_size, train_test=True, n_batches=900):
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

        for epoch in range(epochs + 1):
            for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(batch_size, n_batches=batches)):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                #ADD SOME NOISE TO STABILIZE
                if batch_i%2==0:
                    imgs_B = np.add(imgs_B,np.random.normal(size=(batch_size,300)))

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                if batch_i%10==0:
                    d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], fake)
                    d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], valid)
                else:
                    d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                    d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                                                      batch_i,
                                                                                                      batches,
                                                                                                      d_loss[0],
                                                                                                      100 * d_loss[1],
                                                                                                      g_loss[0],
                                                                                                      elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0 and not batch_i ==0:
                    testwords = ["human", "dog", "cat", "potato", "fat"]
                    fastext_version = find_in_fasttext(testwords)
                    retro_version = find_in_numberbatch(testwords)
                    for idx, word in enumerate(testwords):
                        print(word)
                        retro_representation = self.generator.predict(fastext_version[idx].reshape(1, 300))
                        print(sklearn.metrics.mean_absolute_error(retro_version[idx],
                                                                  retro_representation.reshape((300,))))

                    # self.sample_images(epoch, batch_i)
        self.generator.save("toretro")
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

    rcgan = RetroPixGAN()
    rcgan.train(epochs=2, batch_size=256, sample_interval=500,batches = 1000)
    # rcgan.combined.load_weights("combined_model")
    # exit()
    # rcgan.g_AB.load_weights("toretro")
    data = pickle.load(open('training_testing.data', 'rb'))
    testwords = ["human","dog","cat","potato","fat"]
    fastext_version = find_in_fasttext(testwords)
    retro_version = find_in_numberbatch(testwords)
    for idx,word in enumerate(testwords):
        print(word)
        retro_representation = rcgan.generator.predict(fastext_version[idx].reshape(1, 300))
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


