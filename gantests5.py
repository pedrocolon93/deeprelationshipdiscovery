from __future__ import print_function, division

import datetime
from random import shuffle

import sklearn

import numpy as np
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from tqdm import tqdm
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from tools import find_in_fasttext, find_in_numberbatch, \
    load_training_input_3

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


class RetroCycleGAN():
    def __init__(self, save_index = "0"):
        # Input shape
        self.img_rows = 1
        self.img_cols = 300
        self.channels = 1
        self.img_shape = (self.img_cols,)#, self.channels)
        self.save_index = save_index

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 3                 # Cycle-consistency loss
        self.lambda_id = 0.3 * self.lambda_cycle    # Identity loss

        lr = 0.000002
        cv = 0.4
        cn = 1
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=Adam(lr, clipvalue=cv,clipnorm=cn,amsgrad=True),
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=Adam(lr, clipvalue=cv,clipnorm=cn,amsgrad=True),
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_AB.summary()
        self.g_BA = self.build_generator()
        self.g_BA.summary()
        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=Adam(lr, clipvalue=cv,clipnorm=cn,amsgrad=True))

        plot_model(self.combined)
    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=6):
            """Layers used during downsampling"""
            # d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = Dense(filters)(layer_input)
            d = LeakyReLU(alpha=0.0002)(d)
            d = BatchNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=7, dropout_rate=0.3):
            """Layers used during upsampling"""
            # u = UpSampling2D(size=2)(layer_input)
            # u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            u = Dense(filters)(layer_input)
            u = LeakyReLU(alpha=0.0002)(u)

            # if dropout_rate:
            #     u = Dropout(dropout_rate)(u)
            u = BatchNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        inpt = Input(shape=self.img_shape)
        # Downsampling
        d0 = conv2d(inpt, self.gf*16)
        d1 = conv2d(d0, self.gf*16)
        d2 = conv2d(d1, self.gf*8)
        d3 = conv2d(d2, self.gf*6)
        d4 = conv2d(d3, self.gf*4)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*6)
        u2 = deconv2d(u1, d2, self.gf*8)
        u3 = deconv2d(u2, d1, self.gf*16)

        output_img = Dense(300)(u3)

        return Model(inpt, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=7, normalization=True):
            """Discriminator layer"""
            d = Dense(filters)(layer_input)
            d = LeakyReLU(alpha=0.0002)(d)
            if normalization:
                d = BatchNormalization()(d)
            # d = Dropout(0.75)(d)
            return d

        inpt = Input(shape=self.img_shape)
        d1 = d_layer(inpt, self.df*8, normalization=False)
        d2 = d_layer(d1, self.df*4)
        d3 = d_layer(d2, self.df*2)
        d4 = d_layer(d3, self.df*1)

        validity = Dense(1)(d4)
        return Model(inpt, validity)

    def load_weights(self):
        try:
            self.g_AB.reset_states()
            self.g_BA.reset_states()
            self.combined.reset_states()
            self.d_B.reset_states()
            self.d_A.reset_states()
            self.d_A.load_weights("fromretrodis")
            self.d_B.load_weights("toretrodis")
            self.g_AB.load_weights("toretro")
            self.g_BA.load_weights("fromretro")
            self.combined.load_weights("combined_model")

        except Exception as e:
            print(e)

    def train(self, epochs, batch_size=1, sample_interval=50,noisy_entries_num=5,batches=900,dataset="fasttext",add_noise=False):
        testwords = ["human", "dog", "cat", "potato", "fat"]
        fastext_version = find_in_fasttext(testwords)
        retro_version = find_in_numberbatch(testwords)

        start_time = datetime.datetime.now()
        # self.load_weights(extension="0")
        self.load_weights()
        for idx, word in enumerate(testwords):
            print(word)
            retro_representation = rcgan.g_AB.predict(fastext_version[idx].reshape(1, 300))
            print(sklearn.metrics.mean_absolute_error(retro_version[idx],
                                                      retro_representation.reshape((300,))))
        # Adversarial loss ground truths
        valid = np.ones((batch_size,),)#*noisy_entries_num,) )
        # valid = np.random.uniform(0.9,1.0,size=(batch_size,))
        fake = np.zeros((batch_size,))#*noisy_entries_num,) )
        # fake = np.random.uniform(0.0,0.1,size=(batch_size,))

        X_train = Y_train = X_test = Y_test = None

        seed = 32
        X_train, Y_train, X_test, Y_test = load_training_input_3(seed=seed,test_split=0.001,dataset=dataset)

        def load_batch(batch_size=2):
            l = X_train.shape[0]
            iterable = list(range(0,l))
            shuffle(iterable)
            for ndx in tqdm(range(0, l, batch_size),ncols=10):
                ixs = iterable[ndx:min(ndx + batch_size, l)]
                imgs_A = X_train[ixs]
                imgs_B = Y_train[ixs]
                yield imgs_A,imgs_B
        for epoch in range(epochs+1):
            for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(batch_size)):
                try:
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
                    try:
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
                        g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                              [valid, valid,
                                                               imgs_A, imgs_B,
                                                               imgs_A, imgs_B])
                        elapsed_time = datetime.datetime.now() - start_time

                        if np.isnan(dA_loss_fake).any() or np.isnan(dA_loss_real).any() or np.isnan(dB_loss_fake).any() or np.isnan(dB_loss_real).any() or np.isnan(g_loss).any():
                            print(np.isnan(dA_loss_fake).any() , np.isnan(dA_loss_real).any() , np.isnan(dB_loss_fake).any() , np.isnan(dB_loss_real).any() , np.isnan(g_loss).any())
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
                        if batch_i%100==0:
                            self.save_model()
                    except:
                        print("Reloading the weights!")
                        self.load_weights()
                except Exception as e:
                    print(e)
                    continue
            try:
                # Check at end of epoch!
                for idx, word in enumerate(testwords):
                    print(word)
                    retro_representation = rcgan.g_AB.predict(fastext_version[idx].reshape(1, 300))
                    print(sklearn.metrics.mean_absolute_error(retro_version[idx],
                                                              retro_representation.reshape((300,))))
                self.save_model()
            except Exception as e:
                print(e)

        self.save_model()
        return X_train, Y_train, X_test, Y_test

    def save_model(self):
        self.d_A.save("fromretrodis",include_optimizer=False)
        self.d_B.save("toretrodis",include_optimizer=False)
        self.g_AB.save("toretrogen", include_optimizer=False)
        self.g_BA.save("fromretrogen", include_optimizer=False)
        self.combined.save("combined_model", include_optimizer=False)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    rcgan = RetroCycleGAN()
    X_train, Y_train, X_test, Y_test = rcgan.train(epochs=50, batch_size=2, sample_interval=1000,dataset='crawl')
    # rcgan.combined.load_weights("combined_model")
    # exit()
    # rcgan.g_AB.load_weights("toretro")
    # data = pickle.load(open('training_testing.data', 'rb'))
    testwords = ["human","dog","cat","potato","fat"]
    fastext_version = find_in_fasttext(testwords)
    print(fastext_version)
    retro_version = find_in_numberbatch(testwords)
    print(retro_version)
    # exit()
    for idx,word in enumerate(testwords):
        print(word)
        retro_representation = rcgan.g_AB.predict(fastext_version[idx].reshape(1, 300))
        print(sklearn.metrics.mean_absolute_error(retro_version[idx], retro_representation.reshape((300,))))
