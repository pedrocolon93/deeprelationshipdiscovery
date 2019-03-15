from __future__ import print_function, division

import datetime
import gc
import os
import pickle
import sklearn

import numpy as np
from keras import Sequential
from keras.layers import BatchNormalization, UpSampling2D, Conv2D, Reshape, Activation, Flatten, ZeroPadding2D
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, Nadam, Adadelta, RMSprop, SGD
from sklearn.metrics import mean_squared_error

from tools import load_training_input_2, find_word, find_closest, find_in_fasttext, find_in_retrofitted
import keras.backend as K

class RetroWasserGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 1
        self.img_cols = 300
        self.channels = 1
        self.img_shape = (self.img_cols,)#, self.channels)

        # Configure data loader

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Flatten())
        model.add(Dense(300))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

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
                # if batch_i%5==0:
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
                if batch_i % sample_interval == 0:
                    pass
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
    rcgan.train(epochs=100, batch_size=128, sample_interval=100,batches = 1000)
    # rcgan.combined.load_weights("combined_model")
    # exit()
    # rcgan.g_AB.load_weights("toretro")
    data = pickle.load(open('training_testing.data', 'rb'))
    testwords = ["human","dog","cat","potato","fat"]
    fastext_version = find_in_fasttext(testwords)
    retro_version = find_in_retrofitted(testwords)
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


