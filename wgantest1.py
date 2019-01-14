from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

from tools import load_training_input_2


class WGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        # self.img_shape = (self.img_rows, self.img_cols, self.channels)
        word_vec_dim = 300
        self.img_shape = (word_vec_dim,)
        self.latent_dim = 300

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 10
        self.clip_value = 0.0001
        # optimizer = RMSprop(lr=0.005)
        optimizer = Adam()
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

        model.add(Dense(4096, activation="relu", input_dim=self.latent_dim))

        model.add(Dense(2048, activation="relu", input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # model.add(Reshape((7, 7, 128)))
        model.add(Dense(1024))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # model.add(UpSampling2D())
        # model.add(Conv2D(128, kernel_size=4, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))

        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # model.add(Dense(128))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))

        # model.add(UpSampling2D())
        # model.add(Conv2D(64, kernel_size=4, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))

        # model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        # model.add(Activation("tanh"))
        model.add(Dense(300))
        model.add(Activation("tanh"))


        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Dense(256,input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        # model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(Dense(1024))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(Dense(2048))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        # model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        X_train,Y_train, X_test,Y_test = load_training_input_2(1000000)

        # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        noisy_entries_num = 10
        valid = -np.ones((batch_size*noisy_entries_num, 1))
        fake = np.ones((batch_size*noisy_entries_num, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------
                ####
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                noisy_entries = []
                noisy_outputs = []
                for index in idx:
                    # Generate some noise
                    input_noise = output_noise = noise = np.random.normal(0, 1, (noisy_entries_num, self.latent_dim))
                    # Replace one for the original
                    input_noise[0, :] = X_train[index]
                    output_noise[0, :] = Y_train[index]
                    # Add noise to the original to have some noisy inputs
                    for i in range(1, noise.shape[0]):
                        input_noise[i, :] = X_train[index] + noise[i, :]
                        output_noise[i, :] = Y_train[index] + noise[i, :]

                    noisy_entries.append(input_noise)
                    noisy_outputs.append(output_noise)
                # imgs = Y_train[idx]
                imgs = noisy_outputs[0]
                noise =noisy_entries[0]
                # print("imgs")
                # print(imgs.shape)
                # print("noise")
                # print(noise.shape)
                for entry_idx in range(1,len(noisy_outputs)):
                    # print(noisy_outputs[entry_idx].shape)
                    imgs = np.vstack((imgs,noisy_outputs[entry_idx]))
                    noise = np.vstack((noise,noisy_entries[entry_idx]))
                ####
                # Select a random batch of word embeddings
                # idx = np.random.randint(0, X_train.shape[0], batch_size)

                # imgs = Y_train[idx]

                # Sample noise as generator input
                # noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # noise = X_train[idx]
                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # # If at save interval => save generated image samples
            # if epoch % sample_interval == 0:
            #     self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        print("INput sample")
        print(noise)
        print("Output sample")
        print(gen_imgs)
        # # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 1
        #
        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
        #         axs[i, j].axis('off')
        #         cnt += 1
        # fig.savefig("images/mnist_%d.png" % epoch)
        # plt.close()


if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=4000, batch_size=128, sample_interval=50)
