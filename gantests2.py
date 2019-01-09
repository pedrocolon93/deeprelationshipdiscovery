from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, Adadelta

import matplotlib.pyplot as plt

import sys

import numpy as np

from tools import load_training_input, load_training_input_2


class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 300
        self.img_cols = 1
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 300

        optimizer = Adam(0.0002, amsgrad=True)
        # optimizer = Adadelta()
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        # model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Dense(512,activation='relu',input_dim=self.latent_dim))
        model.add(Dense(256,activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128,activation='relu'))
        model.add(BatchNormalization(momentum=0.8))

        # model.add(Reshape((7, 7, 128)))
        # model.add(UpSampling2D())
        # model.add(Conv2D(128, kernel_size=3, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(UpSampling2D())
        # model.add(Conv2D(64, kernel_size=3, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        # model.add(Flatten(self.latent_dim))
        model.add(Dense(self.latent_dim))
        model.add(Activation("linear"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()
        # model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Dense(64,activation='relu',input_dim=self.latent_dim))
        # model.add(Reshape((7, 7, 128)))
        # model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(Dense(128))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(Dense(256))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        # model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        # img = Input(shape=self.img_shape)
        img = Input(shape=(self.latent_dim,))
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        #
        # # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)
        X_train,Y_train, X_test,Y_test = load_training_input_2(10000000)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images/what I want to generate
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = Y_train[idx]

            # Sample noise and generate a batch of new images
            noise2 = np.random.normal(0, 1, (batch_size, self.latent_dim))
            noise = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            # d_loss_fake_2 = self.discriminator.train_on_batch(noise2,fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                # self.save_imgs(epoch)
                # idx = np.random.randint(0, X_test.shape[0], batch_size)
                # print(self.generator.test_on_batch(X_test[idx],Y_test[idx]))
                abscond_non_retro_string = "abscond 0.0083 0.0106 -0.0268 -0.0179 -0.0123 -0.0072 -0.0058 -0.0372 0.0215 0.0092 -0.0087 -0.0091 -0.0002 -0.0316 -0.0138 -0.0186 0.0189 0.0047 0.0237 0.0043 -0.0229 -0.0097 -0.0151 0.0094 -0.0085 0.0259 -0.0083 0.0093 -0.0311 0.0042 -0.0082 -0.0047 -0.0314 0.0360 -0.0176 -0.0247 -0.0167 0.0047 -0.0126 0.0110 -0.0013 -0.0205 0.0101 -0.0122 0.0060 0.0189 0.0311 0.0070 0.0056 -0.0056 -0.0062 0.0206 -0.0230 -0.0413 -0.0595 -0.0164 -0.0098 -0.0071 -0.0358 -0.0132 0.0216 -0.0071 0.0353 0.0016 -0.0055 -0.0022 0.0198 -0.0473 -0.0023 0.0197 -0.0023 0.0097 0.0426 -0.0248 0.0226 0.0097 0.0219 0.0313 0.0260 0.0198 0.0124 0.0188 -0.0033 0.0273 -0.0140 0.0084 0.0328 0.0050 -0.0147 0.0026 -0.0000 -0.0004 -0.0546 -0.0010 -0.0100 -0.0234 -0.0023 -0.0256 -0.0069 0.0037 0.0184 -0.0153 0.0025 -0.0033 0.0115 -0.0353 -0.0161 -0.0114 0.0079 0.0151 0.0010 0.0172 0.0448 0.0219 -0.0249 -0.0007 0.0135 -0.0117 -0.0040 -0.0212 0.0224 -0.0031 0.0216 -0.0259 0.0281 -0.0152 0.0123 -0.0021 -0.0314 0.0218 0.0015 -0.0040 -0.0079 -0.0083 0.0050 0.0043 -0.0159 0.0147 0.0299 0.0155 0.0092 0.0089 0.0115 -0.0311 0.0300 0.0211 -0.0071 0.0061 -0.0047 -0.0168 0.0086 0.0225 -0.0166 -0.0245 0.0236 -0.0077 -0.0140 -0.0174 -0.0177 0.0032 0.0250 0.0140 -0.0224 0.0173 0.0130 0.0002 0.0274 -0.0072 -0.0231 0.0372 -0.0039 -0.0194 -0.0168 -0.0048 -0.0256 0.0111 0.0283 -0.0138 -0.0065 -0.0137 0.0103 -0.0142 -0.0154 -0.0231 0.0073 -0.0247 0.0107 0.0420 0.0073 0.0143 -0.0061 0.0097 -0.0217 -0.0047 -0.0055 -0.0019 0.0270 0.0021 -0.0064 0.0107 -0.0554 0.0341 -0.0097 0.0129 -0.0084 -0.0033 0.0114 -0.0177 -0.0206 -0.0137 0.0159 -0.0186 0.0458 -0.0003 0.0344 0.0011 -0.0099 0.0272 0.0023 -0.0138 -0.0150 -0.0023 -0.0102 0.0390 -0.0184 -0.0149 -0.0399 0.0448 -0.0124 0.0030 -0.0198 -0.0237 -0.0301 -0.0379 0.0394 0.0074 0.0107 0.0116 0.0025 0.0066 0.0677 0.0044 -0.0065 0.0203 0.0011 0.0007 -0.0023 -0.0304 -0.0320 0.0150 -0.0175 -0.0003 -0.0007 -0.0290 -0.0009 0.0059 0.0029 0.0116 -0.0115 0.0038 -0.0466 0.0101 -0.0172 -0.0422 -0.0049 -0.0273 -0.0213 -0.0297 0.0205 -0.0035 -0.0134 0.0487 -0.0358 -0.0319 -0.0106 -0.0173 0.0521 -0.0056 -0.0125 -0.0032 0.0036 -0.0117 -0.0042 0.0037 0.0135 0.0280 0.0046 -0.0057 -0.0085 0.0146 0.0135 -0.0021 0.0129 0.0088 -0.0240 -0.0376 -0.0146 0.0147 -0.0194 0.0091"
                abscond_retro_string = "abscond 0.1078 0.1119 -0.0016 0.0495 -0.0000 -0.1264 -0.0106 0.1652 -0.0126 0.0014 -0.0152 -0.2043 -0.1176 0.0370 -0.0422 -0.0036 -0.1001 0.1304 -0.1046 0.0412 -0.0869 0.0258 -0.0425 -0.0860 -0.1142 0.0464 -0.0693 -0.0286 -0.0238 -0.0273 -0.1242 0.0355 0.1453 0.0868 0.0621 -0.0667 -0.0268 -0.0045 -0.0517 0.0195 0.1216 0.0675 0.0023 -0.0394 0.0637 0.0011 0.0182 -0.0634 0.0148 0.0288 -0.0035 -0.1871 -0.1273 -0.0948 -0.0088 0.0042 0.0227 -0.0361 -0.0315 0.0192 -0.0185 -0.0205 0.0164 0.0869 0.0870 0.0031 0.0705 -0.0816 0.0203 -0.0322 0.0490 0.0160 0.0381 0.0149 -0.0820 -0.0296 0.0640 -0.0127 -0.0219 -0.0524 0.0362 -0.0185 0.0136 0.0098 0.0577 0.0168 0.0320 -0.0758 -0.0391 -0.0265 -0.0238 -0.1031 -0.0499 -0.0783 -0.0551 -0.0252 -0.0356 -0.0828 -0.0040 0.0604 -0.0173 0.0419 0.0603 -0.0523 -0.0525 -0.0550 -0.0030 -0.0848 -0.0196 0.1151 -0.0197 0.0151 0.0234 -0.0275 0.0093 0.0391 -0.0724 0.0403 0.0170 -0.0186 0.0557 0.0261 0.0008 0.0300 0.0560 -0.0471 -0.0723 -0.0336 -0.0067 -0.0021 -0.0541 0.0661 0.0628 -0.1231 0.0472 0.0382 0.0343 -0.0489 0.0677 -0.0233 -0.0018 -0.0637 -0.0537 0.0262 -0.0072 -0.0641 -0.0023 0.0050 -0.0099 0.0008 -0.1040 0.0255 -0.0774 0.0872 -0.0496 0.1259 0.0727 0.0662 0.0121 0.0274 0.0346 -0.0330 -0.0051 0.0149 -0.0099 0.0617 0.0613 -0.0023 -0.0644 -0.0512 0.0439 0.0105 0.0264 0.0183 -0.0171 0.0141 -0.0330 -0.1142 -0.0598 -0.0745 0.0036 -0.0196 0.0385 -0.1153 -0.0148 0.0290 -0.0159 -0.0546 -0.0229 -0.0159 -0.0613 0.0418 -0.0169 0.0727 0.0739 -0.0360 -0.0401 -0.0260 -0.0012 -0.0452 0.0135 0.1019 0.1254 0.0591 0.0156 0.1324 -0.0012 0.0179 -0.0791 -0.0545 0.0060 0.0070 0.0315 -0.0158 -0.0333 -0.0537 0.1155 0.0209 -0.0414 0.0919 -0.0663 0.0434 0.0534 -0.0324 0.0347 -0.0319 -0.0836 0.0196 -0.0068 0.0489 0.0111 -0.0402 -0.0150 0.0525 0.0733 -0.0244 0.0272 0.0421 0.0245 -0.0382 -0.0431 -0.0394 -0.1068 0.0013 0.0754 -0.0197 -0.0160 0.0295 0.0201 -0.1482 -0.1019 0.0268 -0.0359 -0.0386 0.0520 -0.0556 -0.0306 0.0504 -0.0563 -0.0453 0.0187 -0.0465 0.0508 -0.0409 -0.0045 0.0389 0.0047 -0.0844 -0.0316 0.0251 0.0191 0.0010 -0.0509 0.0382 -0.0021 -0.0544 0.0041 0.0867 -0.0691 0.1239 -0.0463 0.0282 -0.0628 0.0969 -0.0607 0.0281 0.0016 0.0848 0.0033 0.0248 -0.0417 0.0300 0.0178 0.0124 0.0254 -0.0082 0.0339 -0.0189 -0.0541 0.0553"
                y = np.array([float(x) for x in abscond_retro_string.split(" ")[1:]])
                x = [[np.array([float(x) for x in abscond_non_retro_string.split(" ")[1:]]).transpose()]]
                print(y)
                # retrofit_dcgan = pickle.load(open("model_save_final.pickle",'rb'))
                # print(list(common_vocabulary).index("abscond"))
                pred_y = self.generator.predict(x)[0]
                print(pred_y)
                pass

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=10000, batch_size=64, save_interval=500)