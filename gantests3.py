from __future__ import print_function, division

import datetime
import os
import pickle

import numpy as np
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, Nadam, Adadelta, RMSprop, SGD
from sklearn.metrics import mean_squared_error

from tools import load_training_input_2
from gantests4 import find_word, find_closest


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

        # Number of filters in the first layer of G and D
        self.gf = 256
        self.df = 512

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        # optimizer = Adam(0.0002, 0.5,amsgrad=True)
        # optimizer = Adam()
        # optimizer = Nadam()
        # optimizer = SGD(lr=0.01,nesterov=True,momentum=0.8,decay=0.1e-8)
        # optimizer = Adadelta()
        optimizer = RMSprop(lr=0.005)
        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
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
                            optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def dense(layer_input, layer_size):
            """Layers used during downsampling"""
            # d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            # d = LeakyReLU(alpha=0.2)(d)
            # d = InstanceNormalization()(d)
            d = Dense(layer_size)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = BatchNormalization()(d)
            return d

        def undense(layer_input, skip_input, layer_size,  dropout_rate=0):
            """Layers used during upsampling"""
            # u = UpSampling2D(size=2)(layer_input)
            # u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            u = Dense(layer_size,activation='relu')(layer_input)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization()(u)
            # u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = dense(d0, self.gf)
        d2 = dense(d1, self.gf*2)
        d3 = dense(d2, self.gf*4)
        d4 = dense(d3, self.gf*8)

        # Upsampling
        u1 = undense(d4, d3, self.gf*4)
        u2 = undense(u1, d2, self.gf*2)
        u3 = undense(u2, d1, self.gf)

        # u4 = UpSampling2D(size=2)(u3)
        # u4 = Dense(2048)
        # output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
        output_img = Dense(self.img_cols)(u3)
        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, layer_size,  normalization=True):
            """Discriminator layer"""
            d = Dense(layer_size, activation='relu')(layer_input)
            # d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = BatchNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Dense(1,activation='sigmoid')(d4)

        return Model(img, validity)

    def train(self, epochs, batch_size=1, sample_interval=50,noisy_entries_num=5):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size*noisy_entries_num,) )
        fake = np.zeros((batch_size*noisy_entries_num,) )
        # X_train, Y_train, X_test, Y_test = (None,None,None,None)
        # if os.path.exists("training_testing.data"):
        #     print("Loading data")
        #     data = pickle.load(open('training_testing.data','rb'))
        #     X_train = data["X_train"]
        #     Y_train = data["Y_train"]
        #     X_test = data["X_test"]
        #     Y_test = data["Y_test"]
        # else:
            # print("Dumping data")
        X_train = Y_train = X_test = Y_test = None
        file = "data.pickle"
        if not os.path.exists(file):
            X_train, Y_train, X_test, Y_test = load_training_input_2()
            pickle.dump((X_train, Y_train, X_test, Y_test), open(file, "wb"))
        else:
            X_train, Y_train, X_test, Y_test = pickle.load(open("data.pickle", 'rb'))
            # data = {
            #     'X_train':X_train,
            #            'Y_train':Y_train, 'X_test':X_test, 'Y_test':Y_test
            # }
            # pickle.dump(data,open('training_testing.data','wb'))
        n_batches = 900
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
        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                self.latent_dim = 300
                # idx = np.random.randint(0, X_train.shape[0], batch_size)
                noisy_entries = []
                noisy_outputs = []
                for index in range(len(imgs_A)):
                    # Generate some noise
                    input_noise = output_noise = noise = np.random.normal(0, 0.1, (noisy_entries_num, self.latent_dim))
                    # Replace one for the original
                    input_noise[0, :] = imgs_A[index]
                    output_noise[0, :] = imgs_B[index]
                    # Add noise to the original to have some noisy inputs
                    for i in range(1, noise.shape[0]):
                        input_noise[i, :] = imgs_A[index] + noise[i, :]
                        output_noise[i, :] = imgs_B[index] + noise[i, :]
                    noisy_entries.append(input_noise)
                    noisy_outputs.append(output_noise)
                # imgs = Y_train[idx]
                imgs = noisy_outputs[0]
                noise = noisy_entries[0]
                # print("imgs")
                # print(imgs.shape)
                # print("noise")
                # print(noise.shape)
                for entry_idx in range(1, len(noisy_outputs)):
                    # print(noisy_outputs[entry_idx].shape)
                    imgs = np.vstack((imgs, noisy_outputs[entry_idx]))
                    noise = np.vstack((noise, noisy_entries[entry_idx]))
                imgs_A = noise
                imgs_B =imgs


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

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

            if batch_i % sample_interval == 0:
                self.g_AB.save("toretro")
                self.g_BA.save("fromretro")
                self.combined.save("combined_model")


if __name__ == '__main__':
    gan = RetroCycleGAN()
    gan.train(epochs=8, batch_size=128, sample_interval=100)
    # abscond_non_retro_string = "abscond 0.0083 0.0106 -0.0268 -0.0179 -0.0123 -0.0072 -0.0058 -0.0372 0.0215 0.0092 -0.0087 -0.0091 -0.0002 -0.0316 -0.0138 -0.0186 0.0189 0.0047 0.0237 0.0043 -0.0229 -0.0097 -0.0151 0.0094 -0.0085 0.0259 -0.0083 0.0093 -0.0311 0.0042 -0.0082 -0.0047 -0.0314 0.0360 -0.0176 -0.0247 -0.0167 0.0047 -0.0126 0.0110 -0.0013 -0.0205 0.0101 -0.0122 0.0060 0.0189 0.0311 0.0070 0.0056 -0.0056 -0.0062 0.0206 -0.0230 -0.0413 -0.0595 -0.0164 -0.0098 -0.0071 -0.0358 -0.0132 0.0216 -0.0071 0.0353 0.0016 -0.0055 -0.0022 0.0198 -0.0473 -0.0023 0.0197 -0.0023 0.0097 0.0426 -0.0248 0.0226 0.0097 0.0219 0.0313 0.0260 0.0198 0.0124 0.0188 -0.0033 0.0273 -0.0140 0.0084 0.0328 0.0050 -0.0147 0.0026 -0.0000 -0.0004 -0.0546 -0.0010 -0.0100 -0.0234 -0.0023 -0.0256 -0.0069 0.0037 0.0184 -0.0153 0.0025 -0.0033 0.0115 -0.0353 -0.0161 -0.0114 0.0079 0.0151 0.0010 0.0172 0.0448 0.0219 -0.0249 -0.0007 0.0135 -0.0117 -0.0040 -0.0212 0.0224 -0.0031 0.0216 -0.0259 0.0281 -0.0152 0.0123 -0.0021 -0.0314 0.0218 0.0015 -0.0040 -0.0079 -0.0083 0.0050 0.0043 -0.0159 0.0147 0.0299 0.0155 0.0092 0.0089 0.0115 -0.0311 0.0300 0.0211 -0.0071 0.0061 -0.0047 -0.0168 0.0086 0.0225 -0.0166 -0.0245 0.0236 -0.0077 -0.0140 -0.0174 -0.0177 0.0032 0.0250 0.0140 -0.0224 0.0173 0.0130 0.0002 0.0274 -0.0072 -0.0231 0.0372 -0.0039 -0.0194 -0.0168 -0.0048 -0.0256 0.0111 0.0283 -0.0138 -0.0065 -0.0137 0.0103 -0.0142 -0.0154 -0.0231 0.0073 -0.0247 0.0107 0.0420 0.0073 0.0143 -0.0061 0.0097 -0.0217 -0.0047 -0.0055 -0.0019 0.0270 0.0021 -0.0064 0.0107 -0.0554 0.0341 -0.0097 0.0129 -0.0084 -0.0033 0.0114 -0.0177 -0.0206 -0.0137 0.0159 -0.0186 0.0458 -0.0003 0.0344 0.0011 -0.0099 0.0272 0.0023 -0.0138 -0.0150 -0.0023 -0.0102 0.0390 -0.0184 -0.0149 -0.0399 0.0448 -0.0124 0.0030 -0.0198 -0.0237 -0.0301 -0.0379 0.0394 0.0074 0.0107 0.0116 0.0025 0.0066 0.0677 0.0044 -0.0065 0.0203 0.0011 0.0007 -0.0023 -0.0304 -0.0320 0.0150 -0.0175 -0.0003 -0.0007 -0.0290 -0.0009 0.0059 0.0029 0.0116 -0.0115 0.0038 -0.0466 0.0101 -0.0172 -0.0422 -0.0049 -0.0273 -0.0213 -0.0297 0.0205 -0.0035 -0.0134 0.0487 -0.0358 -0.0319 -0.0106 -0.0173 0.0521 -0.0056 -0.0125 -0.0032 0.0036 -0.0117 -0.0042 0.0037 0.0135 0.0280 0.0046 -0.0057 -0.0085 0.0146 0.0135 -0.0021 0.0129 0.0088 -0.0240 -0.0376 -0.0146 0.0147 -0.0194 0.0091"
    # abscond_retro_string = "abscond 0.1078 0.1119 -0.0016 0.0495 -0.0000 -0.1264 -0.0106 0.1652 -0.0126 0.0014 -0.0152 -0.2043 -0.1176 0.0370 -0.0422 -0.0036 -0.1001 0.1304 -0.1046 0.0412 -0.0869 0.0258 -0.0425 -0.0860 -0.1142 0.0464 -0.0693 -0.0286 -0.0238 -0.0273 -0.1242 0.0355 0.1453 0.0868 0.0621 -0.0667 -0.0268 -0.0045 -0.0517 0.0195 0.1216 0.0675 0.0023 -0.0394 0.0637 0.0011 0.0182 -0.0634 0.0148 0.0288 -0.0035 -0.1871 -0.1273 -0.0948 -0.0088 0.0042 0.0227 -0.0361 -0.0315 0.0192 -0.0185 -0.0205 0.0164 0.0869 0.0870 0.0031 0.0705 -0.0816 0.0203 -0.0322 0.0490 0.0160 0.0381 0.0149 -0.0820 -0.0296 0.0640 -0.0127 -0.0219 -0.0524 0.0362 -0.0185 0.0136 0.0098 0.0577 0.0168 0.0320 -0.0758 -0.0391 -0.0265 -0.0238 -0.1031 -0.0499 -0.0783 -0.0551 -0.0252 -0.0356 -0.0828 -0.0040 0.0604 -0.0173 0.0419 0.0603 -0.0523 -0.0525 -0.0550 -0.0030 -0.0848 -0.0196 0.1151 -0.0197 0.0151 0.0234 -0.0275 0.0093 0.0391 -0.0724 0.0403 0.0170 -0.0186 0.0557 0.0261 0.0008 0.0300 0.0560 -0.0471 -0.0723 -0.0336 -0.0067 -0.0021 -0.0541 0.0661 0.0628 -0.1231 0.0472 0.0382 0.0343 -0.0489 0.0677 -0.0233 -0.0018 -0.0637 -0.0537 0.0262 -0.0072 -0.0641 -0.0023 0.0050 -0.0099 0.0008 -0.1040 0.0255 -0.0774 0.0872 -0.0496 0.1259 0.0727 0.0662 0.0121 0.0274 0.0346 -0.0330 -0.0051 0.0149 -0.0099 0.0617 0.0613 -0.0023 -0.0644 -0.0512 0.0439 0.0105 0.0264 0.0183 -0.0171 0.0141 -0.0330 -0.1142 -0.0598 -0.0745 0.0036 -0.0196 0.0385 -0.1153 -0.0148 0.0290 -0.0159 -0.0546 -0.0229 -0.0159 -0.0613 0.0418 -0.0169 0.0727 0.0739 -0.0360 -0.0401 -0.0260 -0.0012 -0.0452 0.0135 0.1019 0.1254 0.0591 0.0156 0.1324 -0.0012 0.0179 -0.0791 -0.0545 0.0060 0.0070 0.0315 -0.0158 -0.0333 -0.0537 0.1155 0.0209 -0.0414 0.0919 -0.0663 0.0434 0.0534 -0.0324 0.0347 -0.0319 -0.0836 0.0196 -0.0068 0.0489 0.0111 -0.0402 -0.0150 0.0525 0.0733 -0.0244 0.0272 0.0421 0.0245 -0.0382 -0.0431 -0.0394 -0.1068 0.0013 0.0754 -0.0197 -0.0160 0.0295 0.0201 -0.1482 -0.1019 0.0268 -0.0359 -0.0386 0.0520 -0.0556 -0.0306 0.0504 -0.0563 -0.0453 0.0187 -0.0465 0.0508 -0.0409 -0.0045 0.0389 0.0047 -0.0844 -0.0316 0.0251 0.0191 0.0010 -0.0509 0.0382 -0.0021 -0.0544 0.0041 0.0867 -0.0691 0.1239 -0.0463 0.0282 -0.0628 0.0969 -0.0607 0.0281 0.0016 0.0848 0.0033 0.0248 -0.0417 0.0300 0.0178 0.0124 0.0254 -0.0082 0.0339 -0.0189 -0.0541 0.0553"
    # y = np.array([float(x) for x in abscond_retro_string.split(" ")[1:]])
    # x = [[np.array([float(x) for x in abscond_non_retro_string.split(" ")[1:]]).transpose()]]
    # print(y)
    # # retrofit_dcgan = pickle.load(open("model_save_final.pickle",'rb'))
    # # print(list(common_vocabulary).index("abscond"))
    # pred_y = gan.g_AB.predict(x)[0]
    # print(pred_y)
    # gan = pickle.load(open("model.pickle","rb"))
    # print(gan.X_test)
    # print("COmpiling and testing!")
    # gan.g_AB.compile(optimizer="adam",loss="mae",metrics=["accuracy"])
    # print(gan.g_AB.evaluate(gan.X_test,gan.Y_test))


