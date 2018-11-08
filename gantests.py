'''
DCGAN on retrofitting OOV words using Keras
Author: Pedro
Based on code by Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''
import os
import pickle

import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, Conv1D
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        print("Discriminator-----")
        self.D = Sequential()
        dim = 32
        depth = 8
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64

        # input_shape = (self.img_rows, self.img_cols)
        self.D.add(Dense(2*dim * dim * depth, input_dim=self.img_rows))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(Activation('relu'))
        # self.D.add(Reshape((dim, dim, depth)))
        self.D.add(Dropout(dropout))
        # self.D.add(Conv2D(depth*1, 5, strides=2,\
        #     padding='same'))
        self.D.add(Dense(dim*depth))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(Dense(dim*depth))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(Dense(dim*depth))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(Dense(dim*depth))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        # self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self,input_dim=100):
        if self.G:
            return self.G
        print("Generator-----")
        self.G = Sequential()
        dropout = 0.4
        depth = 64*4
        dim = 7
        # In: 100
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=input_dim))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        # self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Flatten())
        self.G.add(Dense(input_dim))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        # optimizer = RMSprop(lr=0.00001)#, decay=6e-8)
        # optimizer = RMSprop()
        # optimizer = Adam(lr=0.00001,amsgrad=True)
        # optimizer = RMSprop(lr=0.0000001,decay=3e-8)
        optimizer = Adam(lr=0.0000001,amsgrad=True)

        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        # optimizer = RMSprop(lr=0.00001, decay=3e-8)
        # optimizer = RMSprop(lr=0.000001)
        optimizer = Adam(lr=0.000001,amsgrad=True)
        self.AM = Sequential()
        self.AM.add(self.generator(self.img_rows))
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM


def load_embedding(path,limit=100000):
    if os.path.exists(path):
        words = []
        vectors = []
        f = open(path,encoding="utf-8")
        skip_first = True
        print("Starting loading",path)
        for line in f:
            if limit is not None:
                if(len(words)>=limit):
                    break
            if skip_first ==True:
                skip_first = False
                continue
            linesplit = line.split(" ")
            words.append(linesplit[0])
            vectors.append([float(x) for x in linesplit[1:]])
            # print("Appended")
        print("Finished")
        return words,vectors
    else:
        raise FileNotFoundError(path+" does not exist")

common_vocabulary = None
common_vectors = None
common_retro_vectors = None
def load_training_input():
    global common_vocabulary,common_vectors,common_retro_vectors
    words, vectors = load_embedding("retrogan/wiki-news-300d-1M-subword.vec")
    retrowords, retrovectors =load_embedding("retrogan/numberbatch")
    common_vocabulary = set(words).intersection(set(retrowords))
    common_vocabulary = np.array(list(common_vocabulary))
    common_retro_vectors = np.array([retrovectors[retrowords.index(word)]for word in common_vocabulary])
    common_vectors = np.array([vectors[words.index(word)]for word in common_vocabulary])
    del retrowords,retrovectors,words,vectors
    print("Size of common vocabulary:"+str(len(common_vocabulary)))
    return common_vocabulary,common_vectors,common_retro_vectors

class RETRO_DCGAN(object):
    def __init__(self):
        self.img_rows = 300
        self.img_cols = 1
        self.channel = 1
        global common_retro_vectors
        self.x_train = common_retro_vectors
        self.DCGAN = DCGAN(self.img_rows,self.img_cols,self.channel)
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator(self.img_rows)

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        global common_vectors
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :]
            # noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            # noise = np.random.choice(common_vectors,batch_size,False)
            noise = common_vectors[np.random.choice(len(common_vectors), size=batch_size, replace=False)]
            embeddings_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, embeddings_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            # noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            # noise = np.random.choice(common_vectors,batch_size,False)
            noise = common_vectors[np.random.choice(len(common_vectors), size=batch_size, replace=False)]
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if i%save_interval==0:
                pickle.dump(self, open("model_save_"+str(i)+".pickle", 'wb'))
        pickle.dump(self, open("model_save_final.pickle", 'wb'))


if __name__ == '__main__':

    # mnist_dcgan = MNIST_DCGAN()
    # timer = ElapsedTimer()
    # mnist_dcgan.train(train_steps=1000, batch_size=256, save_interval=500)
    # timer.elapsed_time()
    # mnist_dcgan.plot_images(fake=True)
    # mnist_dcgan.plot_images(fake=False, save2file=True)
    load_training_input()
    print("Loaded input")
    retrofit_dcgan = RETRO_DCGAN()
    print("In train")
    retrofit_dcgan.train(train_steps=3000,save_interval=100)

    abscond_non_retro_string = "abscond 0.0083 0.0106 -0.0268 -0.0179 -0.0123 -0.0072 -0.0058 -0.0372 0.0215 0.0092 -0.0087 -0.0091 -0.0002 -0.0316 -0.0138 -0.0186 0.0189 0.0047 0.0237 0.0043 -0.0229 -0.0097 -0.0151 0.0094 -0.0085 0.0259 -0.0083 0.0093 -0.0311 0.0042 -0.0082 -0.0047 -0.0314 0.0360 -0.0176 -0.0247 -0.0167 0.0047 -0.0126 0.0110 -0.0013 -0.0205 0.0101 -0.0122 0.0060 0.0189 0.0311 0.0070 0.0056 -0.0056 -0.0062 0.0206 -0.0230 -0.0413 -0.0595 -0.0164 -0.0098 -0.0071 -0.0358 -0.0132 0.0216 -0.0071 0.0353 0.0016 -0.0055 -0.0022 0.0198 -0.0473 -0.0023 0.0197 -0.0023 0.0097 0.0426 -0.0248 0.0226 0.0097 0.0219 0.0313 0.0260 0.0198 0.0124 0.0188 -0.0033 0.0273 -0.0140 0.0084 0.0328 0.0050 -0.0147 0.0026 -0.0000 -0.0004 -0.0546 -0.0010 -0.0100 -0.0234 -0.0023 -0.0256 -0.0069 0.0037 0.0184 -0.0153 0.0025 -0.0033 0.0115 -0.0353 -0.0161 -0.0114 0.0079 0.0151 0.0010 0.0172 0.0448 0.0219 -0.0249 -0.0007 0.0135 -0.0117 -0.0040 -0.0212 0.0224 -0.0031 0.0216 -0.0259 0.0281 -0.0152 0.0123 -0.0021 -0.0314 0.0218 0.0015 -0.0040 -0.0079 -0.0083 0.0050 0.0043 -0.0159 0.0147 0.0299 0.0155 0.0092 0.0089 0.0115 -0.0311 0.0300 0.0211 -0.0071 0.0061 -0.0047 -0.0168 0.0086 0.0225 -0.0166 -0.0245 0.0236 -0.0077 -0.0140 -0.0174 -0.0177 0.0032 0.0250 0.0140 -0.0224 0.0173 0.0130 0.0002 0.0274 -0.0072 -0.0231 0.0372 -0.0039 -0.0194 -0.0168 -0.0048 -0.0256 0.0111 0.0283 -0.0138 -0.0065 -0.0137 0.0103 -0.0142 -0.0154 -0.0231 0.0073 -0.0247 0.0107 0.0420 0.0073 0.0143 -0.0061 0.0097 -0.0217 -0.0047 -0.0055 -0.0019 0.0270 0.0021 -0.0064 0.0107 -0.0554 0.0341 -0.0097 0.0129 -0.0084 -0.0033 0.0114 -0.0177 -0.0206 -0.0137 0.0159 -0.0186 0.0458 -0.0003 0.0344 0.0011 -0.0099 0.0272 0.0023 -0.0138 -0.0150 -0.0023 -0.0102 0.0390 -0.0184 -0.0149 -0.0399 0.0448 -0.0124 0.0030 -0.0198 -0.0237 -0.0301 -0.0379 0.0394 0.0074 0.0107 0.0116 0.0025 0.0066 0.0677 0.0044 -0.0065 0.0203 0.0011 0.0007 -0.0023 -0.0304 -0.0320 0.0150 -0.0175 -0.0003 -0.0007 -0.0290 -0.0009 0.0059 0.0029 0.0116 -0.0115 0.0038 -0.0466 0.0101 -0.0172 -0.0422 -0.0049 -0.0273 -0.0213 -0.0297 0.0205 -0.0035 -0.0134 0.0487 -0.0358 -0.0319 -0.0106 -0.0173 0.0521 -0.0056 -0.0125 -0.0032 0.0036 -0.0117 -0.0042 0.0037 0.0135 0.0280 0.0046 -0.0057 -0.0085 0.0146 0.0135 -0.0021 0.0129 0.0088 -0.0240 -0.0376 -0.0146 0.0147 -0.0194 0.0091"
    abscond_retro_string = "abscond 0.1078 0.1119 -0.0016 0.0495 -0.0000 -0.1264 -0.0106 0.1652 -0.0126 0.0014 -0.0152 -0.2043 -0.1176 0.0370 -0.0422 -0.0036 -0.1001 0.1304 -0.1046 0.0412 -0.0869 0.0258 -0.0425 -0.0860 -0.1142 0.0464 -0.0693 -0.0286 -0.0238 -0.0273 -0.1242 0.0355 0.1453 0.0868 0.0621 -0.0667 -0.0268 -0.0045 -0.0517 0.0195 0.1216 0.0675 0.0023 -0.0394 0.0637 0.0011 0.0182 -0.0634 0.0148 0.0288 -0.0035 -0.1871 -0.1273 -0.0948 -0.0088 0.0042 0.0227 -0.0361 -0.0315 0.0192 -0.0185 -0.0205 0.0164 0.0869 0.0870 0.0031 0.0705 -0.0816 0.0203 -0.0322 0.0490 0.0160 0.0381 0.0149 -0.0820 -0.0296 0.0640 -0.0127 -0.0219 -0.0524 0.0362 -0.0185 0.0136 0.0098 0.0577 0.0168 0.0320 -0.0758 -0.0391 -0.0265 -0.0238 -0.1031 -0.0499 -0.0783 -0.0551 -0.0252 -0.0356 -0.0828 -0.0040 0.0604 -0.0173 0.0419 0.0603 -0.0523 -0.0525 -0.0550 -0.0030 -0.0848 -0.0196 0.1151 -0.0197 0.0151 0.0234 -0.0275 0.0093 0.0391 -0.0724 0.0403 0.0170 -0.0186 0.0557 0.0261 0.0008 0.0300 0.0560 -0.0471 -0.0723 -0.0336 -0.0067 -0.0021 -0.0541 0.0661 0.0628 -0.1231 0.0472 0.0382 0.0343 -0.0489 0.0677 -0.0233 -0.0018 -0.0637 -0.0537 0.0262 -0.0072 -0.0641 -0.0023 0.0050 -0.0099 0.0008 -0.1040 0.0255 -0.0774 0.0872 -0.0496 0.1259 0.0727 0.0662 0.0121 0.0274 0.0346 -0.0330 -0.0051 0.0149 -0.0099 0.0617 0.0613 -0.0023 -0.0644 -0.0512 0.0439 0.0105 0.0264 0.0183 -0.0171 0.0141 -0.0330 -0.1142 -0.0598 -0.0745 0.0036 -0.0196 0.0385 -0.1153 -0.0148 0.0290 -0.0159 -0.0546 -0.0229 -0.0159 -0.0613 0.0418 -0.0169 0.0727 0.0739 -0.0360 -0.0401 -0.0260 -0.0012 -0.0452 0.0135 0.1019 0.1254 0.0591 0.0156 0.1324 -0.0012 0.0179 -0.0791 -0.0545 0.0060 0.0070 0.0315 -0.0158 -0.0333 -0.0537 0.1155 0.0209 -0.0414 0.0919 -0.0663 0.0434 0.0534 -0.0324 0.0347 -0.0319 -0.0836 0.0196 -0.0068 0.0489 0.0111 -0.0402 -0.0150 0.0525 0.0733 -0.0244 0.0272 0.0421 0.0245 -0.0382 -0.0431 -0.0394 -0.1068 0.0013 0.0754 -0.0197 -0.0160 0.0295 0.0201 -0.1482 -0.1019 0.0268 -0.0359 -0.0386 0.0520 -0.0556 -0.0306 0.0504 -0.0563 -0.0453 0.0187 -0.0465 0.0508 -0.0409 -0.0045 0.0389 0.0047 -0.0844 -0.0316 0.0251 0.0191 0.0010 -0.0509 0.0382 -0.0021 -0.0544 0.0041 0.0867 -0.0691 0.1239 -0.0463 0.0282 -0.0628 0.0969 -0.0607 0.0281 0.0016 0.0848 0.0033 0.0248 -0.0417 0.0300 0.0178 0.0124 0.0254 -0.0082 0.0339 -0.0189 -0.0541 0.0553"
    y = [float(x) for x in abscond_retro_string.split(" ")[1:]]
    x = [[np.array([float(x) for x in abscond_non_retro_string.split(" ")[1:]]).transpose()]]
    print(x)
    # retrofit_dcgan = pickle.load(open("model_save_final.pickle",'rb'))
    # print(list(common_vocabulary).index("abscond"))

    y_pred = retrofit_dcgan.generator.predict(x)
    print(y)
    print(y_pred)