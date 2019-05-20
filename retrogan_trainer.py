from __future__ import print_function, division

import datetime
import os
from random import shuffle

import sklearn

import numpy as np
from keras.engine import Layer
from keras.engine.saving import load_model
from keras.layers import BatchNormalization, add, multiply, Conv1D, Reshape, Flatten, UpSampling1D, MaxPooling1D
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import plot_model
from tqdm import tqdm
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import tools
from tools import find_in_fasttext, find_in_retrofitted, \
    load_noisiest_words, load_noisiest_words_dataset, find_in_dataset, find_closest_2

from numpy.random import seed

seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
os.environ['KMP_DUPLICATE_LIB_OK']='True'



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
        return K.tf.multiply(self.k, x)

    def compute_output_shape(self, input_shape):
        return input_shape


def attention(layer_input):
    # ATTENTION PART STARTS HERE
    attention_probs = Dense(layer_input._keras_shape[1], activation='softmax')(layer_input)
    attention_mul = multiply([layer_input, attention_probs]
                             )
    attention_scale = ConstMultiplierLayer()(attention_mul)
    attention = add([layer_input, attention_scale])
    # ATTENTION PART FINISHES HERE
    return attention

class RetroCycleGAN():
    def __init__(self, save_index = "0",save_folder="./", generator_size =32, 
                 discriminator_size=64,word_vector_dimensions=300,
                 discriminator_lr=0.0004,generator_lr=0.0001,
                 lambda_cycle=3,lambda_id_weight=0.4,
                 clip_value=0.5,cn=4):
        self.save_folder = save_folder
        # Input shape
        self.word_vector_dimensions = word_vector_dimensions
        self.img_shape = (self.word_vector_dimensions,)#, self.channels)
        self.save_index = save_index

        # Number of filters in the first layer of G and D
        self.gf = generator_size
        self.df = discriminator_size

        # Loss weights
        self.lambda_cycle = lambda_cycle                 # Cycle-consistency loss
        self.lambda_id = lambda_id_weight * self.lambda_cycle    # Identity loss

        d_lr = discriminator_lr
        g_lr = generator_lr
        cv = clip_value
        cn = cn
        self.d_A = self.build_discriminator(name="word_vector_discriminator")
        self.d_B = self.build_discriminator(name="retrofitted_word_vector_discriminator")
        
        def create_opt(lr):
            return Adam(lr, clipvalue=cv,clipnorm=cn)
        
        self.d_A.compile(loss='mse',
            optimizer=create_opt(d_lr),
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=create_opt(d_lr),
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator(name="to_retro_generator")
        self.g_AB.summary()
        plot_model(self.g_AB,show_shapes=True)
        self.g_BA = self.build_generator(name="from_retro_generator")
        self.g_BA.summary()
        # Input images from both domains
        unfit_wv = Input(shape=self.img_shape,name="plain_word_vector")
        fit_wv = Input(shape=self.img_shape,name="retrofitted_word_vector")

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
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        unfit_wv_id, fit_wv_id ],
                              name="combinedmodel")
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                              #TODO ADD A CUSTOM LOSS THAT SIMPLY ADDS A
                              # GENERALIZATION CONSTRAINT ON THE MAE
                            optimizer=create_opt(g_lr))

        plot_model(self.combined,to_file="RetroGAN.png",show_shapes=True)
    def build_generator(self,name):
        """U-Net Generator"""

        def dense(layer_input, filters, f_size=6, normalization=True):
            """Layers used during downsampling"""
            d = Dense(filters,activation="relu")(layer_input)
            if normalization:
                d = BatchNormalization()(d)
            return d

        def conv1d(layer_input,filters,f_size=6,strides=1,normalization=True):
            d = Conv1D(filters,f_size,strides=strides,activation="relu")(layer_input)

            return d

        def deconv1d(layer_input,filters,f_size=6,strides=1,normalization=True):
            d = UpSampling1D(filters,f_size,strides=strides,activation="relu")(layer_input)
            return d


        # Image input
        inpt = Input(shape=self.img_shape)
        #Continue into fc layers
        d0 = dense(inpt, self.gf*8,normalization=False)
        r = Reshape((-1, 1))(d0)
        # Downscaling
        # t1 = conv1d(r,self.gf*8,f_size=6)
        t2 = conv1d(r,self.gf*4,f_size=8)
        t3 = conv1d(t2,self.gf,f_size=4)
        f = MaxPooling1D(pool_size=4)(t3)
        f = Flatten()(f)
        attn = attention(f)

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

        #Last 2 fc layers
        #Maybe attention
        # r = Reshape((-1, 1))(attn)
        # # Downscaling
        # t4 = conv1d(r, self.gf, f_size=4)
        # t5 = conv1d(t4, self.gf *2, f_size=4)
        # f = Flatten()(t5)
        # f = UpSampling1D(size=2)(t5)
        d4 = dense(attn, self.gf*8,normalization=False)
        output_img = Dense(dimensionality)(d4)
        return Model(inpt, output_img,name=name)

    def build_discriminator(self,name):

        def d_layer(layer_input, filters, f_size=7, normalization=True):
            """Discriminator layer"""
            d = Dense(filters,activation="relu")(layer_input)
            # d = LeakyReLU(alpha=0.0002)(d)
            if normalization:
                d = BatchNormalization()(d)
            # d = Dropout(0.75)(d)
            return d

        inpt = Input(shape=self.img_shape)
        d1 = d_layer(inpt, self.df*16, normalization=False)
        d1 = d_layer(d1, self.df*8)
        d2 = d_layer(d1, self.df*4)
        d2 = attention(d2)
        d3 = d_layer(d2, self.df*2)
        d4 = d_layer(d3, self.df*1)
        validity = Dense(1)(d4)
        return Model(inpt, validity,name=name)

    def load_weights(self):
        try:
            self.g_AB.reset_states()
            self.g_BA.reset_states()
            self.combined.reset_states()
            self.d_B.reset_states()
            self.d_A.reset_states()
            self.d_A.load_weights(os.path.join(self.save_folder,"fromretrodis.h5"))
            self.d_B.load_weights(os.path.join(self.save_folder,"toretrodis.h5"))
            self.g_AB.load_weights(os.path.join(self.save_folder,"toretro.h5"))
            self.g_BA.load_weights(os.path.join(self.save_folder,"fromretro.h5"))
            self.combined.load_weights(os.path.join(self.save_folder,"combined_model.h5"))

        except Exception as e:
            print(e)

    def train(self, epochs, dataset, batch_size=1, sample_interval=50,noisy_entries_num=5,batches=900,add_noise=False):
        testwords = ["human", "dog", "cat", "potato", "fat"]
        fastext_version = find_in_dataset(testwords, dataset=dataset["directory"]+dataset["original"])
        retro_version = find_in_dataset(testwords, dataset=dataset["directory"]+dataset["retrofitted"])

        start_time = datetime.datetime.now()
        # self.load_weights(extension="0")
        # self.load_weights()
        for idx, word in enumerate(testwords):
           print(word)
           retro_representation = rcgan.g_AB.predict(fastext_version[idx].reshape(1, dimensionality))
           print(sklearn.metrics.mean_absolute_error(retro_version[idx],
                                                     retro_representation.reshape((dimensionality,))))
        # Adversarial loss ground truths
        # fake = np.random.uniform(0.0,0.1,size=(batch_size,))

        X_train = Y_train = X_test = Y_test = None

        seed = 32
        X_train, Y_train= load_noisiest_words_dataset(dataset,
                                                                       save_folder="fasttext_model/",
                                                                       threshold=0.90,
                                                                       cache=False)
        print("Done")
        # X_train, Y_train, X_test, Y_test = load_training_input_3(seed=seed,test_split=0.001,dataset=dataset)

        def load_batch(batch_size=2):
            l = X_train.shape[0]
            iterable = list(range(0,l))
            shuffle(iterable)
            for ndx in tqdm(range(0, l, batch_size),ncols=30):
                ixs = iterable[ndx:min(ndx + batch_size, l)]
                imgs_A = X_train[ixs]
                imgs_B = Y_train[ixs]
                yield imgs_A,imgs_B
        for epoch in range(epochs+1):
            noise = np.random.normal(size=(batch_size,dimensionality),scale=0.01)
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
                if epoch%2==0:
                    imgs_A = np.add(noise[0:imgs_A.shape[0],:],imgs_A)

                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                valid = np.ones((imgs_A.shape[0],), )  # *noisy_entries_num,) )
                # valid = np.random.uniform(0.9,1.0,size=(batch_size,))
                fake = np.zeros((imgs_A.shape[0],))  # *noisy_entries_num,) )

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
            try:
                # Check at end of epoch!
                for idx, word in enumerate(testwords):
                    print(word)
                    retro_representation = rcgan.g_AB.predict(fastext_version[idx].reshape(1, dimensionality))
                    print(sklearn.metrics.mean_absolute_error(retro_version[idx],
                                                              retro_representation.reshape((dimensionality,))))
                self.save_model()
            except Exception as e:
                print(e)

        self.save_model()
        return X_train, Y_train,X_train,Y_train
        # return X_train, Y_train, X_test, Y_test

    def save_model(self):
        self.d_A.save(os.path.join(self.save_folder,"fromretrodis.h5"),include_optimizer=False)
        self.d_B.save(os.path.join(self.save_folder,"toretrodis.h5"),include_optimizer=False)
        self.g_AB.save(os.path.join(self.save_folder,"toretrogen.h5"), include_optimizer=False)
        self.g_BA.save(os.path.join(self.save_folder,"fromretrogen.h5"), include_optimizer=False)
        self.combined.save(os.path.join(self.save_folder,"combined_model.h5"), include_optimizer=False)


if __name__ == '__main__':
    config = tf.ConfigProto()
    global dimensionality
    dimensionality = 300
    tools.dimensionality=dimensionality
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
    postfix = "ft"
    save_folder = "fasttext_model/trained_retrogan/"+str(datetime.datetime.now())+postfix
    if not os.path.exists(save_folder):
        os.makedirs(save_folder,exist_ok=True)
    rcgan = RetroCycleGAN(save_folder=save_folder)
    ds = {"original":"unfitted.hd5clean",
                                                                        "retrofitted":"fitted-debias.hd5clean",
                                                            "directory":"./fasttext_model/"}
    # X_train, Y_train, X_test, Y_test = rcgan.train(epochs=50, batch_size=32, sample_interval=100,
    #                                                dataset=ds)

    # rcgan.g_AB.load_weights(save_folder+"/toretrogen.h5")
    # X_train, Y_train, X_test, Y_test = rcgan.train(epochs=50, batch_size=32, sample_interval=100,
    #                                                dataset=ds)
    rcgan.g_AB.load_weights("fasttext_model/trained_retrogan/2019-05-14 22:55:42.280715ft/toretrogen.h5")

    testwords = ["human","dog","cat","potato","fat"]

    fastext_version = find_in_dataset(testwords,dataset=ds["directory"]+ds["original"])
    print(fastext_version)
    retro_version = find_in_dataset(testwords,dataset=ds["directory"]+ds["retrofitted"])
    print(retro_version)

    for idx,word in enumerate(testwords):
        print(word)
        retro_representation = rcgan.g_AB.predict(fastext_version[idx].reshape(1, dimensionality))

        tools.find_closest_in_dataset(retro_representation,n_top=20,dataset=ds["directory"]+ds["retrofitted"])
        print(sklearn.metrics.mean_absolute_error(retro_version[idx], retro_representation.reshape((dimensionality,))))
