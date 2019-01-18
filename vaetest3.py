'''Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

from keras.layers import Lambda, Input, Dense, BatchNormalization, Dropout, LeakyReLU
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
from sklearn.preprocessing import MinMaxScaler

from tools import load_training_input_2


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim),stddev=1)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VAE():
    def __init__(self,a_weight=1, b_weight=1, batch_norm = True,dropout=0.4, input_dimension=300, output_dimension=300,
                 intermediate_dimension=64, batch_size = 128, latent_dim=64, epochs=50,use_mse = True,
                 intermediate_layer_count = 4):
        # image_size = x_train.shape[1]
        # original_dim = image_size * image_size
        # x_train = np.reshape(x_train, [-1, original_dim])
        # x_test = np.reshape(x_test, [-1, original_dim])
        # x_train = x_train.astype('float32') / 255
        # x_test = x_test.astype('float32') / 255

        # network parameters
        self.input_shape = (input_dimension,)
        self.output_shape = (output_dimension,)
        self.intermediate_dim = intermediate_dimension
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.vae = None
        self.encoder = None
        self.decoder = None
        self.inputs = None
        self.outputs = None
        self.z_mean = None
        self.z_log_var = None
        self.use_mse = use_mse
        self.models = None
        self.loss = None
        self.intermediate_layer_count = intermediate_layer_count
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.a_weight = a_weight
        self.b_weight = b_weight

    def create_vae(self):
        # VAE model = encoder + decoder
        # build encoder model
        self.inputs = Input(shape=self.input_shape, name='encoder_input')
        x = Dense(self.intermediate_dim,)(self.inputs)
        x = LeakyReLU()(x)
        for i in range(2,self.intermediate_layer_count+2):
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = Dense(self.intermediate_dim * i,activation="tanh")(x)
            # x = LeakyReLU()(x)
            x = Dropout(self.dropout)(x)

        self.z_mean = Dense(self.latent_dim, name='z_mean')(x)
        self.z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])

        # instantiate encoder model
        self.encoder = Model(self.inputs, [self.z_mean, self.z_log_var, z], name='encoder')
        self.encoder.summary()
        # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(self.intermediate_dim * (self.intermediate_layer_count+1), activation='tanh')(latent_inputs)
        for i in reversed(range(1,self.intermediate_layer_count+1)):
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = Dense(self.intermediate_dim*i,activation="tanh")(x)
            # x = LeakyReLU()(x)
            x = Dropout(self.dropout)(x)


        self.outputs = Dense(self.output_shape[0])(x)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, self.outputs, name='decoder')
        self.decoder.summary()
        # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        self.outputs = self.decoder(self.encoder(self.inputs)[2])
        vae = Model(self.inputs, self.outputs, name='vae_mlp')

        self.vae =vae
        return vae

    def configure_vae(self):
        self.models = (self.encoder, self.decoder)

        # VAE loss = mse_loss or xent_loss + kl_loss
        if self.use_mse:
            reconstruction_loss = mse(self.inputs, self.outputs)
        else:
            reconstruction_loss = binary_crossentropy(self.inputs,
                                                      self.outputs)
        def my_vae_loss(y_true,y_pred):
            reconstruction_loss = mse(y_true,y_pred)
            reconstruction_loss *= self.input_shape[0]
            kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            # scaling_factor = 1
            a_weight = self.a_weight
            b_weight = self.b_weight
            print(kl_loss)
            vae_loss = K.mean(a_weight*reconstruction_loss + b_weight*kl_loss)/((a_weight+b_weight)/2)
            return vae_loss
        self.loss = my_vae_loss
        # self.vae.add_loss(vae_loss)

    def compile_vae(self):
        # optimizer = RMSprop(decay=1e-6)
        # optimizer = Adam()
        optimizer = SGD(decay=1e-8,nesterov=True)
        self.vae.compile(optimizer=optimizer,loss=self.loss)
        self.vae.summary()
        # plot_model(self.vae,
        #            to_file='vae_mlp.png',
        #            show_shapes=True)

    def load_weights(self,filename):
        self.vae.load_weights(filename)


    def fit(self, X_train, X_test, weight_filename='vae_mlp_mnist.h5' ):
        self.vae.fit(X_train, X_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_test, X_test))
        self.vae.save_weights(weight_filename)

    def evaluate(self, Y_test):
        print(self.vae.evaluate(Y_test))

    def predict(self,input):
        pred = self.vae.predict(input)
        return pred

# MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()





if __name__ == '__main__':
    X_train= Y_train= X_test= Y_test = None
    regen = False
    normalize = False
    file = "data.pickle"
    if not os.path.exists(file) or regen:
        X_train, Y_train, X_test, Y_test = load_training_input_2(normalize=normalize)
        pickle.dump((X_train, Y_train, X_test, Y_test), open(file, "wb"))
    else:
        X_train, Y_train, X_test, Y_test = pickle.load(open("data.pickle",'rb'))


    print("Min\tMax")
    print("Train")
    print(np.min(X_train),np.max(X_train))
    print(np.min(Y_train),np.max(Y_train))
    print("Test")
    print(np.min(X_test),np.max(X_test))
    print(np.min(Y_test),np.max(Y_test))
    print("End")

    # #
    input_vae = VAE(a_weight=1,b_weight=10,intermediate_layer_count=2,latent_dim=64,intermediate_dimension=64,epochs=50)
    input_vae.create_vae()
    input_vae.configure_vae()
    input_vae.compile_vae()
    input_vae.fit(X_train,X_test,"input_vae.h5")
    print(X_test)
    print(input_vae.predict(X_test))


    output_vae = VAE(a_weight=1,b_weight=100, intermediate_layer_count=6,latent_dim=64,intermediate_dimension=64,epochs=50)
    output_vae.create_vae()
    output_vae.configure_vae()
    output_vae.compile_vae()
    output_vae.fit(Y_train, Y_test,"output_vae.h5")
    print(Y_test)
    print(output_vae.predict(Y_test))


    # plot_results(models,
    #              data,
    #              batch_size=batch_size,
    #              model_name="vae_mlp")