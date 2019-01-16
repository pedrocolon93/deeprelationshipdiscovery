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

from keras.layers import Lambda, Input, Dense, BatchNormalization
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
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

class VAE():
    def __init__(self, input_dimension=300, output_dimension=300, intermediate_dimension=512, batch_size = 128, latent_dim=64, epochs=50,use_mse = True,intermediate_layer_count = 4):
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

    def create_vae(self):
        # VAE model = encoder + decoder
        # build encoder model
        self.inputs = Input(shape=self.input_shape, name='encoder_input')
        x = Dense(self.intermediate_dim, activation='relu')(self.inputs)

        for i in range(2,self.intermediate_layer_count+2):
            x = Dense(self.intermediate_dim * i, activation='relu')(x)

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
        x = Dense(self.intermediate_dim * (self.intermediate_layer_count+1), activation='relu')(latent_inputs)
        for i in reversed(range(1,self.intermediate_layer_count+1)):
            x = Dense(self.intermediate_dim*i, activation='relu')(x)

        self.outputs = Dense(self.output_shape[0], activation='tanh')(x)

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
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            return vae_loss
        self.loss = my_vae_loss
        # self.vae.add_loss(vae_loss)

    def compile_vae(self):
        optimizer = RMSprop(lr=0.00001, decay=1e-8)
        # optimizer = SGD(decay=1e-8,nesterov=True)
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

    def predict(self,input):
        pred = self.vae.predict(input)
        return pred

# MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()





if __name__ == '__main__':
    X_train= Y_train= X_test= Y_test = None
    file = "data.pickle"
    if not os.path.exists(file):
        X_train, Y_train, X_test, Y_test = load_training_input_2()
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

    #
    input_vae = VAE()
    input_vae.create_vae()
    input_vae.configure_vae()
    input_vae.compile_vae()
    input_vae.fit(X_train,X_test,"input_vae.h5")
    print(X_test)
    print(input_vae.predict(X_test))


    output_vae = VAE()
    output_vae.create_vae()
    output_vae.configure_vae()
    output_vae.compile_vae()
    output_vae.fit(Y_train, Y_test,"output_vae.h5")
    print(Y_test)
    print(input_vae.predict(Y_test))


    # plot_results(models,
    #              data,
    #              batch_size=batch_size,
    #              model_name="vae_mlp")