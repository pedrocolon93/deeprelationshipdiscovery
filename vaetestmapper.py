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

from keras.layers import Lambda, Input, Dense, BatchNormalization, Dropout
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
from vaetest3 import VAE

if __name__ == '__main__':
    X_train= Y_train= X_test= Y_test = None
    regen = False
    file = "data.pickle"
    if not os.path.exists(file) or regen:
        X_train, Y_train, X_test, Y_test = load_training_input_2(normalize=False)
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
    # input_vae = VAE(intermediate_layer_count=6,latent_dim=128)
    # input_vae.create_vae()
    # input_vae.configure_vae()
    # input_vae.compile_vae()
    # input_vae.fit(X_train,X_test,"input_vae.h5")
    # print(X_test)
    # print(input_vae.predict(X_test))


    output_vae = VAE(a_weight=1, b_weight=1, intermediate_layer_count=1, latent_dim=16, intermediate_dimension=512,intermediate_mult = 1,
                    epochs=5,loss_weight=0.00001,batch_size=64,lr=0.0005,batch_norm=True)
    output_vae.create_vae()
    output_vae.configure_vae()
    output_vae.compile_vae()
    output_vae.fit_2(X_train, Y_train)
    print(Y_test)
    print(output_vae.predict(X_test))


    # plot_results(models,
    #              data,
    #              batch_size=batch_size,
    #              model_name="vae_mlp")