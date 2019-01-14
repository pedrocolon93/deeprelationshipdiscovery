import os
import pickle

import numpy as np

from tools import load_training_input_2
from vaetest1 import VAE

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

