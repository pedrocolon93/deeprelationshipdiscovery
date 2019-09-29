from __future__ import print_function, division

import datetime
import os

import numpy as np
import pandas
import pandas as pd
import sklearn
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.engine.saving import load_model
from keras.optimizers import Adam
from numpy.random import seed

import tools
from retrogan_trainer import ConstMultiplierLayer
from tools import find_in_fasttext, find_in_retrofitted, \
    find_closest_2
from vocabulary_cleaner import cleanup_vocabulary_nb_based

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    # Hardware config options
    num_cores = 8
    GPU = False
    CPU = True
    num_CPU = 1
    num_GPU = 0

    if GPU:
        num_GPU = 1
        num_CPU = 1
    if CPU:
        num_CPU = 1
        num_GPU = 0

    # TF config options
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores,
                            allow_soft_placement=True,
                            device_count={'CPU': num_CPU,
                                          'GPU': num_GPU})
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

    # config.log_device_placement = True  # to log device placement (on which device the operation ran)

    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    # Software parameters
    trained_model_path = "fasttext_model/trained_retrogan/2019-07-21 23:12:49.367429ft/toretrogen.h5"
    retroembeddings_folder = "./trained_models/retroembeddings/" + str(datetime.datetime.now())
    clean = False

    try:
        os.mkdir(retroembeddings_folder)
    except:
        pass
    retrogan_word_vector_output_path = retroembeddings_folder + "/" + "retroembeddings.h5"
    dataset = 'mine'
    tools.directory = "fasttext_model/"
    dimensionality = 300
    tools.dimensionality = dimensionality
    tools.datasets["mine"] = ["unfitted.hd5clean", "fitted-debias.hd5clean"]
    if clean:
        numberbatch_file_loc = 'retrogan/mini.h5'
        target_file_loc = tools.directory + tools.datasets["mine"][0]
        cleanup_vocabulary_nb_based(numberbatch_file_loc, target_file_loc)
        tools.datasets["mine"][0] += "clean"

    print("Dataset:", tools.datasets[dataset])
    plain_word_vector_path = plain_retrofit_vector_path = tools.directory
    plain_word_vector_path += tools.datasets[dataset][0]
    plain_retrofit_vector_path += tools.datasets[dataset][1]

    # Load the model and init the weights
    to_retro_converter = load_model(trained_model_path,
                                    custom_objects={"ConstMultiplierLayer": ConstMultiplierLayer},
                                    compile=False)
    to_retro_converter.compile(optimizer=Adam(), loss=['mae'])
    to_retro_converter.load_weights(trained_model_path)

    # Generate retrogan embeddings
    print("Generating embeddings")
    retro_df = pandas.DataFrame()
    #
    word_embeddings = pd.read_hdf(plain_word_vector_path, 'mat', encoding='utf-8')
    # # word_embeddings = word_embeddings.loc[[x for x in word_embeddings.index if "." not in x]]
    vals = np.array(
        to_retro_converter.predict(np.array(word_embeddings.values).reshape((-1, dimensionality)), verbose=1))
    retro_word_embeddings = pd.DataFrame(data=vals, index=word_embeddings.index)
    retro_word_embeddings.to_hdf(retrogan_word_vector_output_path, "mat")
    # retrogan_word_vector_output_path = "trained_models/retroembeddings/2019-05-15 01:00:00.000000/retroembeddings.h5"
    # retro_word_embeddings = pd.read_hdf(retrogan_word_vector_output_path,"mat")
    # Specific word tests
    print("The dataset is ", dataset)
    testwords = ["human", "dog", "cat", "potato", "fat"]
    print("The test word vectors are:", testwords)
    # ft version
    fastext_version = find_in_fasttext(testwords, dataset=dataset)
    print("The original vectors are", fastext_version)
    # original retrofitted version
    retro_version = find_in_retrofitted(testwords, dataset=dataset)
    print("The original retrofitted vectors are", retro_version)

    # Check the closest by cosine dist
    print("Finding the words that are closest in the default numberbatch mappings")
    for idx, word in enumerate(testwords):
        print(word)
        retro_representation = retro_version[idx].reshape(1, dimensionality)
        find_closest_2(retro_representation, dataset=dataset)
        print(sklearn.metrics.mean_absolute_error(retro_version[idx], retro_representation.reshape((dimensionality,))))

    print("Finding the words that are closest to the predictions/mappings that we make")
    for idx, word in enumerate(testwords):
        print(word)
        retro_representation = to_retro_converter.predict(fastext_version[idx].reshape(1, dimensionality))
        print(tools.find_closest_in_dataset(retro_representation, retrogan_word_vector_output_path))
        print(sklearn.metrics.mean_absolute_error(retro_version[idx], retro_representation.reshape((dimensionality,))))

    # print("Evaluating in the entire test dataset for the error.")
    # Load the testing data
    # X_train,Y_train,X_test,Y_test = load_noisiest_words(dataset=dataset)
    # print("Train error",to_retro_converter.evaluate(X_train, Y_train))
    # print("Test error",to_retro_converter.evaluate(X_test,Y_test))
