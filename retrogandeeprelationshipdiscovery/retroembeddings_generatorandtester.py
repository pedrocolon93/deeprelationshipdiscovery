from __future__ import print_function, division

import pandas
import pandas as pd
import sklearn
from failed_tests.retrogan_trainer_attractrepel import *
from tensorflow_core.python.framework.random_seed import set_random_seed
from tensorflow_core.python.keras.optimizer_v2.adam import Adam
from tensorflow_core.python.keras.saving.save import load_model

import tools
from tools import find_in_fasttext
from vocabulary_cleaner import cleanup_vocabulary_nb_based

seed(1)

set_random_seed(2)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    # Software parameters
    trained_model_path = "final_retrogan/0/finaltoretrogen.h5"
    experiment_name = "ft_full_paperdata"
    retroembeddings_folder = "./trained_models/retroembeddings/" + experiment_name
    clean = False
    try:
        os.mkdir(retroembeddings_folder)
    except:
        pass
    retrogan_word_vector_output_path = retroembeddings_folder + "/" + "retroembeddings.h5"
    dataset = 'mine'
    tools.directory = "ft_full_paperdata/"
    dimensionality = 300
    tools.dimensionality = dimensionality
    # tools.datasets["mine"] = ["unfitted.hd5clean", "fitted-debias.hd5clean"]
    tools.datasets["mine"] = ["completefastext.txt.hdf", "fullfasttext.hdf"]

    if clean:
        numberbatch_file_loc = 'retrogan/mini.h5'
        target_file_loc = tools.directory + tools.datasets["mine"][0]
        cleanup_vocabulary_nb_based(numberbatch_file_loc, target_file_loc)
        tools.datasets["mine"][0] += "clean"

    print("Dataset:", tools.datasets[dataset])
    plain_word_vector_path = plain_retrofit_vector_path = tools.directory
    plain_word_vector_path += tools.datasets[dataset][0]
    plain_retrofit_vector_path += tools.datasets[dataset][1]
    print("Loading the model!")
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
    word_embeddings = pd.read_hdf(plain_word_vector_path, 'mat', encoding='utf-8').swapaxes(0,1)
    # # word_embeddings = word_embeddings.loc[[x for x in word_embeddings.index if "." not in x]]
    vals = np.array(
        to_retro_converter.predict(np.array(word_embeddings.values).reshape((-1, dimensionality)),batch_size=64)
    )
    to_txt = True

    if to_txt:
        print("Writing text vectors to ./vectors.txt")
        with open("vectors.txt","w") as f:
            for idx, index in enumerate(word_embeddings.swapaxes(0,1).index):
                v = vals[idx,:]
                f.write(str(index)+" ")
                for value in v:
                    f.write(str(value)+" ")
                f.write("\n")


    retro_word_embeddings = pd.DataFrame(data=vals, index=word_embeddings.swapaxes(0,1).index)
    print("Writing the vectors...")
    retro_word_embeddings.to_hdf(retrogan_word_vector_output_path, "mat")
    retro_word_embeddings = retro_word_embeddings.swapaxes(0,1)
    print(word_embeddings)
    print(retro_word_embeddings)

    testwords = ["human", "cat"]
    print("The test word vectors are:", testwords)
    # ft version
    fastext_words = find_in_fasttext(testwords, dataset=dataset,prefix="en_")
    print("The original vectors are", fastext_words)
    # original retrofitted version
    og_retro = pd.read_hdf(plain_retrofit_vector_path,"mat")
    og_ft = pd.read_hdf(plain_word_vector_path,"mat")
    retro_version = tools.find_in_dataset(testwords, dataset=og_retro, prefix="en_")
    print("The original retrofitted vectors are", retro_version)

    print("Finding the words that are closest in the og fastext")
    for idx, word in enumerate(testwords):
        print(word)
        f = fastext_words[idx]
        print(tools.find_closest_in_dataset(f, plain_word_vector_path))
    print("Finding the words that are closest in the og retrofitting ")
    for idx, word in enumerate(testwords):
        print(word)
        r = retro_version[idx]
        print(tools.find_closest_in_dataset(r, plain_retrofit_vector_path))

    print("Finding the words that are closest to the predictions/mappings that we make")
    for idx, word in enumerate(testwords):
        print(word)
        retro_representation = to_retro_converter.predict(fastext_words[idx].reshape(1, dimensionality))
        print(tools.find_closest_in_dataset(retro_representation, retrogan_word_vector_output_path))
        print(sklearn.metrics.mean_absolute_error(retro_version[idx], retro_representation.reshape((dimensionality,))))

    # print("Evaluating in the entire test dataset for the error.")
    # Load the testing data
    # X_train,Y_train,X_test,Y_test = tools.load_noisiest_words(dataset=dataset)
    # print("Train error",to_retro_converter.evaluate(X_train, Y_train))
    # print("Test error",to_retro_converter.evaluate(X_test,Y_test))
