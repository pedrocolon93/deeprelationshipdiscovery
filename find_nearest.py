import faiss
from tqdm import tqdm

import tools
import pandas as pd
import numpy as np

from rcgan import RetroCycleGAN

if __name__ == '__main__':
    # dimensionality = 300
    # n_top = 10
    # o = pd.read_hdf("trained_models/retroembeddings/ft_full_alldata/retroembeddings.h5", 'mat', encoding='utf-8')
    # testvec = tools.find_in_dataset(["cat"], o,prefix="")
    # # testvec = pred_y
    # # print(testvec)
    # index = faiss.IndexFlatIP(dimensionality)  # build the index
    # # print(index.is_trained)
    # index.add(o.values.astype(np.float32))  # add vectors to the index
    # # print(index.ntotal)
    # tst = np.array([testvec.astype(np.float32)])
    # tst = tst.reshape((tst.shape[0], tst.shape[-1]))
    # # print(tst.shape)
    # D, I = index.search(tst, n_top)  # sanity check
    # # print(I)
    # # print(D)
    # # print(o.iloc[I[0]])
    # final_n_results = o.iloc[I[0]].values
    # final_n_results_words = o.index[I[0]].values
    # print(final_n_results_words)
    print("Starting")
    # dataset = {
    #     "original": "completefastext.txt.hdf",
    #     "retrofitted": "finalfullfasttext.hdf",
    #     "directory": "ft_full_paperdata/",
    #     "rc": None
    # }
    dataset = {
        "original": "glove_840b_unseen.hdf",
        "retrofitted": "glove_840b_retrogan.hdf",
        "directory": "light-ls-test-data/",
        "rc": None
    }
    outputname = "glove_840b_retrogan.txt"
    # dataset = {
    #     "original": "allgove.hdf",
    #     "retrofitted": "fullglove.hdf",
    #     "directory": "glove_full_alldata/",
    #     "rc": None
    # }
    tools.datasets.update({"mine": [dataset["original"], dataset["retrofitted"]]})
    rcgan = RetroCycleGAN(save_folder="test", batch_size=32, generator_lr=0.0001, discriminator_lr=0.001)
    rcgan.load_weights(preface="final", folder="final_retrogan/glovefinaltest")
    print("\n")
    # sl = tools.test_sem(rcgan.g_AB, dataset, dataset_location="testing/SimLex-999.txt",
    #                     fast_text_location="fasttext_model/cc.en.300.bin")[0]
    # sv = tools.test_sem(rcgan.g_AB, dataset, dataset_location="testing/SimVerb-3500.txt",
    #                     fast_text_location="fasttext_model/cc.en.300.bin")[0]
    dimensionality = 300
    # X_train, Y_train = tools.load_all_words_dataset_3(dataset,
    #                                                   save_folder="adversarial_paper_data/",
    #                                                   threshold=0.90,
    #                                                   cache=False,
    #                                                   remove_constraint=None)
    o = pd.read_hdf(dataset["directory"]+dataset["original"], 'mat', encoding='utf-8')
    o = o.dropna()
    cns = []
    print("Loading concepts")
    for i in tqdm(o.index):
        cns.append(i)
    X_train = o.loc[cns, :]
    print(X_train)
    # print(X_train.loc["christen"])
    # testwords = ["human"]
    # print("The test word vectors are:", testwords)
    # ft version
    vals = np.array(
        rcgan.g_AB.predict(np.array(X_train.values).reshape((-1, dimensionality)),
                           batch_size=64,verbose=1)
    )

    testds = pd.DataFrame(data=vals, index=X_train.index)
    sl = tools.test_sem(rcgan.g_AB, o, dataset_location="testing/SimLex-999.txt",
                        fast_text_location="fasttext_model/cc.en.300.bin", prefix="")[0]

    testds.dropna(inplace=True)
    print("Dumping to hdf")
    testds.to_hdf(dataset["directory"]+outputname+"hdf","mat")

    Y_train = testds.loc[cns,:]
    print("Dumping to text file")
    output_ar = dataset["directory"]+outputname
    with open(output_ar,"w") as ar:
        for concept in tqdm(cns):
            try:
                rvline = str(concept)+" "+' '.join(map(str, Y_train.loc[concept,:]))+"\n"
                ar.write(rvline)
            except Exception as e:
                print("Could not process line:",concept,e)
                continue
    print("Finished the dumps")
    exit(0)
    print(testds)
    tools.directory = dataset["directory"]
    tools.dimensionality=300
    tools.datasets.update({"mine": [dataset["original"], dataset["retrofitted"]]})

    fastext_words = tools.find_in_fasttext(testwords, dataset="mine", prefix="")
    print("Closest in default fasttext")
    for idx,word in enumerate(testwords):
        print(tools.find_closest_in_dataset(fastext_words[idx], o, n_top=10))

    print("*"*100)
    for idx, word in enumerate(testwords):
        print(word)
        retro_representation = rcgan.g_AB.predict(fastext_words[idx].reshape(1, dimensionality))
        print(tools.find_closest_in_dataset(retro_representation, testds,n_top=10))