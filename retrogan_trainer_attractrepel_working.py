from __future__ import print_function, division

import os
import shutil

# from tensorflow_core.python.keras import backend as K
from rcgan import RetroCycleGAN

os.environ["TF_KERAS"] = "1"
from tensorflow.python.framework.ops import disable_eager_execution
# from tensorflow_core.python.framework.random_seed import set_random_seed
# from tensorflow_core.python.keras.optimizer_v2.adam import Adam
# from tensorflow_core.python.keras.optimizer_v2.rmsprop import RMSProp
# from tensorflow_core.python.keras.optimizer_v2
# from tensorflow_core.python.keras.utils.vis_utils import plot_model
# from tensorflow_core.python.keras.optimizers import Adadelta
import tensorflow as tf

# tf.debugging.set_log_device_placement(True)

import tools


# seed(1)
#
# set_random_seed(2)
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.compat.v1.summary.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

dimensionality = 300

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    print("Removing!!!")
    print("*" * 100)
    shutil.rmtree("logs/", ignore_errors=True)
    print("Done!")
    print("*" * 100)
    disable_eager_execution()
    # with tf.device('/GPU:0'):
    tools.dimensionality = dimensionality
    postfix = "ftar"
    test_ds = [
        # {
        #     "original":"completefastext.txt.hdf",
        #     "retrofitted":"fullfasttext.hdf",
        #     "directory":"ft_full_alldata/",
        #     "rc":None
        # },
        # {
        #     "original":"completefastext.txt.hdf",
        #     "retrofitted":"disjointfasttext.hdf",
        #     "directory":"ft_disjoint_alldata/",
        #     "rc":"adversarial_paper_data/simlexsimverb.words"
        # },
        # {
        #     "original":"completeglove.txt.hdf",
        #     "retrofitted":"fullglove.hdf",
        #     "directory":"glove_full_alldata/",
        #     "rc":None
        # },
        # {
        #     "original":"completeglove.txt.hdf",
        #     "retrofitted":"disjointglove.hdf",
        #     "directory":"glove_disjoint_alldata/",
        #     "rc":"adversarial_paper_data/simlexsimverb.words"
        # },
        # {
        #     "original": "completeglove.txt.hdf",
        #     "retrofitted": "disjointglove.hdf",
        #     "directory": "glove_disjoint_paperdata/",
        #     "rc": "adversarial_paper_data/simlexsimverb.words"
        # },
        # {
        #     "original": "completeglove.txt.hdf",
        #     "retrofitted": "fullglove.hdf",
        #     "directory": "glove_full_paperdata/",
        #     "rc": None
        # },
        # {
        #     "original": "completefastext.txt.hdf",
        #     "retrofitted": "disjointfasttext.hdf",
        #     "directory": "ft_disjoint_paperdata/",
        #     "rc": "adversarial_paper_data/simlexsimverb.words"
        # },
        {
            "original": "completefastext.txt.hdf",
            "retrofitted": "fullfasttext.hdf",
            "directory": "ft_full_paperdata/",
            "rc": None
        }
    ]
    print("Testing")
    print(test_ds)
    print("Checking that everything exists")
    for ds in test_ds:
        a = os.path.exists(os.path.join(ds["directory"], ds["original"]))
        b = os.path.exists(os.path.join(ds["directory"], ds["retrofitted"]))
        print(a, b)
        if not a or \
                not b:
            raise FileNotFoundError("Files in " + str(ds) + "\ndo not exist")
    models = []
    results = []
    final_save_folder = "./final_retrogan"
    os.makedirs(final_save_folder, exist_ok=True)

    for idx, ds in enumerate(test_ds):
        save_folder = "models/trained_retrogan/" + ds["directory"]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        print("Training")
        print(ds)
        tools.directory = ds["directory"]
        rcgan = RetroCycleGAN(save_folder=save_folder, batch_size=32, generator_lr=0.0001, discriminator_lr=0.001)

        # rcgan.load_weights(preface="final", folder="/media/pedro/ssd_ext/mltests/models/trained_retrogan/2020-01-27 00:34:26.680643ftar")
        sl = tools.test_sem(rcgan.g_AB, ds, dataset_location="testing/SimLex-999.txt",
                            fast_text_location="fasttext_model/cc.en.300.bin")[0]
        models.append(rcgan)
        ds_res = rcgan.train(pretraining_epochs=250, epochs=0, batch_size=32, dataset=ds, rc=ds["rc"])
        results.append(ds_res)
        print("*" * 100)
        print(ds, results[-1])
        print("*" * 100)
        print("Saving")
        model_save_folder = os.path.join(final_save_folder, str(idx))
        os.makedirs(model_save_folder, exist_ok=True)
        with open(os.path.join(model_save_folder, "config"), "w") as f:
            f.write(str(ds))
        with open(os.path.join(model_save_folder, "results"), "w") as f:
            f.write(str(results[-1]))
        models[-1].save_folder = model_save_folder
        models[-1].save_model("final")
        print("Done")
