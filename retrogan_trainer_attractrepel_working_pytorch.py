from __future__ import print_function, division

import argparse
import os
import shutil

import tensorflow as tf
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# from tensorflow.python.framework.ops import disable_eager_execution
import torch

import tools
from rcgan_pytorch import RetroCycleGAN


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.compat.v1.summary.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--disable_eager_execution', type=str2bool, default=True,
                        help='Whether to disable eager execution or not')
    parser.add_argument('--logdir', type=str, default="logs/",
                        help='Directory where tensorboard logging will go to')
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Whether to use fp16 calculation speed up.")
    parser.add_argument("--savepostfix", default="retrogan",
                        help="A postfix to the saved name of the model")
    parser.add_argument("original", default="ft_nb_seen.h5",
                        help="The HDF file of the original embeddings.  "
                             "Usually these are fasttext or glove embeddings.")
    parser.add_argument("retrofitted", default="nb_retrofitted_ook_attractrepel.h5",
                        help="The HDF file of the retrofitted embeddings.  "
                             "Usually these are retrofitted counterparts to the original embeddings "
                             "(e.g. after attract-repel of the embeddings found in original)")
    parser.add_argument("model_name", default="nb_ook",
                        help="The location of the hdf files supplied as original and retrofitted")
    parser.add_argument("save_folder",
                        help="Location where the model will be saved to")
    parser.add_argument("--checkpoint_prefix", default="checkpoint_",
                        help="The prefix for the checkpoints generated while training")
    parser.add_argument("--epochs_per_checkpoint", default=4, type=int,
                        help="The amount of epochs per checkpoint saved.")
    parser.add_argument("--epochs", default=50, type=int, help="Amount of epochs")
    parser.add_argument("--iters", default=None, type=int, help="Amount of iterations, overrides epochs")

    parser.add_argument("--g_lr", default=0.00005, type=float, help="Generator learning rate")
    parser.add_argument("--d_lr", default=0.0001, type=float, help="Discriminator learning rate")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--dis_train_amount", default=3, type=int,
                        help="The amount of times to run a discriminator through the batch")



    parser.add_argument("--one_way_mm", type=str2bool, default=True,
                        help="Whether to use fp16 calculation speed up.")
    parser.add_argument("--cycle_mm", type=str2bool, default=True,
                        help="Whether to use fp16 calculation speed up.")
    parser.add_argument("--cycle_dis", type=str2bool, default=True,
                        help="Whether to use fp16 calculation speed up.")
    parser.add_argument("--id_loss", type=str2bool, default=True,
                        help="Whether to use fp16 calculation speed up.")
    args = parser.parse_args()

    # print("Configuring GPUs to use only needed memory")
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)
    # print("Done!")

    print("Clearing logs!!!")
    print("*" * 100)
    logdir = args.logdir
    shutil.rmtree(logdir, ignore_errors=True)
    print("Done!")
    print("*" * 100)
    # print("Disabling eager execution for speed...")
    # if args.disable_eager_execution:
    #     disable_eager_execution()
    print("Enabling mixed precision for speed...")
    # if args.fp16:
    #     policy = mixed_precision.Policy('mixed_float16')
    #     mixed_precision.set_policy(policy)
    #     print('Compute dtype: %s' % policy.compute_dtype)
    #     print('Variable dtype: %s' % policy.variable_dtype)

    tools.dimensionality = 300
    postfix = args.savepostfix

    test_ds = [
        {
            "original": args.original,  # "ft_nb_seen.h5",
            "retrofitted": args.retrofitted,  # "nb_retrofitted_ook_attractrepel.h5",
            "model_name": args.model_name,  # "Data/nb_ook/",
        },
    ]
    print("Testing")
    print(args)
    print("Checking that everything exists")
    for ds in test_ds:
        a = os.path.exists(os.path.join(ds["original"]))
        b = os.path.exists(os.path.join(ds["retrofitted"]))
        print(a, b)
        if not a:

            raise FileNotFoundError("Original file not found "+str(os.path.join(ds["original"])))
        if not b:
            raise FileNotFoundError("Retrofitted file not found"+str(os.path.join(ds["retrofitted"])))

    models = []
    results = []
    save_folder = args.save_folder
    for idx, ds in enumerate(test_ds):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        print("Training")
        print(ds)
        bs = args.batch_size
        rcgan = RetroCycleGAN(
            save_folder=args.save_folder,
            generator_lr=args.g_lr,
            discriminator_lr=args.d_lr,
            one_way_mm=args.one_way_mm,
            cycle_mm=args.cycle_mm,
            cycle_dis=args.cycle_dis,
            id_loss=args.id_loss,
            name=args.model_name
        )
        # rcgan.load_weights(preface="final", folder="trained_models/retrogans/ft_nb_retrogan/")
        rcgan.to_device("cpu")
        sl = tools.test_sem(rcgan.g_AB, ds, dataset_location="testing/SimLex-999.txt",
                            fast_text_location="fasttext_model/cc.en.300.bin", prefix="en_", pt=True)[0]
        cuda = torch.cuda.is_available()
        rcgan.to_device("cuda" if cuda else "cpu")

        # continue
        rcgan.save_model("test")
        models.append(rcgan)
        ds_res = rcgan.train(epochs=args.epochs, batch_size=bs, dataset=ds,
                             save_folder=rcgan.save_folder,
                             epochs_per_checkpoint=args.epochs_per_checkpoint,
                             dis_train_amount=args.dis_train_amount,
                             iters=args.iters)
        results.append(ds_res)
        print("*" * 100)
        print(ds, results[-1])
        print("*" * 100)
        print("Saving")
        model_save_folder = os.path.join(args.save_folder, str(idx))
        os.makedirs(model_save_folder, exist_ok=True)
        with open(os.path.join(model_save_folder, "config"), "w") as f:
            f.write(str(ds))
        with open(os.path.join(model_save_folder, "results"), "w") as f:
            f.write(str(results[-1]))
        models[-1].save_folder = model_save_folder
        models[-1].save_model("final")
        print("Done")
