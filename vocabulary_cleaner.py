import pandas as pd
import gc

def cleanup_vocabulary_nb_based(numberbatch_file_loc,target_file_loc,replace=False):
    print('Loading nb and taking out the indexes')
    numberbatch_voc = pd.read_hdf(numberbatch_file_loc,'mat')
    nb_vocabulary = numberbatch_voc.index
    del numberbatch_voc
    gc.collect()
    print('Done\nLoading target vocabulary and intersecting indexes')
    target_voc = pd.read_hdf(target_file_loc,'mat')
    clean_voc = target_voc.loc[target_voc.index.intersection(nb_vocabulary)]
    print('Saving. Replace:',replace)
    fname = None
    if replace:
        fname=target_file_loc
        clean_voc.to_hdf(fname,'mat')
    else:
        fname=target_file_loc+'clean'
        clean_voc.to_hdf(fname,'mat')
    return fname


if __name__ == '__main__':
    numberbatch_file_loc = 'retrogan/mini.h5'
    target_file_loc = 'trained_models/retroembeddings/2019-04-0813:03:02.430691/retroembeddings.h5'
    cleanup_vocabulary_nb_based(numberbatch_file_loc,target_file_loc)