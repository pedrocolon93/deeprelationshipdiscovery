import argparse

import fasttext
import torch
from tqdm import tqdm

from rcgan_pytorch import RetroCycleGAN
import numpy as np

def get_embeddings(input_vecs,input_words,retrogan_model):
    batch_size = 128
    input_vecs_tensor = torch.tensor(input_vecs).float()
    results = []
    for i in range(0,len(input_words),batch_size):
        batch = input_vecs_tensor[i:i+batch_size,:]
        retrofitted_batch = retrogan_model(batch)
        results.append(retrofitted_batch)
    retrofitted_words = torch.cat(results)
    return retrofitted_words


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("inputtext",
                        help="Text vectors that will be retrofitted",
                        default="cleaned_corpus.txt")
    parser.add_argument("outputtext",default="retrofitted_vecs.txt",
                        help="The output text file")
    parser.add_argument("retroganmodelpath",default="model.bin",
                        help="The output hdf file")

    args = parser.parse_args()
    input_vecs = []
    input_vecs_indexes = []
    print("Loading input vecs")
    with open(args.inputtext) as inputfile:
        for line in tqdm(inputfile):
            line = line.strip()
            if len(line)==0: continue
            input_vecs_indexes.append(line.split()[0])
            v = [float(x) for x in line.split()[1:]]
            v = np.array(v)
            v /= np.linalg.norm(v)
            input_vecs.append(v)
    print("Loaded:",len(input_vecs),"vectors")
    retrogan_location = args.retroganmodelpath
    print("Loading retrogan at",retrogan_location)
    retrogan_model = RetroCycleGAN.load_model(retrogan_location,device="cpu")
    retrogan_model.put_eval()
    retrofitted_vecs = get_embeddings(input_vecs,input_vecs_indexes,retrogan_model)
    retrofitted_vecs = retrofitted_vecs.cpu().detach().numpy()
    with open(args.outputtext, "w") as outfile:
        for idx, vec in enumerate(input_vecs_indexes):
            out = ' '.join([vec]+[str(x) for x in retrofitted_vecs[idx,:]]+["\n"])
            outfile.write(out)
