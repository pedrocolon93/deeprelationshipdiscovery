import argparse
import pandas as pd

def load_text_embeddings(original):
    vecs = []
    idxs = []
    with open(original) as f:
        for line in f:
            line = line.strip()
            ls = line.split()

            if line != "" and len(ls) > 2:
                vecs.append([float(x) for x in ls[1:]])
                idxs.append(ls[0])
    return pd.DataFrame(index=idxs,data=vecs)

def print_to_text(X_train,outfile):
    with open(outfile,"w") as out:
        for word in X_train.index:
            out.write(" ".join([word]+[str(x) for x in X_train.loc[word,:]]+["\n"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("seen", default="ft_nb_seen.txt",
                        help="The HDF file of the original embeddings.  "
                             "Usually these are fasttext or glove embeddings.")
    parser.add_argument("adjusted", default="nb_retrofitted_ook_attractrepel.txt",
                        help="The HDF file of the retrofitted embeddings.  "
                             "Usually these are retrofitted counterparts to the original embeddings "
                             "(e.g. after attract-repel of the embeddings found in original)")
    parser.add_argument("--prefix",default="fix")
    args = parser.parse_args()

    o = load_text_embeddings(args.seen)
    r = load_text_embeddings(args.adjusted)
    print("Old shapes")
    print(o.shape)
    print(r.shape)
    cns = r.index.intersection(o.index)
    print("Intersecting on",len(cns))
    X_train = o.loc[cns, :]
    Y_train = r.loc[cns, :]
    print("New shapes")
    print(X_train.shape)
    print(Y_train.shape)
    print_to_text(X_train,args.prefix+args.seen.split("/")[-1])
    print_to_text(Y_train,args.prefix+args.adjusted.split("/")[-1])

