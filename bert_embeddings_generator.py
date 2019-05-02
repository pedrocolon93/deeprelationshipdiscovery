from bert_serving.client import BertClient
import pandas as pd
bc = BertClient()
ft_vocab = pd.read_hdf('fasttext_model/unfitted.hd5','mat')
clean_idxs = [idx.replace('/c/en/','') for idx in ft_vocab.index]
clean_idxs = clean_idxs[1:]
res = bc.encode(clean_idxs)
bert_embeddings = pd.DataFrame(res,ft_vocab.index[1:])
print(bert_embeddings)
bert_embeddings.to_hdf('./bert_unfitted','mat')
print(res)