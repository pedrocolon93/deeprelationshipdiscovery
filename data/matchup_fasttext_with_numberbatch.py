'''
Match up FastText embeddings with NumberBatch embeddings
'''

from tqdm import tqdm


# if __name__ == '__main__':
#     ## Read all FastText data as a Python dict
#     fasttext_tokens = []
#     fasttext_data = []
#     numberbatch_tokens = []
#     with open('cc.en.300.vec', 'r') as f:
#         fasttext_tokens = [
#             x.strip().split(' ')[0]
#             for x in f.readlines()[1:]
#         ]
#         fasttext_data = [
#             x.strip().split(' ')
#             for x in f.readlines()[1:]
#         ]
#     ## Read all NumberBatch data as a Python dict
#     with open('numberbatch-en-19.08.txt', 'r') as f:
#         numberbatch_tokens = [
#             x.strip().split(' ')[0].replace("/c/en/", "")
#             for x in f.readlines()[1:]
#         ]
#     found = set(numberbatch_tokens).intersection(set(fasttext_data))
#     # ## Separate the data
#     # fasttext_tokens = list(fasttext_data.keys())
#     # fasttext_embeds = list(fasttext_data.values())
#     # numberbatch_tokens = list(numberbatch_data.keys())
#     # numberbatch_embeds = list(numberbatch_data.values())
#     ## Match up FastText to NumberBatch
#     matched_fasttext_data = []
#     for i in tqdm(range(len(fasttext_data))):
#         ft_datum = fasttext_data[i]
#         ft_token = ft_datum[0]
#         if ft_token in found:
#             # matched_fasttext_data[ft_token] = ft_embed
#             matched_fasttext_data.append(ft_datum)
#     ## Write matched-up FastText to a new file
#     matched_fasttext_data = [
#         ' '.join(x) + '\n' for x in tqdm(matched_fasttext_data)
#     ]
#     with open('fasttext-en-20.09.txt', 'w') as f:
#         f.writelines(matched_fasttext_data)


# ---------- OLD CODE BELOW ----------

if __name__ == '__main__':

    # ## Read all FastText data as a Python dict
    # fasttext_tokens = []
    # fasttext_data = []
    # with open('cc.en.300.vec', 'r') as f:
    #     for x in tqdm(f):
    #         x = x.strip().split(' ')
    #         fasttext_tokens.append(x[0])
    #         fasttext_data.append(x)
    # ## Read all NumberBatch data as a Python dict
    # numberbatch_tokens = []
    # with open('numberbatch-en-19.08.txt', 'r') as f:
    #     for x in tqdm(f):
    #         numberbatch_tokens.append(x.strip().split(' ')[0].replace("/c/en/", ""))

    ## Read all FastText data as a Python dict
    # fasttext_data = []
    # numberbatch_tokens = []
    with open('cc.en.300.vec', 'r') as f:
        fasttext_data = [
            x.strip().split(' ')
            for x in tqdm(f.readlines()[1:])
        ]

    ## Read all NumberBatch data as a Python dict
    with open('numberbatch-en-19.08.txt', 'r') as f:
        numberbatch_tokens = [
            x.strip().split(' ')[0].replace('/c/en/', '')
            for x in tqdm(f.readlines()[1:])
        ]

    # ## Separate the data
    # fasttext_tokens = list(fasttext_data.keys())
    # fasttext_embeds = list(fasttext_data.values())
    # numberbatch_tokens = list(numberbatch_data.keys())
    # numberbatch_embeds = list(numberbatch_data.values())

    # ## Match up FastText to NumberBatch
    # nbmap = [True for x in numberbatch_tokens]
    # matched_fasttext_data = []
    # for i in tqdm(range(len(fasttext_data))):
    #     ft_datum = fasttext_data[i]
    #     ft_token = ft_datum[0]
    #     try:
    #         if nbmap[ft_token]:
    #             matched_fasttext_data.append(ft_datum)
    #     except Exception or KeyError:
    #         continue

    # matched_fasttext_data = []
    # # for i in tqdm(range(len(fasttext_data))):
    # #     ft_datum = fasttext_data[i]
    # #     ft_token = ft_datum[0]
    # #     if ft_token in numberbatch_tokens:
    # #         matched_fasttext_data.append(ft_datum)
    # for nb_token in tqdm(numberbatch_tokens):
    #     if nb_token in fasttext_tokens:
    #         idx = fasttext_tokens.index(nb_token)
    #         matched_fasttext_data.append(fasttext_data[idx])

    ## Match up FastText to NumberBatch
    matched_fasttext_data = []
    for i in tqdm(range(len(fasttext_data))):
        ft_datum = fasttext_data[i]
        ft_token = ft_datum[0]
        if ft_token in numberbatch_tokens:
            # matched_fasttext_data[ft_token] = ft_embed
            matched_fasttext_data.append(ft_datum)

    ## Write matched-up FastText to a new file
    matched_fasttext_data = [
        ' '.join(x) + '\n' for x in tqdm(matched_fasttext_data)
    ]
    with open('fasttext-en-20.09.txt', 'w') as f:
        f.writelines(matched_fasttext_data)
