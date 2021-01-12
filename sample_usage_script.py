import fasttext
import torch

from rcgan_pytorch import RetroCycleGAN


def get_embeddings(words,fasttext_model,retrogan_model):
    retrofitted_words = []
    for word in words:
        original_embeddings = fasttext_model.get_word_vector(word)
        tensor_original_embeddings = torch.tensor(original_embeddings)
        retrofitted_embedding = retrogan_model(tensor_original_embeddings)
        retrofitted_words.append(retrofitted_embedding.cpu().detach().numpy())
    return retrofitted_words


if __name__ == '__main__':
    print("Original words")
    words = ["cat","dog","human","potato"]
    fasttext_location = "fasttext_model/cc.en.300.bin"
    fasttext_model = fasttext.load_model(fasttext_location)
    retrogan_location = "oov_test_1_0/retrogan_1_0/finalcomplete.bin"
    retrogan_model = RetroCycleGAN.load_model(retrogan_location,device="cpu")
    retrofitted_words = get_embeddings(words,fasttext_model,retrogan_model)
    print("Retrofitted vectors")
    print(retrofitted_words)