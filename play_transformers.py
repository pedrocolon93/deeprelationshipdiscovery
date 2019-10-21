from transformers import TFBertModel, BertConfig, BertTokenizer, TFBertEmbeddings

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')
    model.summary()