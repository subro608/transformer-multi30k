import torch
import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import random
import spacy

de = spacy.load('de')
en = spacy.load('en')


def tokenize_de(text):

    return [tok.text for tok in de.tokenizer(text)]


def tokenize_en(text):

    return [tok.text for tok in en.tokenizer(text)]


def data():
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
    TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    input_dim = len(SRC.vocab)
    output_dim = len(TRG.vocab)
    pad_idx = SRC.vocab.stoi['<pad>']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 128

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device)
    return train_iterator, valid_iterator, test_iterator,input_dim,output_dim,pad_idx
