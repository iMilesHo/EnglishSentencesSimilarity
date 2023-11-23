import pandas as pd
import numpy as np
import random as rnd
import torch
import nltk
from nltk.data import find
import ssl
from collections import defaultdict
from sklearn.model_selection import train_test_split


# avoid SSL certificate error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# check if punkt tokenizer models are downloaded
try:
    find('tokenizers/punkt')
    print("Punkt Tokenizer Models are already downloaded.")
except LookupError:
    print("Punkt Tokenizer Models not found. Downloading them...")
    nltk.download('punkt')

def split_data(Q1, Q2, split_ratio=0.8):
    """ Split the data into training and validation sets. """
    cut_off = int(len(Q1) * split_ratio)
    train_Q1, train_Q2 = Q1[:cut_off], Q2[:cut_off]
    val_Q1, val_Q2 = Q1[cut_off:], Q2[cut_off:]
    return train_Q1, train_Q2, val_Q1, val_Q2

def tokens_to_sentence(tokenized_sentence, inverse_vocab):
    return ' '.join([inverse_vocab[token] for token in tokenized_sentence if token in inverse_vocab])

if __name__ == '__main__':
    pass