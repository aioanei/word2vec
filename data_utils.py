import numpy as np
import urllib.request
import os
import zipfile
class TextProcessor:
    def __init__(self, window_size=5):
        self.window = window_size
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
    def download_text (self):
        # using this dataset for word embedding training because 
        # it doesn't require any preprocessing
        url = "http://mattmahoney.net/dc/text8.zip"
        if not os.path.exists("text8"):
            print("Downloading text8")
            urllib.request.urlretrieve(url, "text8.zip")
            with zipfile.ZipFile("text8.zip", 'r') as z:
                z.extractall()
        with open("text8", 'r') as f:
            text = f.read().split()
        return text
    def build_vocab(self, text):
        # sorting for better indexing and reproducibility
        unique_words = sorted(list(set(text)))
        self.vocab_size = len(unique_words)
        self.word2idx = {word: idx for idx, word in enumerate(unique_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(unique_words)
        return self.vocab_size
    def get_batches(self, tokens):
        token_ids = [self.word2idx[token] for token in tokens]
        for i in range (len(token_ids)):
            start = max(0, i - self.window)
            end = min(len(token_ids), i + self.window + 1)
            context = token_ids[start:i] + token_ids[i+1:end]
            # yielding instead of returning allows to generate batches 
            # on the fly without storing them all in memory
            yield context, token_ids[i]