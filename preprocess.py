import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import string as st
import re
from IPython.display import display
import nltk
from nltk import PorterStemmer, WordNetLemmatizer
#nltk.download('all')
import os

class preprocess:
    def __init__(self, dataset):
        self.dataset = dataset.copy()

    def preprocess_col(self,collumn):
        self.dataset[collumn] = self.dataset[collumn].apply(lambda x: self.apply(x))
        return self.dataset

    def apply(self, text):
        text = self._remove_punct(text)
        tokens = self._tokenize(text)
        larger_tokens_only = self._remove_small_words(tokens)
        no_stop_words = self._remove_stopwords(larger_tokens_only)
        #stem_words = self._stemming(no_stop_words)
        # print(stem_words)
        # maybe don't apply stemming and lemmatization
        lemma_words = self._lemmatize(no_stop_words)
        # there is no need to do pos_tag
        #pos_tag = _get_pos_tag(lemma_words)
        clean_text = self._return_sentences(lemma_words)
        return clean_text

    def _remove_punct(self, text):
        return ("".join([ch for ch in text if ch not in st.punctuation]))

    def _tokenize(self, text):
        text = re.split('\s+', text)
        return [x.lower() for x in text]

    def _remove_small_words(self, text):
        return [x for x in text if len(x) > 3]

    def _remove_stopwords(self, text):
        return [word for word in text if word not in nltk.corpus.stopwords.words('english')]

    # Apply stemming to get root words
    def _stemming(self, text):
        ps = PorterStemmer()
        return [ps.stem(word) for word in text]

    # Apply lemmatization on tokens
    def _lemmatize(self, text):
        word_net = WordNetLemmatizer()
        return [word_net.lemmatize(word) for word in text]

    def _get_pos_tag(self, tokenized_sentence):
        return nltk.pos_tag(tokenized_sentence)

    # Create sentences to get clean text as input for vectors
    def _return_sentences(self, tokens):
        return " ".join([word for word in tokens])



