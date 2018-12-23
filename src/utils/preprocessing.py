# Imports
import string
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, StringMatcher

class WordEmbeddings:
    """
    Class to load and handle the GloVe Word Embeddings
    """

    def __init__(self):
        self.vocabulary = set()
        self.word_to_vec = {}
        self.word_to_index = {}
        self.index_to_word = {}

    def load_glove(self, path):
        """
        Loads a pretrained GloVe model
        Expects a path to a GloVe pretrained word embeddings file
        """

        with open(path, 'r') as file:
            for line in file:
                line = line.strip().split()
                self.vocabulary.add(line[0])
                self.word_to_vec[line[0]] = np.array(line[1:], dtype='float64')
    
            for x,y in enumerate(sorted(self.vocabulary)):
                self.word_to_index[y] = x
                self.index_to_word[x] = y
        
    def in_vocab(self, word):
        """
        Checks if a word is present in the vocabulary
        """
        return (word in self.vocabulary)

    def autocorrect(self, wrong_word):
        """
        Attempts to map a wrongly spelt word to the closest one present in the vocabulary.
        THIS IS NOT COSINE SIMILARITY. THIS IS AUTOCORRECT.
        """

        if self.in_vocab(wrong_word):
            return wrong_word

        closest_ratio = 0.0
        closest_word = None
        for word in self.vocabulary:
            if fuzz.ratio(word,wrong_word) > closest_ratio :
                closest_word = word
                closest_ratio = fuzz.ratio(word,wrong_word)
        return closest_word

    def similarity(self, word_1, word_2):
        """
        Returns the cosine similarity of word_1 and word_2
        """
        
        assert (self.in_vocab(word_1) and self.in_vocab(word_2))

        u = self.word_to_vec[word_1]
        v = self.word_to_vec[word_2]

        dot = np.sum(u * v)
        norm_u = np.sqrt(np.sum(u ** 2))
        norm_v = np.sqrt(np.sum(v ** 2))
        cosine_similarity = dot / (norm_u * norm_v)

        return cosine_similarity
