import numpy as np
import pandas as pd
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Input, Bidirectional
from keras.models import Model
from .embedder import Embedder
from ..utils.data import WordEmbeddings

def QuestionEncoder(lstm_nodes=128, seq_length=1000, embeddings_dim=100):
    inputs = Input((seq_length, embeddings_dim))
    hidden_layer = Bidirectional(LSTM(lstm_nodes, return_state=True, return_sequences=False))(inputs)
    model = Model(inputs=inputs, outputs=hidden_layer)

    return model

def ParagraphEncoder(lstm_nodes=256, seq_length=1000, embeddings_dim=100):
    inputs = Input((seq_length, embeddings_dim))
    hidden_layer = Bidirectional(LSTM(lstm_nodes, return_state=True, return_sequences=False))(inputs)
    model = Model(inputs=inputs, outputs=hidden_layer)

    return model

class Reader:

    def __init__(self, path, seq_length=1000, stopwords='default'):
        self.glove_embeddings = WordEmbeddings()
        self.glove_embeddings.load_glove(path)
        
        assert (len(self.glove_embeddings.word_to_index) != 0)

        self.embedder = Embedder(self.glove_embeddings, seq_length=seq_length, stopwords=stopwords)

        self.question_encoder = QuestionEncoder()
        self.paragraph_encoder = ParagraphEncoder()
    
    def predict(self, paragraphs, questions):
        q_input = self.embedder.get_embeddings(questions)
        p_input = self.embedder.get_embeddings(paragraphs)

        q_prediction = self.question_encoder.predict(q_input)
        p_prediction = self.paragraph_encoder.predict(p_input)

        return p_prediction, q_prediction