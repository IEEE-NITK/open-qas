'''
    Question encoding for Document Reader
'''

import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Input
from keras.models import Model

MAX_WORDS = 50

def GetGlove(glove_file):
    with open(glove_file, 'r') as file:
        words = set()
        word_to_vec, word_to_index, index_to_word = {},{},{}

        for line in file:
            line = line.strip().split()
            words.add(line[0])
            word_to_vec[line[0]] = np.array(line[1:], dtype='float64')
    
    for x,y in enumerate(sorted(words)):
        word_to_index[y] = x
        index_to_word[x] = y
        
    return {'word_to_vec': word_to_vec, 'word_to_index': word_to_index, 'index_to_word': index_to_word}

def QuestionIndices(Q, word_to_vec, word_to_index):
    x_index = np.zeros((len(Q), MAX_WORDS)) 

    for i in range(len(Q)):
        words = Q[i].lower().strip().split()
        j = 0
        for w in words:
            x_index[i,j] = word_to_index[w]
            j += 1
    
    return x_index
    
def EmbeddingLayer(word_to_vec, word_to_index):

    EMBED_DIM = word_to_vec['the'].shape[0]
    VOCAB_SIZE = len(word_to_index)+1

    embed_matrix = np.zeros((VOCAB_SIZE, EMBED_DIM))
    for word, index in word_to_index.items():
        embed_matrix[index] = word_to_vec[word]
    
    embedded_layer = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, trainable = False, input_length=MAX_WORDS)
    embedded_layer.build((None, ))
    embedded_layer.set_weights([embed_matrix])
    
    return embedded_layer

def Encode(word_to_vec, word_to_index):
    input = Input((50, ), dtype='int32')
    
    embedding_layer = EmbeddingLayer(word_to_vec, word_to_index)
    embeddings = embedding_layer(input)
    x = LSTM(128, return_sequences=False)(embeddings)
    
    model = Model(inputs=input, outputs=x)
    
    return model

def main():
    print('glove')
    params = GetGlove('glove.6B.50d.txt')
    word_to_vec, word_to_index, index_to_word = params['word_to_vec'],params['word_to_index'],params['index_to_word']
   
    print('indices')
    ind = QuestionIndices(['Which is the tallest building'],word_to_vec,word_to_index)
   
    print('model')
    model = Encode(word_to_vec,word_to_index)
    print(model.predict(ind))

if __name__ == '__main__':
    main()
        