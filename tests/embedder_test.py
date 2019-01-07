import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from openqas.utils.data import WordEmbeddings
from openqas.reader.embedder import Embedder

def test():
    glove_embeddings = WordEmbeddings()
    glove_embeddings.load_glove("../data/glove.6B.100d.txt")
    embedder = Embedder(glove_embeddings, seq_length=10)
    docs = ["My name is guru", "I love Guru's food"]

    tokens, sequences = embedder.tokenize(docs)
    print(tokens)
    print(sequences)
    vals = embedder.get_embeddings(docs)
    print(vals)
    print(vals.shape)

test()