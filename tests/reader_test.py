import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from openqas.utils.data import WordEmbeddings
from openqas.reader.embedder import Embedder
from openqas.reader.reader import Reader

def test():
    docs = ["My name is guru. I love Guru's food. I am from India."]
    ques = ["What is my name", "What does Guru love"]

    reader = Reader(path="../data/glove.6B.100d.txt", seq_length=15)
    p_pre, q_pre = reader.encode(docs, ques)

    print(len(p_pre))
    for i in range(0, len(p_pre)):
        print(p_pre[i].shape)
    
    print(len(q_pre))
    for i in range(0, len(q_pre)):
        print(q_pre[i].shape)
    
test()