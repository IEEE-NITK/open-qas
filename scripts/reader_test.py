import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from openqas.utils.data import WordEmbeddings
from openqas.reader.embedder import Embedder
from openqas.reader.reader import Reader

def test():
    """docs = ["My name is guru. I love Guru's food. I am from India."]
    ques = ["What is my name", "What does Guru love"]

    reader = Reader(path="../data/glove.6B.100d.txt", seq_length=15)
    p_pre, q_pre = reader.encode(docs, ques)

    print(len(p_pre))
    for i in range(0, len(p_pre)):
        print(p_pre[i].shape)
    
    print(len(q_pre))
    for i in range(0, len(q_pre)):
        print(q_pre[i].shape)
    """
    
    reader_test = Reader('../data/glove.6B.100d.txt')
    print(reader_test.encode("Assistive technology\n\nAssistive technology is an umbrella term that includes assistive, adaptive, and rehabilitative devices for people with disabilities while also including the process used in selecting, locating, and using them. People who have disabilities often have difficulty performing activities of daily living (ADLs) independently, or even with assistance. ADLs are self-care activities that include toileting, mobility (ambulation), eating, bathing, dressing and grooming. Assistive technology can ameliorate the effects of disabilities that limit the ability to perform ADLs","What is assistive technology ?"))
    
test()