
# coding: utf-8

# In[118]:


from nltk.corpus import wordnet
from fuzzywuzzy import fuzz, StringMatcher
import csv
import nltk
import re
import string
import zipfile
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


# In[106]:


glove_data_file ='glove.6B.100d.txt'

def loadGloveModel(gloveFile,words):   
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        if(word in words):
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    return model


# In[107]:


def cleaned_words(sentance):
    words = re.split(r'\W+', sentance)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    final_words=[]
    for word in stripped:
        if(word is not ''):
            final_words.append(word)
    return final_words


# In[108]:


def pos_tagging_list(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


# In[109]:


def ner(doc):
    tokenized_doc = nltk.word_tokenize(doc)
    tagged_s = nltk.pos_tag(tokenized_doc)
    chunked_s = nltk.ne_chunk(tagged_s)
    named_entities={}
    for tree in chunked_s:
        if hasattr(tree,'label'):
            entity_name = ' '.join(c[0] for c in tree.leaves())
            entity_type = tree.label()
            if entity_name not in named_entities.keys():
                named_entities[entity_name]=entity_type
    return named_entities


# In[110]:


def get_word_synonyms_from_sent(word, sent):
    word_synonyms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemma_names():
            if lemma in sent and lemma != word:
                word_synonyms.append(lemma)
    return word_synonyms


# In[111]:


def term_frequency(words):
    word_counts = {}
    for word in words:
        if(word in word_counts.keys()):
            word_counts[word]=word_counts[word]+1
        else:
            word_counts[word]=1
    for key,value in word_counts.items():
        word_counts[key]= value/len(words)
    return word_counts


# In[112]:


def exact_match(word,question):
    ans = []
    synonym = get_word_synonyms_from_sent(word,question) 
    if(word in question):
        ans.append(1)
    else:
        ans.append(0)
        
    if(word.lower() in question):
        ans.append(1)
    else:
        ans.append(0)
        
    if(len(synonym) >0 ):
        ans.append(1)
    else:
        ans.append(0)
    return ans


# In[119]:


def similarity(word,model_words):
    most_sim_count =0
    most_sim_word = word
    for test_words in model_words.keys():
        if fuzz.ratio(word,test_words)>most_sim_count :
             most_sim_word = test_words
    return most_sim_word
        


# In[120]:


def paragraph_encoding(paragraph,question):
    para_vector=[]
    tokens = cleaned_words(paragraph)
    tokens_question = cleaned_words(question)
    model = loadGloveModel(glove_data_file,tokens)
    pos_tags = pos_tagging_list(" ".join(tokens))
    named_entity = ner(" ".join(tokens))
    word_freq = term_frequency(tokens)
    word_count =0
    for word in tokens:
        #Adding word embeddings
        if word in model.keys():
            word_embedding = model[word].tolist()
        else:
            most_similar_word = similarity(word,model)
            word_embedding = model[most_similar_word].tolist()
        exact_match_vector = exact_match(word,tokens_question)
        #Adding matched words
        for binary in exact_match_vector:
            word_embedding.append(binary)
        #Add manual_feature
        #Add pos tags
        word_embedding.append(pos_tags[word_count][1])
        #Add term frequency
        word_embedding.append(word_freq[word])
        #Add named entity recognition
        if word in named_entity:
            word_embedding.append(named_entity[word])
        else:
            word_embedding.append('O')
        para_vector.append(word_embedding)
    return para_vector
        
        
        


# In[121]:


para= "hello Andrew, i am very happy"
q = "what is glad"
paragraph_encoding(para,q)


# In[ ]:




