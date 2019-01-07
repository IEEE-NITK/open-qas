#ner test

def ner(text):
    tokenized_doc = nltk.word_tokenize(doc)
    tagged_s = nltk.pos_tag(tokenized_doc)
    chunked_s = nltk.ne_chunk(tagged_s)
    print(chunked_s, type(chunked_s))
    named_entities={}
    for tree in chunked_s:
        if hasattr(tree,'label'):
            entity_name = ' '.join(c[0] for c in tree.leaves())
            entity_type = tree.label()
            if entity_name not in named_entities.keys():
                named_entities[entity_name]=entity_type
    return named_entities