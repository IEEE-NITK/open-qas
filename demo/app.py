'''
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
'''
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn as sk
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from deeppavlov import build_model, configs

model = build_model(configs.squad.squad, download=True)
'''
wiki_path = "/mnt/data/wiki.db"
tfidf_path = "/mnt/data/wiki.db.tfidf.pkl"

print("Initialising Retriever")
ranker = WikiRetriever(wiki_path)
print("Loading IDs")
ranker.load_ids()
print("Loading TFIDF")
ranker.load(tfidf_path)
'''
# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

class QuestionForm(Form):
    quest = TextField('Question:', validators=[validators.required()])
 
def get_contexts(question):
    docs = pd.read_json('1000-wiki-edu-parsed.json', lines=True)
    STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
    }
    vectorizer = CountVectorizer(ngram_range=(1,2), strip_accents='unicode', stop_words=STOPWORDS)
    doc_counts = vectorizer.fit_transform(docs.text)
    features = vectorizer.get_feature_names()
    h_vectorizer = HashingVectorizer(ngram_range=(1,2), strip_accents='unicode', stop_words=STOPWORDS)
    h_doc_counts = h_vectorizer.transform(docs.text)
    transformer = TfidfTransformer()
    doc_tfidf = transformer.fit_transform(h_doc_counts)
    query = [question]
    query_count = h_vectorizer.transform(query)
    query_tfidf = transformer.transform(query_count)
    res = np.dot(query_tfidf, np.transpose(doc_tfidf))
    res.toarray()
    k = min(6,len(res.toarray()))
    ind = np.argpartition(res.data, -k)[-k:]
    ind_sort = ind[np.argsort(-res.data[ind])]
    best_doc_indices = res.indices[ind_sort]
    contexts = []
    for index in best_doc_indices:
        contexts.append(docs.text[index])
    return contexts


@app.route("/", methods=['GET', 'POST'])
def question():
    form = QuestionForm(request.form)
    if (request.method == 'POST'):
        quest = request.form['quest']
        #Put the context string in this
        '''
        doc_ids, doc_titles, doc_scores, docs = ranker.find_best_docs([quest], k=10, return_docs=True)

        context = docs
        '''
        answe = "None"
        context = get_contexts(quest)
        if(len(context) == 0):
            answer = "Answer not found"
        else:
            score = -1
            for cont in context:
                answer_array = model([cont],[quest])
                ans_score =  answer_array[2][0]
                print(ans_score)
                if(ans_score > score ):
                    answer = answer_array[0][0]
                    score = ans_score
        
    if form.validate():
        flash(answer)
    else:
        flash('Please enter a valid question')
    
    return render_template('main.html', form=form)
    

if __name__ == "__main__":
    app.run()
