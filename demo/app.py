import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from openqas.retriever.retriever import WikiRetriever

from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from deeppavlov import build_model, configs

model = build_model(configs.squad.squad, download=True)

wiki_path = "/mnt/data/wiki.db"
tfidf_path = "/mnt/data/wiki.db.tfidf.pkl"

print("Initialising Retriever")
ranker = WikiRetriever(wiki_path)
print("Loading IDs")
ranker.load_ids()
print("Loading TFIDF")
ranker.load(tfidf_path)

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

class QuestionForm(Form):
    quest = TextField('Question:', validators=[validators.required()])
 
@app.route("/", methods=['GET', 'POST'])
def question():
    form = QuestionForm(request.form)
    if (request.method == 'POST'):
        quest = request.form['quest']
        #Put the context string in this

        doc_ids, doc_titles, doc_scores, docs = ranker.find_best_docs([quest], k=10, return_docs=True)

        context = docs
        answer = context[1]
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
