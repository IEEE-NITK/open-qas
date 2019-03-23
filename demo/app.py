from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from deeppavlov import build_model, configs
model = build_model(configs.squad.squad, download=True)

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
        context = ["Anumeha is a gt","Anumeha is very nice"]
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