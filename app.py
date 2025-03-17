import numpy as np
from flask import Flask, request, render_template
import pickle

app= Flask(__name__)
model= pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods= ['POST'])
def predict():
    int_ft =[int(x) for x in request.form.values()]
    ft= [np.array(int_ft)]
    prediction=model.predict(ft)
    result=prediction[0]

    return render_template('index.html', prediction = result)

if __name__=="__main__":
    app.run(debug=True)
                   
