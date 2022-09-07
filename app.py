#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from flask import Flask, request,render_template
import pickle
from collections.abc import Mapping, Sequence

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('flask_app.html')

@app.route('/predict',methods = ['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('flask_app.html',prediction_text = 'The salary will be $ {}'.format(prediction))
    
    

if __name__ == '__main__':
    #from waitress import serve
    app.run(debug = True)




