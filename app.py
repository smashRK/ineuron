import pickle
import json
from flask import Flask,render_template,app,request,jsonify,url_for,redirect
import numpy as np
import pandas as pd
import sklearn 
from logging import FileHandler, ERROR


app = Flask(__name__)

def transform_data(new_data):
    sex={'male':1,'female':0}
    new_data.sex= sex[new_data.sex]
    smoker={'yes':1,'no':0}
    new_data.smoker= smoker[new_data.smoker]
    region={'northeast':0,'northwest':1,'southeast':2,'southwest':3}
    new_data.region= region[new_data.region]
    return new_data

regmodel = pickle.load(open('regressor.pkl','rb'))
scaling = pickle.load(open('scale.pkl','rb'))


@app.route('/')
def login():
   return render_template('home.html')




@app.route("/predict",methods=['POST'])
def predict():
    age = float(request.form['age'])
    sex = str(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = float(request.form['children'])
    smoker = str(request.form['smoker'])
    region = str(request.form['region'])

    data = {"age":age,
        "sex":sex,
        "bmi":bmi,
        "children":children,
        "smoker":smoker,
        "region":region}

    pred_data = pd.DataFrame([data])
    new_data = transform_data(pred_data.loc[0])
    pred_value = regmodel.predict(scaling.transform(pd.DataFrame(new_data).transpose()))
    output = str(round(np.float64(pred_value),ndigits=2))
    return render_template("home.html",prediction="Insurance prediction is {}".format(output))
    




if __name__=="__main__":
    app.run(debug=True)

