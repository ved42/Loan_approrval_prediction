from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)


scaler=pickle.load(open("/config/workspace/model/scaler.pkl", "rb"))
model = pickle.load(open("/config/workspace/model/model_prediction.pkl", "rb"))

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        Gender=int(request.form.get("Gender"))
        Married = int(request.form.get('Married'))
        Dependents = int(request.form.get('Dependents'))
        Education = int(request.form.get('Education'))
        Self_Employed = int(request.form.get('Self_Employed'))
        ApplicantIncome = int(request.form.get('ApplicantIncome'))
        CoapplicantIncome = int(request.form.get('CoapplicantIncome'))
        LoanAmount = int(request.form.get('LoanAmount'))
        Loan_Amount_Term = int(request.form.get('Loan_Amount_Term'))
        Credit_History = float(request.form.get('Credit_History'))
        Property_Area = int(request.form.get('Property_Area'))
        

        new_data=scaler.transform([[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = ' Approved'
        else:
            result ='not approved'
            
        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")