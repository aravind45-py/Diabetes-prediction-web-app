# main application server

from flask import Flask,render_template,redirect,url_for,request
from sklearn.preprocessing import StandardScaler
import joblib as jb
import os
model = jb.load('model.sav')

scaler = jb.load('scaler.sav')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/check_diabetes')
def Check_diabetes():
    return render_template('diabetes.html')
@app.route('/result',methods=['GET','POST'])
def result():
        a1 = int(request.args.get('preg'))
        a2 = int(request.args.get('Glucose'))
        a3 = int(request.args.get('BloodPressure'))
        a4 = int(request.args.get('SkinThickness'))
        a5 = int(request.args.get('Insulin'))
        a6 = float(request.args.get('BMI'))
        a7 = float(request.args.get('DiabetesPedigreeFunction'))
        a8 = int(request.args.get('Age'))
        res = model.predict_proba(scaler.transform([[a1,a2,a3,a4,a5,a6,a7,a8]]))[0][-1]
        res = res*100
        return render_template('result.html',res=res)

if __name__=='__main__':
    app.run(debug=True)