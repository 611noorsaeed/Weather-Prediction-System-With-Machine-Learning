from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# bringing modle here
model = pickle.load(open('model.pkl','rb'))

# creating flask app
app = Flask(__name__)

# creating paths
@app.route('/')
def index():
    return  render_template("index.html")
@app.route('/predict',methods=['POST'])
def predict():
    precipitation  = float(request.form["precipitation"])
    temp_max = float(request.form["temp_max"])
    temp_min = float(request.form["temp_min"])
    wind = float(request.form["wind"])
    year = int(request.form["year"])
    month =int(request.form["month"])
    day  = int(request.form["day"])

    features = np.array([[precipitation,temp_max,temp_min,wind,year,month,day]])

    pred =  model.predict(features).reshape(1,-1)[0]

    return render_template("index.html", mess = pred)



# python
if __name__ == "__main__":
    app.run(debug=True)