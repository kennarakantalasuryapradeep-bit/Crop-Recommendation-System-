from flask import Flask, request, render_template
import numpy as np
import pickle

# Load model
model = pickle.load(open('RandomForest.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    ls = np.array(feature_list).reshape(1, -1)

    prediction = model.predict(ls)

    crop_list = ['rice', 'maize', 'jute', 'cotton', 'coconut', 'papaya', 'orange', 
 'apple', 'muskmelon', 'watermelon', 'grapes', 'mango', 'banana', 
 'pomegranate', 'lentil', 'blackgram', 'mungbean', 'mothbeans', 
 'pigeonpeas', 'kidneybeans', 'chickpea', 'coffee']

    if prediction[0] in crop_list:
        crop = prediction[0]
        result = f"{crop} is the best crop to be cultivated right there"
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
