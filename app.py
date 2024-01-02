
from flask import Flask, render_template, request
import pandas as pd
import pickle


app = Flask(__name__)

#Load the trained model
model = pickle.load(open('models/model.pkl', 'rb'))


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(request.form['province']), float(request.form['vehicle_type']),
                    float(request.form['customer_type']), float(request.form['ly_insurer']),
                    float(request.form['pelak_type']), float(request.form['vehicle_age']),
                    float(request.form['life_history']), float(request.form['passenger_history']),
                    float(request.form['financial_history'])]

        # Convert the features to a DataFrame
        input_data = pd.DataFrame([features])

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction=f'Predicted Cluster: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)

        
   
   
    
    
    
        
                       
                       
        