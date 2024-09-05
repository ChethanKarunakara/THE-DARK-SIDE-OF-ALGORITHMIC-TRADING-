from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('/Users/chethankarunakara/Desktop/final_rental_price_model.pkl')
scaler = joblib.load('/Users/chethankarunakara/Desktop/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    features = [float(x) for x in request.form.values()]
    features_array = np.array(features).reshape(1, -1)  # Reshape for the model
    scaled_features = scaler.transform(features_array)
    
    # Make prediction
    prediction_log = model.predict(scaled_features)
    prediction = np.expm1(prediction_log)  # Convert back from log scale
    
    # Render result in HTML
    return render_template('index.html', prediction_text=f'Predicted Rental Price: ${prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True, port=5004)
