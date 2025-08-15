import numpy as np
from flask import Flask, render_template, request
import pickle
import math

app = Flask(__name__)

# Load your model
model2 = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form and convert to integers
    int_features = [int(i) for i in request.form.values()]
    
    # Convert to numpy array and reshape for prediction
    final_features = np.array(int_features).reshape(1, -1)

    # Predict using the model
    prediction = model2.predict(final_features)
    output = round(prediction[0], 2)

    return render_template(
        'index.html',
        predict_text=f'Number of weekly riders: {math.floor(output)}'
    )

if __name__ == "__main__":
    app.run(debug=True)
