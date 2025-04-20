from flask import Flask, render_template, request
import pandas as pd
from joblib import load
import os

app = Flask(__name__)

# Load model and binarizer
model = load('model/disease_predictor_model.pkl')
binarizer = load('model/symptom_binarizer.pkl')

# Load precautions data
precaution_df = pd.read_csv('datasets/disease_precaution.csv')

# Extract all symptoms from the MultiLabelBinarizer
all_symptoms = binarizer.classes_

@app.route('/')
def index():
    return render_template('index.html', symptoms=all_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')

    if not selected_symptoms:
        return render_template('index.html', symptoms=all_symptoms, message="❌ Please select at least 1 symptom.")

    # Convert symptoms into binary input for model
    input_data = binarizer.transform([selected_symptoms])
    prediction = model.predict(input_data)[0]

    # Get precautions for predicted disease
    precautions = precaution_df[precaution_df['Disease'].str.lower() == prediction.lower()]
    precaution_list = []
    if not precautions.empty:
        precaution_list = [
            precautions.iloc[0]['Precaution_1'],
            precautions.iloc[0]['Precaution_2'],
            precautions.iloc[0]['Precaution_3'],
            precautions.iloc[0]['Precaution_4']
        ]

    return render_template(
        'index.html',
        symptoms=all_symptoms,
        prediction=prediction,
        selected=selected_symptoms,
        precautions=precaution_list
    )

if __name__ == '__main__':
    app.run(debug=True)
