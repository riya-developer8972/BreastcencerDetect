from flask import Blueprint, render_template, request
import joblib
import numpy as np

# Create a Blueprint for routes
main = Blueprint('main', __name__)

# Load the trained model (10 features only)
model = joblib.load("C:/Users/RIYA BANERJEE/Desktop/pythonfile/python.py/finalyearproject/models/model.pkl")

@main.route('/')
def home():
    return render_template("index.html")

@main.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect 10 features from form
        features = [
            float(request.form['mean_radius']),
            float(request.form['mean_texture']),
            float(request.form['mean_perimeter']),
            float(request.form['mean_area']),
            float(request.form['mean_smoothness']),
            float(request.form['mean_compactness']),
            float(request.form['mean_concavity']),
            float(request.form['mean_concave_points']),
            float(request.form['mean_symmetry']),
            float(request.form['mean_fractal_dimension'])
        ]

        # Reshape for prediction
        final_features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_features)[0]

        # Interpret result
        if prediction == 1:
            result = "The patient is likely to have Malignant (Cancer)."
        else:
            result = "The patient is likely to have Benign (No Cancer)."

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        # If something goes wrong, show the error
        return render_template("index.html", prediction_text=f"Error: {str(e)}")
