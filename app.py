from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Path to model
MODEL_PATH = os.path.join("models", "final_model.pkl")
model = joblib.load(MODEL_PATH)

# Feature names (same as training)
FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

@app.route('/')
def home():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values
        input_data = [float(request.form[feat]) for feat in FEATURES]

        # Convert to DataFrame
        df = pd.DataFrame([input_data], columns=FEATURES)

        # Predict class
        prediction = model.predict(df)[0]

        # Predict probability (optional, useful for extra info)
        proba = model.predict_proba(df)[0][1]  # Probability of malignant

        # Format result
        result = "🚨 MALIGNANT (Cancer Detected)" if int(prediction) == 1 else "✅ BENIGN (No Cancer)"

        return render_template('result.html', result=result, probability=f"{proba:.2f}")

    except Exception as e:
        return f"Error: {str(e)}"

# Run locally with: python app.py
if __name__ == '__main__':
    app.run(debug=True)
