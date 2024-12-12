from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import shap

app = Flask(__name__)

# ===============================
# Load model and scaler
# ===============================
model = joblib.load("best_rf_model.pkl")  # Replace with the actual model path
scaler = joblib.load("scaler.pkl")        # Replace with the actual scaler path
encoder = joblib.load("college_encoder.pkl")  # Replace with the actual encoder path

# Define feature names
features = ['College', 'MPG', 'PPG', 'RPG', 'APG']

# ===============================
# Home route
# ===============================
@app.route("/")
def index():
    return render_template("index.html")  # Load HTML template

# ===============================
# Prediction route
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive JSON data from the frontend
        input_data = request.json
        input_df = pd.DataFrame([input_data])

        # Encode text feature (College)
        input_df['College'] = encoder.transform(input_df['College'])

        # Standardize input data
        input_scaled = scaler.transform(input_df)

        # Model prediction
        prediction = model.predict(input_scaled)

        # Generate feature importance using SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)[0]

        # Package feature importance results
        shap_results = [{"feature": features[i], "importance": shap_values[i]} for i in range(len(features))]

        # Return prediction and feature importance
        return jsonify({
            "draft_position": int(prediction[0]),
            "shap_values": shap_results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
