from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load saved files
model = joblib.load("disease_predictor_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")
feature_columns = joblib.load("model_features.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    # Ensure the endpoint is tested with a POST request and JSON payload
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    input_dict = dict.fromkeys(feature_columns, 0)

    for s in symptoms:
        for col in feature_columns:
            if s.strip().lower() in col.lower():
                input_dict[col] = 1

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)
    predicted = label_encoder.inverse_transform(prediction)[0]

    return jsonify({"prediction": predicted})

if __name__ == "__main__":
    app.run(port=5000)
