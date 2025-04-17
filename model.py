import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from difflib import get_close_matches
import warnings
import joblib
warnings.filterwarnings("ignore")

# Load cleaned data
df = pd.read_csv("Cleaned_DiseaseAndSymptoms.csv")

# Fill any potential missing values
df.fillna("unknown", inplace=True)

# Prepare features and target
X = pd.get_dummies(df.drop("Disease", axis=1))
Y = df["Disease"]

# Encode target labels
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Evaluate model
y_pred = model.predict(x_test)
print("\n📊 Classification Report:")
print(classification_report(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred), zero_division=1))

# Accept user input
input_symptoms = input("\n🤒 Enter symptoms (comma-separated): ").strip().lower().split(',')
input_symptoms = [s.strip().replace(" ", "_") for s in input_symptoms]

# Match user input symptoms with actual symptom features using fuzzy matching
symptom_columns = X.columns
input_dict = dict.fromkeys(symptom_columns, 0)

for symptom in input_symptoms:
    # Match symptom to closest known feature
    matches = get_close_matches(symptom, symptom_columns, n=1, cutoff=0.6)
    if matches:
        input_dict[matches[0]] = 1
    else:
        print(f"⚠️ Unrecognized symptom: '{symptom}'")

# Create input DataFrame
input_df = pd.DataFrame([input_dict])

# Predict disease
prediction = model.predict(input_df)
predicted_disease = label_encoder.inverse_transform(prediction)[0]

# Show result
print(f"\n🩺 You possibly have: {predicted_disease}")

# Top 3 predictions
probs = model.predict_proba(input_df)[0]
top_indices = probs.argsort()[-3:][::-1]
top_diseases = label_encoder.inverse_transform(top_indices)
top_probs = probs[top_indices]

print("\n🔎 Top 3 possible diseases:")
for disease, prob in zip(top_diseases, top_probs):
    print(f"- {disease.title()}: {prob:.2f}")

# Warn if uncertain
if top_probs[0] < 0.7:
    print("\n⚠️ The model is uncertain. Please provide more specific symptoms or consult a doctor.")

import joblib

joblib.dump(model, "disease_predictor_model.joblib")

# Save label encoder
joblib.dump(label_encoder, "label_encoder.joblib")

# Save feature columns (used for user input formatting)
joblib.dump(X.columns.tolist(), "model_features.joblib")
