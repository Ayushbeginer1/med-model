import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import warnings
import re
from difflib import get_close_matches

warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("Cleaned_DiseaseAndSymptoms.csv")
df.fillna("unknown", inplace=True)
df.drop_duplicates(inplace=True)  # Remove duplicate rows

# Encode categorical symptoms (one-hot encoding)
X = pd.get_dummies(df.drop("Disease", axis=1))
Y = df["Disease"]

# Label encoding for target
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
print("\n📊 Classification Report:")
print(classification_report(label_encoder.inverse_transform(y_test),
                            label_encoder.inverse_transform(y_pred),
                            zero_division=1))

# Save model, encoder, and features
joblib.dump(model, "disease_predictor_model.joblib")
joblib.dump(label_encoder, "label_encoder.joblib")
joblib.dump(X.columns.tolist(), "model_features.joblib")

# ---------------- Interactive Prediction ----------------

# Load saved artifacts
model = joblib.load("disease_predictor_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")
model_features = joblib.load("model_features.joblib")

# Build a mapping of normalized feature names to original feature names.
# For example, "Symptom_4_vomiting" -> "vomiting"
def normalize_feature(feature):
    normalized = re.sub(r"^symptom_\d+_", "", feature, flags=re.IGNORECASE)
    return normalized

normalized_mapping = {normalize_feature(f): f for f in model_features}

# Accept user input
input_symptoms = input("\n🤒 Enter symptoms (comma-separated): ").strip().lower().split(',')
input_symptoms = [s.strip().replace(" ", "_") for s in input_symptoms]

# Fuzzy matching symptoms with user confirmation using normalized names
matched_symptoms = []
normalized_feature_list = list(normalized_mapping.keys())

for symptom in input_symptoms:
    match = get_close_matches(symptom, normalized_feature_list, n=1, cutoff=0.4)
    if match:
        corrected_norm_symptom = match[0]
        # Get the actual feature name from our mapping:
        corrected_feature = normalized_mapping[corrected_norm_symptom]
        if corrected_norm_symptom != symptom:
            user_input = input(f"🔄 Did you mean '{corrected_norm_symptom}' instead of '{symptom}'? (yes/no): ").strip().lower()
            if user_input == "yes":
                matched_symptoms.append(corrected_feature)
            else:
                print(f"⚠️ Skipping unrecognized symptom: '{symptom}'")
        else:
            matched_symptoms.append(corrected_feature)
    else:
        print(f"⚠️ Unrecognized symptom: '{symptom}'")

# (Optional) Suggest common symptoms if none matched
if not matched_symptoms:
    print("\n💡 Common symptoms you can try:")
    common_symptoms = model_features[:10]
    for feat in common_symptoms:
        print(f"- {normalize_feature(feat).replace('_', ' ')}")

# Show matched symptoms
if matched_symptoms:
    print("\n✅ Matched symptoms used for prediction:")
    for s in matched_symptoms:
        print("-", normalize_feature(s).replace('_', ' '))
else:
    print("❌ No symptoms matched. Please re-enter symptoms.")
    exit()  # Exit if no symptoms are matched

# Prepare input vector for prediction
input_dict = dict.fromkeys(model_features, 0)
for s in matched_symptoms:
    input_dict[s] = 1

input_df = pd.DataFrame([input_dict])

# Predict
prediction = model.predict(input_df)
predicted_disease = label_encoder.inverse_transform(prediction)[0]
print(f"\n🩺 You possibly have: {predicted_disease}")

# Top 5 predictions
probs = model.predict_proba(input_df)[0]
top_indices = probs.argsort()[-10:][::-1]  
top_diseases = label_encoder.inverse_transform(top_indices)
top_probs = probs[top_indices]

print("\n🔎 Top 10 possible diseases:")
for disease, prob in zip(top_diseases, top_probs):
    print(f"- {disease.title()}: \nprobability in relation to model threshold:- {prob:.2f}")

if top_probs[0] < 0.7:
    print("\n⚠️ The model is uncertain. Please provide more symptoms or consult a doctor.")
