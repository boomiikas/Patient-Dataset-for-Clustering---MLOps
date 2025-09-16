import gradio as gr
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# Load dataset
df = pd.read_csv("patient_data_cleaned.csv")

# Separate features and target
X_raw = df.drop(columns=["heart_disease"])
y = df["heart_disease"]

# Encode categorical variables
X_encoded = pd.get_dummies(X_raw, columns=["gender", "residence_type", "smoking_status"], drop_first=True)
feature_columns = X_encoded.columns

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# PCA
pca = PCA(n_components=7, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_pca)

# Create cluster-to-heart_disease mapping (majority vote)
cluster_mapping = {}
for cluster_id in set(dbscan_labels):
    if cluster_id == -1:  # Skip noise
        continue
    cluster_indices = (dbscan_labels == cluster_id)
    majority_label = y[cluster_indices].mode()[0]  # most common heart_disease value in cluster
    cluster_mapping[cluster_id] = majority_label


def cluster_patient(age, gender, chest_pain_type, blood_pressure, cholesterol,
                    max_heart_rate, exercise_angina, plasma_glucose, skin_thickness,
                    insulin, bmi, diabetes_pedigree, hypertension, residence_type, smoking_status):

    # Convert user input into DataFrame
    user_data = pd.DataFrame([[
        age, gender, chest_pain_type, blood_pressure, cholesterol,
        max_heart_rate, exercise_angina, plasma_glucose, skin_thickness,
        insulin, bmi, diabetes_pedigree, hypertension, residence_type, smoking_status
    ]], columns=[
        "age", "gender", "chest_pain_type", "blood_pressure", "cholesterol",
        "max_heart_rate", "exercise_angina", "plasma_glucose", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "hypertension", "residence_type", "smoking_status"
    ])

    # Encode
    user_encoded = pd.get_dummies(user_data, columns=["gender", "residence_type", "smoking_status"], drop_first=True)
    user_encoded = user_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale + PCA
    user_scaled = scaler.transform(user_encoded)
    user_pca = pca.transform(user_scaled)

    # Combine with existing PCA dataset
    X_combined = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(1, 8)])
    X_combined.loc["new_patient"] = user_pca[0]

    # Re-run DBSCAN on combined data
    db = DBSCAN(eps=3, min_samples=5)
    labels = db.fit_predict(X_combined)
    new_label = labels[-1]

    # Return cluster result
    if new_label == -1:
        return "This patient is considered NOISE (does not belong to any cluster)."
    else:
        predicted_hd = cluster_mapping.get(new_label, "Unknown")
        if predicted_hd == 1:
            return f"Patient belongs to Cluster {new_label} → Likely No Heart Disease"
        elif predicted_hd == 0:
            return f"Patient belongs to Cluster {new_label} → Likely HEART DISEASE"
        else:
            return f"Patient belongs to Cluster {new_label}, but heart disease status is UNKNOWN."


# Gradio inputs
inputs = [
    gr.Number(label="Age"),
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Number(label="Chest Pain Type"),
    gr.Number(label="Blood Pressure"),
    gr.Number(label="Cholesterol"),
    gr.Number(label="Max Heart Rate"),
    gr.Number(label="Exercise Angina (0=No, 1=Yes)"),
    gr.Number(label="Plasma Glucose"),
    gr.Number(label="Skin Thickness"),
    gr.Number(label="Insulin"),
    gr.Number(label="BMI"),
    gr.Number(label="Diabetes Pedigree"),
    gr.Number(label="Hypertension (0=No, 1=Yes)"),
    gr.Radio(["Urban", "Rural"], label="Residence Type"),
    gr.Radio(["never smoked", "formerly smoked", "smokes"], label="Smoking Status")
]

# Launch Gradio app
demo = gr.Interface(
    fn=cluster_patient,
    inputs=inputs,
    outputs="text",
    title="Patient Clustering App",
    description="Enter patient details to see which cluster (DBSCAN eps=3, PCA=7) they belong to and if it's likely heart disease."
)

if __name__ == "__main__":
    demo.launch()
