import gradio as gr
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# ===========================
# Your ML Setup
# ===========================
df = pd.read_csv("patient_data_cleaned.csv")

X_raw = df.drop(columns=["heart_disease"])
y = df["heart_disease"]

X_encoded = pd.get_dummies(X_raw, columns=["gender", "residence_type", "smoking_status"], drop_first=True)
feature_columns = X_encoded.columns

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

pca = PCA(n_components=7, random_state=42)
X_pca = pca.fit_transform(X_scaled)

dbscan = DBSCAN(eps=3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_pca)

cluster_mapping = {}
for cluster_id in set(dbscan_labels):
    if cluster_id == -1:
        continue
    cluster_indices = (dbscan_labels == cluster_id)
    majority_label = y[cluster_indices].mode()[0]
    cluster_mapping[cluster_id] = majority_label


def cluster_patient(age, gender, chest_pain_type, blood_pressure, cholesterol,
                    max_heart_rate, exercise_angina, plasma_glucose, skin_thickness,
                    insulin, bmi, diabetes_pedigree, hypertension, residence_type, smoking_status):

    user_data = pd.DataFrame([[age, gender, chest_pain_type, blood_pressure, cholesterol,
                               max_heart_rate, exercise_angina, plasma_glucose, skin_thickness,
                               insulin, bmi, diabetes_pedigree, hypertension, residence_type, smoking_status]],
                             columns=["age", "gender", "chest_pain_type", "blood_pressure", "cholesterol",
                                      "max_heart_rate", "exercise_angina", "plasma_glucose", "skin_thickness",
                                      "insulin", "bmi", "diabetes_pedigree", "hypertension", "residence_type", "smoking_status"])

    user_encoded = pd.get_dummies(user_data, columns=["gender", "residence_type", "smoking_status"], drop_first=True)
    user_encoded = user_encoded.reindex(columns=feature_columns, fill_value=0)

    user_scaled = scaler.transform(user_encoded)
    user_pca = pca.transform(user_scaled)

    X_combined = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(1, 8)])
    X_combined.loc["new_patient"] = user_pca[0]

    db = DBSCAN(eps=2, min_samples=5)
    labels = db.fit_predict(X_combined)
    new_label = labels[-1]

    if new_label == -1:
        result = "This patient is considered NOISE (does not belong to any cluster)."
    else:
        predicted_hd = cluster_mapping.get(new_label, "Unknown")
        if predicted_hd == 1:
            result = f"Patient belongs to Cluster {new_label} → Likely No Heart Disease"
        elif predicted_hd == 0:
            result = f"Patient belongs to Cluster {new_label} → Likely HEART DISEASE"
        else:
            result = f"Patient belongs to Cluster {new_label}, but heart disease status is UNKNOWN."

    return result   # ✅ Only return text


# ===========================
# LOGIN SYSTEM
# ===========================
USERNAME = "admin"
PASSWORD = "1234"

def login(user, pwd):
    if user == USERNAME and pwd == PASSWORD:
        return gr.update(visible=False), gr.update(visible=True), ""
    else:
        return gr.update(), gr.update(), "❌ Invalid username or password"


# ===========================
# Gradio UI with Background
# ===========================
with gr.Blocks(
    css="""
    body {
        background: linear-gradient(135deg, #d9a7c7, #fffcdc);
    }
    .gr-button {
        background-color: #6a5acd !important;
        color: white !important;
        border-radius: 10px !important;
    }
    .gr-textbox, .gr-radio {
        background: #ffffffaa !important;
        border-radius: 10px !important;
    }
    """
) as demo:
    # Login Page
    with gr.Column(visible=True) as login_page:
        gr.Markdown("## 🔑 Login Page", elem_id="title")
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        error_msg = gr.Label(label="Status")

    # App Page
    with gr.Column(visible=False) as app_page:
        gr.Markdown("## 🩺 Patient Clustering App")
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
        output = gr.Textbox(label="Prediction")
        predict_btn = gr.Button("Predict")
        logout_btn = gr.Button("Logout")

        predict_btn.click(cluster_patient,inputs,output)

    # Button actions
    login_btn.click(login, [username, password], [login_page, app_page, error_msg])

    logout_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False), ""),
        inputs=[],
        outputs=[login_page, app_page, error_msg]
    )



if __name__ == "__main__":
    demo.launch()
