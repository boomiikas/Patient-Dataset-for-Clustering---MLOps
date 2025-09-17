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


# ===========================
# Prediction Function
# ===========================
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

    db = DBSCAN(eps=3, min_samples=5)
    labels = db.fit_predict(X_combined)
    new_label = labels[-1]

    # 🎨 Color-coded HTML output (no emojis)
    if new_label == -1:
        return """<div style='background:#FFD966; padding:15px; border-radius:10px; 
                   font-size:18px; font-weight:bold; color:#333;'>
                   This patient is considered NOISE (does not belong to any cluster).
                  </div>"""
    else:
        predicted_hd = cluster_mapping.get(new_label, "Unknown")
        if predicted_hd == 1:
            return f"""<div style='background:#90EE90; padding:15px; border-radius:10px; 
                        font-size:18px; font-weight:bold; color:#064420;'>
                        Patient belongs to Cluster {new_label} → Likely No Heart Disease
                       </div>"""
        elif predicted_hd == 0:
            return f"""<div style='background:#FF7F7F; padding:15px; border-radius:10px; 
                        font-size:18px; font-weight:bold; color:#5A0000;'>
                        Patient belongs to Cluster {new_label} → Likely HEART DISEASE
                       </div>"""
        else:
            return f"""<div style='background:#FFD966; padding:15px; border-radius:10px; 
                        font-size:18px; font-weight:bold; color:#333;'>
                        Patient belongs to Cluster {new_label}, but heart disease status is UNKNOWN.
                       </div>"""


# ===========================
# LOGIN SYSTEM
# ===========================
USERNAME = "admin"
PASSWORD = "1234"

def login(user, pwd):
    if user == USERNAME and pwd == PASSWORD:
        return gr.update(visible=False), gr.update(visible=True), ""
    else:
        return gr.update(), gr.update(), "Invalid username or password"


# ===========================
# Custom CSS for Theme
# ===========================
css = """
body {
    background: linear-gradient(135deg, #ff9a9e, #fad0c4, #fbc2eb, #a1c4fd, #c2e9fb);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

#title {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #333;
}

.gr-button {
    border-radius: 12px !important;
    padding: 12px 20px !important;
    font-size: 16px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
    box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
    color: white !important;
}

#login_btn {
    background: linear-gradient(45deg, #28a745, #5cd65c) !important;
}
#login_btn:hover {
    transform: scale(1.05);
    opacity: 0.9;
}

#predict_btn {
    background: linear-gradient(45deg, #007bff, #33adff) !important;
}
#predict_btn:hover {
    transform: scale(1.05);
    opacity: 0.9;
}

#prediction_box {
    display: flex;
    justify-content: center;
    margin-top: 20px;
}

#prediction_box > div {
    width: 80%;
    max-width: 600px;
    text-align: center;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0px 6px 12px rgba(0,0,0,0.15);
}

#logout_btn {
    background: linear-gradient(45deg, #dc3545, #ff6b81) !important;
}
#logout_btn:hover {
    transform: scale(1.05);
    opacity: 0.9;
}

#component-0, #component-1, #component-2 {
    background: rgba(255,255,255,0.85);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 8px 15px rgba(0,0,0,0.1);
    margin: auto;
    width: 70%;
}
"""


# ===========================
# Gradio App
# ===========================
with gr.Blocks(css=css) as demo:
    # Login Page
    with gr.Column(visible=True) as login_page:
        gr.Markdown("## Login Page", elem_id="title")
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login", elem_id="login_btn")
        error_msg = gr.Label(label="Status")

    # App Page
    with gr.Column(visible=False) as app_page:
        gr.Markdown("## Patient Clustering App", elem_id="title")
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
            gr.Radio(["never smoked", "unknown", "smokes"], label="Smoking Status")
        ]
        output = gr.HTML(label="", elem_id="prediction_box")
        predict_btn = gr.Button("Predict", elem_id="predict_btn")
        logout_btn = gr.Button("Logout", elem_id="logout_btn")

        predict_btn.click(cluster_patient, inputs, output)

    # Button actions
    login_btn.click(login, [username, password], [login_page, app_page, error_msg])

    logout_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False), ""),
        inputs=[],
        outputs=[login_page, app_page, error_msg]
    )

if __name__ == "__main__":
    demo.launch()
