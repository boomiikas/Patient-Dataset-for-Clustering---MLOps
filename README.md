# 🩺 Patient Clustering App (DBSCAN + PCA + Gradio + Login Page)

This project is a **Machine Learning + Web App integration** where patient data is clustered using **DBSCAN (eps=3) with PCA (7 components)**.  
A **login page (HTML + CSS)** is provided to secure access before using the app.

---

## 🚀 Features
- **Clustering Model**:
  - Uses DBSCAN with eps=3, min_samples=5
  - Dimensionality reduction with PCA (7 components)
  - Patient input collected via Gradio interface
  - Returns the cluster and **heart disease likelihood**
- **Login Page**:
  - Responsive HTML + CSS login form
  - Gradient background with modern UI
  - Redirects to app after login
- **Tech Stack**:
  - Python (Pandas, Scikit-learn, Gradio)
  - HTML + CSS (for Login Page)
  - Joblib (optional for saving models)

---

## 📂 Project Structure
```
patient-clustering-app/
│── app.py                 # Main Gradio application
│── login.html             # Login page (HTML + CSS)
│── patient_data_cleaned.csv # Patient dataset
│── requirements.txt       # Python dependencies
│── README.md              # Documentation
```

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/patient-clustering-app.git
   cd patient-clustering-app
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On macOS/Linux
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Running the App

1. Start the Gradio app:
   ```bash
   python app.py
   ```

2. Open the **login.html** file in your browser.  
   After login, it will redirect to your **Gradio app (localhost:7860)**.

---

## 📊 Example Patient Input
- Age: `45`
- Gender: `Male`
- Chest Pain Type: `2`
- Blood Pressure: `130`
- Cholesterol: `240`
- Max Heart Rate: `150`
- Exercise Angina: `0`
- Plasma Glucose: `120`
- Skin Thickness: `30`
- Insulin: `80`
- BMI: `25.4`
- Diabetes Pedigree: `0.6`
- Hypertension: `1`
- Residence Type: `Urban`
- Smoking Status: `formerly smoked`

**Output Example**:
```
Patient belongs to Cluster 0 → Likely HEART DISEASE
```

---

## 📦 requirements.txt
```txt
pandas
scikit-learn
gradio
joblib
```

---

## 🛡️ Future Improvements
- Connect login with **Flask/Django backend** for authentication
- Save and load clustering model using **joblib**
- Store user inputs in database for analytics

---

## ✨ Author
Created by **[Your Name]** – Data Science + Web Integration Project 🚀

