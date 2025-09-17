# 🩺 Patient Clustering Web App (Gradio + Python + MLflow)

This is a **Machine Learning + Web App** built using [Gradio](https://www.gradio.app/).  
It provides a simple **login system** and, after successful login, allows users to input patient details for clustering and prediction.  
The project also integrates **MLflow** for experiment tracking and MLOps workflow management.
**Demo link :** https://huggingface.co/spaces/boomiikas/Patient-dataset-clustering

---

## ✨ Features
- 🔑 **Login Authentication** (single username & password check)
- 🎨 **Simple, colorful, and decent background styling**
- 🤖 **Patient clustering** using DBSCAN with PCA (dimensionality reduction)
- 📊 Predicts **patient cluster** and **heart disease likelihood**
- 📈 **MLflow integration** for experiment tracking (models, metrics, parameters, artifacts)
- 🚀 Easy to run locally with Python

---

## 📂 Project Structure
```
project-folder/
│── app.py            # Main Gradio app
│── requirements.txt  # Python dependencies
│── README.md         # Project documentation
│── music.mp3         # (Optional) sound file if you want audio feedback
│── assets/           # (Optional) images, logos, etc.
│── mlruns/           # MLflow experiment tracking data (auto-generated)
```

---

## ⚙️ Installation

1. Clone this repository or download the code:
   ```bash
   git clone https://github.com/your-username/patient-clustering-app.git
   cd patient-clustering-app
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Running the App
Run the following command:
```bash
python app.py
```

This will start a local server (default: `http://127.0.0.1:7860/`).  
Open it in your browser to access the app.

---

## 🔑 Default Login
- **Username:** `admin`  
- **Password:** `password123`  

---

## 📊 MLflow Usage
This project uses **MLflow** to manage experiments and track model performance.

### Start MLflow UI
Run:
```bash
mlflow ui
```
Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) to view:
- 📌 Experiment Runs  
- 📌 Parameters (e.g., `eps`, `min_samples`, `n_components`)  
- 📌 Metrics (e.g., Silhouette score, accuracy)  
- 📌 Artifacts (saved models, visualizations)  

### Example Workflow
1. Preprocess data and try multiple clustering methods (DBSCAN, KMeans, etc.).  
2. Log runs with MLflow to compare **Silhouette Scores**.  
3. Deploy the best-performing model in `app.py`.  

---

## 📸 Screenshots

<img width="1228" height="639" alt="image" src="https://github.com/user-attachments/assets/725311cd-f90a-42f4-977f-de62d8d1bbb6" />
<img width="1025" height="879" alt="image" src="https://github.com/user-attachments/assets/560c7291-c3fc-47ff-bbb8-ab7c253d22cf" />

---

## 🛠️ Tech Stack
- Python 🐍
- Gradio 🎨
- Scikit-learn 🤖 (DBSCAN + PCA)
- Pandas / NumPy 📊
- MLflow 📈 (for experiment tracking and MLOps)

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
