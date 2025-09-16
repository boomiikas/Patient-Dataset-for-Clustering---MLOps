# 🩺 Patient Clustering Web App (Gradio + Python)

This is a **Machine Learning + Web App** built using [Gradio](https://www.gradio.app/).  
It provides a simple **login system** and, after successful login, allows users to input patient details for clustering and prediction.

---

## ✨ Features
- 🔑 **Login Authentication** (single username & password check)
- 🎨 **Simple, colorful, and decent background styling**
- 🤖 **Patient clustering** using DBSCAN with PCA (dimensionality reduction)
- 📊 Predicts **patient cluster** and **heart disease likelihood**
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

*(You can modify this in `app.py`.)*

---

## 📸 Screenshots
*(Add your own screenshots here — login page, prediction page, results, etc.)*

---

## 📌 Notes
- Make sure to keep `app.py` and any assets (e.g., images, optional music) in the same folder.  
- If you want to enable **button sound effects**, you can re-add `music.mp3` and use the `js` event in Gradio.  

---

## 🛠️ Tech Stack
- Python 🐍
- Gradio 🎨
- Scikit-learn 🤖 (for DBSCAN + PCA)
- Pandas / NumPy 📊

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
