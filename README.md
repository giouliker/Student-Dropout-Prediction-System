# Student-Dropout-Prediction-System

A Machine Learning system designed to predict student dropout risks and academic success. This tool helps educational institutions identify at-risk students early and intervene proactively.

## 📊 Overview
* **Problem:** High dropout rates in higher education.
* **Solution:** A Predictive Model using Random Forest to classify students into three categories: **Dropout**, **Enrolled**, and **Graduate**.
* **Key Features:**
  * Analyzes socio-economic factors (Tuition fees, Debt).
  * Evaluates academic performance (Grades, Approved Units).
  * Provides real-time predictions via a Web App.

## 🛠️ Technologies Used
* **Python** (Pandas, NumPy)
* **Scikit-Learn** (Random Forest Classifier)
* **Streamlit** (Web Application Interface)
* **Google Colab** (Development Environment)

## 📈 Model Performance
* **Accuracy:** ~77%
* **Key Insight:** The model successfully identifies the majority of at-risk students (Recall for Dropout class is high), which is critical for early warning systems.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app:
   ```bash
   streamlit run app.py
