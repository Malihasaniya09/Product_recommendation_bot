🛍️ Product Recommendation System
📌 Project Overview

This project implements a Product Recommendation System using machine learning.
It predicts relevant products for users based on their behavior (e-commerce dataset) with the goal of achieving at least 80% precision@5.

The workflow is divided into three phases:

Phase 1: Data preprocessing, exploratory data analysis (EDA), and model training (Random Forest, MultiOutputClassifier).

Phase 2: Model deployment using FastAPI with /predict endpoint.

Phase 3: Streamlit chatbot application that integrates the model and an AI function-calling agent for interactive recommendations.

⚙️ Tech Stack

Python 3.9+

Pandas / NumPy – Data preprocessing & feature engineering

Scikit-learn – ML modeling & evaluation

Matplotlib / Seaborn – Data visualization

FastAPI – Model deployment

Streamlit – Frontend interface

Pickle – Model serialization

📂 Project Structure
product_recommendation/
│── data/                  # Raw & processed datasets
│── notebooks/             # EDA & model building (Jupyter/Colab)
│── models/                # Saved ML models (.pkl)
│── api/                   
│   └── fast_api.py            # FastAPI app with /predict endpoint
│── app/
│   └── streamlit_app.py   # Streamlit chatbot interface
│── requirements.txt       # Dependencies
│── README.md              # Project documentation

🚀 Getting Started
1️⃣ Clone the Repository
git clone https://github.com/yourusername/product_recommendation.git
cd product_recommendation

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train & Save Model (Phase 1)

Run the notebook:

jupyter notebook notebooks/model_training.ipynb


This will:

Perform EDA

Train the recommendation model

Save the model as models/recommender.pkl

4️⃣ Run FastAPI (Phase 2)
uvicorn api.main:app --reload


API will be available at:
👉 http://127.0.0.1:8000/docs

5️⃣ Run Streamlit App (Phase 3)
streamlit run app/streamlit_app.py


The chatbot UI will open in your browser.
