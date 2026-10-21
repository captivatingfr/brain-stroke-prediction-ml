# Brain Stroke Prediction Using Machine Learning

## Project Overview
This project demonstrates a machine-learning workflow for predicting the stroke class from demographic and health-related features in a dataset.

## Objective
To explore and preprocess the dataset, train and evaluate classification models, compare their performance, and demonstrate a trained model through an interactive web application.

## Algorithms
- Logistic Regression
- Decision Tree
- Random Forest

## Workflow
1. Dataset loading and exploration
2. Exploratory data analysis and visualization
3. Categorical feature encoding
4. Training and testing data split
5. Model training and hyperparameter tuning
6. Evaluation using precision, recall, F1-score and ROC-AUC
7. Model export
8. Interactive application using Streamlit

## Project Files
- `brain-stroke-prediction-using-lr-dt-rf.ipynb` — ML notebook
- `stroke_model.joblib` — saved model and feature configuration
- `app.py` — Streamlit application
- `requirements.txt` — dependencies
- `README.md` — project documentation

## Run Locally
Install dependencies:

`pip install -r requirements.txt`

Start the app:

`streamlit run app.py`

## Limitations
This is an academic demonstration trained on a specific dataset. Its outputs are not medical diagnoses, and the model has not been validated for clinical use.
