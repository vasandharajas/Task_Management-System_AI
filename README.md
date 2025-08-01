﻿🧠 AI-Powered Task Management System
An intelligent task management system that uses Natural Language Processing (NLP) and Machine Learning (ML) to automatically classify, prioritize, and analyze tasks based on their descriptions.

🚀 Overview
This project helps automate task handling by:

 1.Predicting task categories (e.g., Bug, Feature, Documentation)

 2.Predicting priority levels (High, Medium, Low)

 3.Optionally analyzing task sentiment (Positive/Negative)

 4.Visualizing model performance and confusion matrices via a user-friendly Streamlit dashboard

🔍 Key Features
 1.Text Classification using NLP (TF-IDF, preprocessing)

 2.Priority Prediction using Random Forest and XGBoost

 3.Real-Time Dashboard built with Streamlit

 4.Evaluation Reports with metrics: Accuracy, Precision, Recall, F1-Score

 5.Confusion Matrix Visualizations for model performance

 6.Optionally includes Workload Balancing Logic (assigning tasks based on current user load)

🧪 Models Used
 1.TF-IDF Vectorizer for converting task descriptions into feature vectors

 2.Naive Bayes / SVM for task category classification

 3.Random Forest / XGBoost for task priority prediction

 4.Label Encoding for output targets

 5.Joblib for model serialization

📊 Dashboard Preview
<!-- Replace with an actual image if available -->

 1.Enter any task description

 2.View predicted category and priority

 3.Show detailed evaluation metrics

 4.Compare model performance visually

📦 Dependencies
 1.scikit-learn

 2.xgboost

 3.streamlit

 4.joblib

 5.matplotlib

📌 Final Notes
. The system improves task management automation using AI.
. Can be extended to include team member recommendations, due date suggestions, or task reassignment features.
