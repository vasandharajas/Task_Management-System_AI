### Import Libraries
import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils.multiclass import unique_labels
from textblob import TextBlob

# Load saved models and transformers
@st.cache_resource
def load_models():
    try:
        base_path = "your path"
        rf_model = joblib.load(os.path.join(base_path, "your rfmodel path"))
        xgb_model = joblib.load(os.path.join(base_path, "your xgbmodel path"))
        le = joblib.load(os.path.join(base_path, "your lable path"))
        tfidf = joblib.load(os.path.join(base_path, "your tfidf path"))
        return rf_model, xgb_model, le, tfidf
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e.filename}")
        return None, None, None, None

# Load models
rf_model, xgb_model, le, tfidf = load_models()

# App UI
st.title("🧠 AI Task Priority Prediction Dashboard")
st.markdown("Use AI models to classify the **priority level** of tasks based on their descriptions.")

# User Input
user_input = st.text_area("✍️ Enter a task description to predict priority and analyze sentiment:")

# Prediction & Visualization
if user_input and all([rf_model, xgb_model, le, tfidf]):
    vect = tfidf.transform([user_input])
    pred_rf = le.inverse_transform(rf_model.predict(vect))[0]
    pred_xgb = le.inverse_transform(xgb_model.predict(vect))[0]

    # Sentiment Analysis
    sentiment = TextBlob(user_input).sentiment
    sentiment_label = "Positive" if sentiment.polarity >= 0 else "Negative"

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🎯 **Random Forest Prediction:** `{pred_rf}`")
        st.info(f"📊 **XGBoost Prediction:** `{pred_xgb}`")

    with col2:
        st.write("💬 **Sentiment Analysis**")
        st.metric("Polarity", f"{sentiment.polarity:.2f}")
        st.metric("Sentiment", sentiment_label)

    # Priority Level Graph
    st.subheader("🔍 Task Insights")
    priority_counts = {
        "Random Forest": pred_rf,
        "XGBoost": pred_xgb
    }
    fig, ax = plt.subplots()
    sns.barplot(x=list(priority_counts.keys()), y=list(le.transform(list(priority_counts.values()))), palette="pastel", ax=ax)
    ax.set_ylabel("Encoded Priority Level")
    st.pyplot(fig)

# Evaluation Metrics & Confusion Matrix
if st.button("📉 Show Evaluation Metrics & Confusion Matrices"):
    x_test_path = "your x-test path"
    y_test_path = "your y-test path"

    if os.path.exists(x_test_path) and os.path.exists(y_test_path):
        X_test = joblib.load(x_test_path)
        y_test = joblib.load(y_test_path)
        y_test_encoded = le.transform(y_test)

        y_pred_rf = rf_model.predict(X_test)
        y_pred_xgb = xgb_model.predict(X_test)

        present_labels = unique_labels(y_test_encoded, y_pred_rf)

        # Random Forest Metrics
        st.subheader("📘 Random Forest Metrics")
        st.text(classification_report(y_test_encoded, y_pred_rf, target_names=le.inverse_transform(present_labels)))
        st.metric("Accuracy", f"{accuracy_score(y_test_encoded, y_pred_rf):.2f}")
        st.metric("Precision", f"{precision_score(y_test_encoded, y_pred_rf, average='weighted'):.2f}")
        st.metric("Recall", f"{recall_score(y_test_encoded, y_pred_rf, average='weighted'):.2f}")
        st.metric("F1 Score", f"{f1_score(y_test_encoded, y_pred_rf, average='weighted'):.2f}")

        fig_rf, ax_rf = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y_test_encoded, y_pred_rf,
            labels=present_labels,
            display_labels=le.inverse_transform(present_labels),
            cmap=plt.cm.Blues,
            ax=ax_rf
        )
        st.pyplot(fig_rf)

        # XGBoost Metrics
        st.subheader("📙 XGBoost Metrics")
        st.text(classification_report(y_test_encoded, y_pred_xgb, target_names=le.inverse_transform(present_labels)))
        st.metric("Accuracy", f"{accuracy_score(y_test_encoded, y_pred_xgb):.2f}")
        st.metric("Precision", f"{precision_score(y_test_encoded, y_pred_xgb, average='weighted'):.2f}")
        st.metric("Recall", f"{recall_score(y_test_encoded, y_pred_xgb, average='weighted'):.2f}")
        st.metric("F1 Score", f"{f1_score(y_test_encoded, y_pred_xgb, average='weighted'):.2f}")

        fig_xgb, ax_xgb = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y_test_encoded, y_pred_xgb,
            labels=present_labels,
            display_labels=le.inverse_transform(present_labels),
            cmap=plt.cm.Blues,
            ax=ax_xgb
        )
        st.pyplot(fig_xgb)
    else:
        st.warning("📂 Test data not found. Please ensure `X_test_tfidf.pkl` and `y_test.pkl` are in the `models` folder.")

# Summary & Final Comments
st.markdown("---")
st.subheader("📝 Final Comments & Summary")
st.markdown("""
- ✅ **Finalize models** for task classification and priority prediction using Random Forest and XGBoost.
- 🧩 Dashboard enables **real-time predictions** and shows both model outcomes.
- 📈 Includes **sentiment analysis** to assess task tone (positive/negative).
- 📊 **Evaluation section** visualizes confusion matrices and classification metrics.
- 🔍 Use this interface to **demo your project**, share insights, and validate model performance.
""")
