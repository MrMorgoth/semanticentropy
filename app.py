import streamlit as st
import shap
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

# Load SHAP explainer
def get_shap_explainer(pipeline):
    explainer = shap.Explainer(pipeline)
    return explainer

# Train a simple sentiment analysis model
def train_sentiment_model():
    # Example training data (positive and negative sentences)
    data = pd.DataFrame({
        'text': ['I love this product', 'This is terrible', 'Absolutely fantastic', 'Not good at all', 'I am very happy', 'This is the worst'],
        'label': [1, 0, 1, 0, 1, 0]
    })
    
    vectorizer = CountVectorizer()
    model = LogisticRegression()

    # Create a pipeline with vectorizer and model
    pipeline = make_pipeline(vectorizer, model)
    pipeline.fit(data['text'], data['label'])

    return pipeline

# Predict sentiment and explain with SHAP values
def predict_and_explain(text, pipeline, explainer):
    # Predict sentiment
    prediction = pipeline.predict([text])[0]
    prediction_proba = pipeline.predict_proba([text])[0]

    # Explain prediction using SHAP values
    shap_values = explainer([text])
    return prediction, prediction_proba, shap_values

# Initialize the Streamlit app
st.title("Chatbot with SHAP Value Explanations")

# Train model and initialize SHAP explainer
pipeline = train_sentiment_model()
explainer = get_shap_explainer(pipeline)

# Chatbot interaction
st.write("Ask a question or type a statement:")

user_input = st.text_input("Your input", key="input_text")

if user_input:
    # Get prediction and SHAP values
    prediction, prediction_proba, shap_values = predict_and_explain(user_input, pipeline, explainer)

    # Display the prediction
    sentiment = "Positive" if prediction == 1 else "Negative"
    st.write(f"Prediction: **{sentiment}** (Confidence: {prediction_proba[prediction]:.2f})")

    # Display SHAP values as a force plot
    st.write("Explanation of model's decision:")
    shap_values_text = shap_values[0]
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values_text.values, user_input, matplotlib=True))

# Function to display SHAP force plot in Streamlit
def st_shap(plot, height=None):
    """Render a SHAP plot in Streamlit"""
    import matplotlib.pyplot as plt
    import io

    fig, ax = plt.subplots(figsize=(12, 5))
    shap.force_plot(plot, ax=ax)
    st.pyplot(fig)
