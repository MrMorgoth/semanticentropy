import streamlit as st
import shap
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

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

    return pipeline, vectorizer, model

# Get SHAP explainer for the model
def get_shap_explainer(model, X_train_transformed):
    explainer = shap.LinearExplainer(model, X_train_transformed, feature_perturbation="interventional")
    return explainer

# Predict sentiment and explain with SHAP values
def predict_and_explain(text, vectorizer, model, explainer):
    # Transform the input text using the vectorizer
    transformed_text = vectorizer.transform([text])
    
    # Predict sentiment
    prediction = model.predict(transformed_text)[0]
    prediction_proba = model.predict_proba(transformed_text)[0]

    # Explain prediction using SHAP values
    shap_values = explainer.shap_values(transformed_text)
    return prediction, prediction_proba, shap_values

# Function to display SHAP force plot in Streamlit
def st_shap(plot):
    """Render a SHAP plot in Streamlit"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    shap.force_plot(plot, ax=ax)
    st.pyplot(fig)

# Initialize the Streamlit app
st.title("Chatbot with SHAP Value Explanations")

# Train model and initialize SHAP explainer
pipeline, vectorizer, model = train_sentiment_model()

# Use transformed data from the training set to initialize SHAP explainer
X_train_transformed = pipeline.named_steps['countvectorizer'].transform(['I love this product', 'This is terrible', 'Absolutely fantastic'])
explainer = get_shap_explainer(model, X_train_transformed)

# Chatbot interaction
st.write("Ask a question or type a statement:")

user_input = st.text_input("Your input", key="input_text")

if user_input:
    # Get prediction and SHAP values
    prediction, prediction_proba, shap_values = predict_and_explain(user_input, vectorizer, model, explainer)

    # Display the prediction
    sentiment = "Positive" if prediction == 1 else "Negative"
    st.write(f"Prediction: **{sentiment}** (Confidence: {prediction_proba[prediction]:.2f})")

    # Display SHAP values as a force plot
    st.write("Explanation of model's decision:")
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], feature_names=vectorizer.get_feature_names_out()))
