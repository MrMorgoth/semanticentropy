import streamlit as st
import requests
import numpy as np
from collections import Counter
from math import log2

# Claude API interaction
def query_claude_api(prompt, api_key):
    """Query the Claude API and return the response"""
    url = "https://api.anthropic.com/v1/complete"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "prompt": prompt,
        "model": "claude-v1",
        "max_tokens_to_sample": 100,
        "temperature": 0.7
    }
    
    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()
    return response_data.get("completion", "")

# Entropy calculation function
def calculate_entropy(responses):
    """Calculate entropy of the responses"""
    response_count = Counter(responses)
    total_responses = len(responses)
    
    entropy = -sum((count / total_responses) * log2(count / total_responses) for count in response_count.values())
    return entropy

# Streamlit app layout
st.title("Chatbot with Claude (LLM) - Response Uncertainty Estimation")

# Input field for user's query
user_input = st.text_input("Enter your query:", key="user_input")

# Claude API key (You can replace this with a more secure method of handling API keys)
api_key = st.text_input("Enter your Claude API key:", type="password", key="api_key")

# If a query is entered, process the input
if user_input and api_key:
    # Query the LLM API 3 times
    responses = []
    with st.spinner("Querying Claude..."):
        for _ in range(3):
            response = query_claude_api(user_input, api_key)
            responses.append(response)
    st.write(responses)
    # Display the responses
    st.write("### Responses from Claude:")
    for idx, response in enumerate(responses):
        st.write(f"Response {idx + 1}: {response}")

    # Calculate entropy to estimate uncertainty
    entropy = calculate_entropy(responses)
    
    # Display the entropy
    st.write(f"### Estimated Entropy (Uncertainty): {entropy:.2f}")
    
    # Higher entropy indicates more uncertainty in the LLM's responses
    if entropy > 1.0:
        st.write("⚠️ High entropy: The model is uncertain about its response.")
    else:
        st.write("✅ Low entropy: The model is fairly confident in its responses.")

# Add a brief explanation of entropy
st.write("""
#### Entropy in Language Models:
Entropy measures the uncertainty in the model's responses. Higher entropy means the model is producing a wider variety of answers, indicating uncertainty. Lower entropy means the responses are more consistent, suggesting greater confidence.
""")
