import streamlit as st
import requests
import numpy as np
from collections import Counter
from math import log2
import anthropic

#api_key = st.secrets["CLAUDE_API_KEY"]

st.set_page_config(page_title="LLM Uncertainty", page_icon="䷅", layout="centered", initial_sidebar_state="auto", menu_items=None)
# Streamlit app layout
st.title("Chatbot with Claude (LLM) - Response Uncertainty Estimation")

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=st.secrets["CLAUDE_API_KEY"],
)

# Claude API interaction
def query_claude_api(prompt):
    """Query the Claude API and return the response"""
    message = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": prompt}
    ]
)
    return message.content[0].text

# Entropy calculation function
def calculate_entropy(responses):
    """Calculate entropy of the responses"""
    # Convert responses to strings to ensure they can be handled by Counter
    response_strings = [str(response) for response in responses]
    
    response_count = Counter(response_strings)
    total_responses = len(response_strings)
    
    entropy = -sum((count / total_responses) * log2(count / total_responses) for count in response_count.values())
    return entropy

# Input field for user's query
user_input = st.text_input("Enter your query:", key="user_input")

# If a query is entered, process the input
if user_input:
    # Query the LLM API 3 times
    responses = []
    with st.spinner("Querying Claude..."):
        for _ in range(3):
            response = query_claude_api(user_input)
            if response:  # Ensure we only store non-empty responses
                responses.append(response)

    # Check if we received valid responses
    if responses:
        # Select one of the responses to display (e.g., the first one)
        chosen_response = responses[0]

        # Display the responses
        with st.chat_message("assistant"):
        #st.write("### Responses from Claude:")
            st.write(chosen_response)
        #for idx, response in enumerate(responses):
            #st.write(f"Response {idx + 1}: {response}")

        # Calculate entropy to estimate uncertainty
        entropy = calculate_entropy(responses)
        
        # Display the entropy
        #st.write(f"### Estimated Entropy (Uncertainty): {entropy:.4f}")
        
        # Higher entropy indicates more uncertainty in the LLM's responses
        if entropy > 1.0:
            st.write("⚠️ High entropy: The model is uncertain about its response.")
        else:
            st.write("✅ Low entropy: The model is fairly confident in its responses.")
    else:
        st.error("Error: No valid responses received from Claude.")
    
# Add a brief explanation of entropy
st.write("""
#### Entropy in Language Models:
Entropy measures the uncertainty in the model's responses. Higher entropy means the model is producing a wider variety of answers, indicating uncertainty. Lower entropy means the responses are more consistent, suggesting greater confidence.
""")
