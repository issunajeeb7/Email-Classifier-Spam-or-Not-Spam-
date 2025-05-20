import streamlit as st
import joblib
from nb import preprocess_text

# Load the saved model
try:
    model = joblib.load('spam_model.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please ensure the model has been trained.")
    st.stop()

# Set page config for a minimal look in white colour
st.set_page_config(
    page_title="Email Spam Detector",
    layout="centered"
)

# Custom CSS for modern minimal design
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 15px;
        font-size: 16px;
    }
    .stButton > button {
        background-color: #000000;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        padding: 10px 25px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #333333;
        transform: translateY(-2px);
    }
    .result-box {
        padding: 20px;
        border-radius: 4px;
        margin-top: 20px;
        text-align: center;
        font-size: 18px;
    }
    .spam {
        background-color: #ffebee;
        color: #c62828;
    }
    .not-spam {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Title with custom styling
st.markdown("<h1 style='text-align: center; color: #ffffff; margin-bottom: 30px;'>Email Spam Detector</h1>", unsafe_allow_html=True)

# Text input for message
message = st.text_area(
    "Enter your message:",
    height=150,
    placeholder="Type or paste your email message here...",
    key="message_input"
)

# Only show prediction when there's input
if message.strip():
    # Preprocess and predict
    processed_input = preprocess_text(message)
    prediction = model.predict([processed_input])[0]
    probability = model.predict_proba([processed_input])[0]
    
    # Get confidence score
    confidence = probability[1] if prediction == 1 else probability[0]
    
    # Display result with custom styling
    result = "Spam" if prediction == 1 else "Not Spam"
    result_color = "spam" if prediction == 1 else "not-spam"
    
    st.markdown(
        f"""<div class='result-box {result_color}'>
            <strong>{result}</strong><br>
            <span style='font-size: 14px;'>Confidence: {confidence:.2%}</span>
        </div>""",
        unsafe_allow_html=True
    )
    
    # Additional explanation
    st.markdown(
        "<div style='margin-top: 30px; padding: 20px; background-color: #f5f5f5; border-radius: 4px;'>"
        "<h4 style='margin: 0; color: #333333;'>How it works:</h4>"
        "<p style='margin-top: 10px; color: #666666;'>"
        "This spam detector uses machine learning to analyze the content and patterns "
        "in your message. It looks for common spam indicators like suspicious words, "
        "unusual formatting, and other patterns typically found in spam messages."
        "</p>"
        "</div>",
        unsafe_allow_html=True
    )
else:
    # Show placeholder message when input is empty
    st.markdown(
        "<div style='text-align: center; color: #666666; margin-top: 30px;'>"
        "Enter a message above to check if it's spam"
        "</div>",
        unsafe_allow_html=True
    )