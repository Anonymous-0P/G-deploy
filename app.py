import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from googletrans import Translator
import os

@st.cache_resource
def load_model():
    model_name = "DumbKid-AI007/Espada-S1-7B"  # Update with your model path
    offload_folder = "./offload_folder"
    os.makedirs(offload_folder, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", offload_folder=offload_folder)
    return tokenizer, model

# Load model and tokenizer
tokenizer, model = load_model()

# Initialize translator
translator = Translator()

# Dictionary of language codes for Google Translate
languages = {
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Urdu": "ur"
}

# Streamlit app title
st.title("GENERA AI")

# User input text area
user_input = st.text_area("Enter your text here:")

# Language selection dropdown
selected_language = st.selectbox("Choose output language:", list(languages.keys()))

# Generate response button
button = st.button("Generate Response")

# Generate response when button is clicked
if user_input and button:
    # Tokenize input text
    inputs = tokenizer(user_input, return_tensors='pt').to(model.device)
    
    # Generate output text
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=200)
    
    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Translate generated text to selected language
    translated_text = translator.translate(generated_text, dest=languages[selected_language]).text
    
    # Display the translated response
    st.write("Generated Response:", translated_text)