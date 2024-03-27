import re 
import pickle
import joblib
from ast import literal_eval
import random
import streamlit as st
import numpy as np
import pandas as pd 
from streamlit_chat import message
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize session state for chat messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Load models and data
max_seq_length = 30
df_input = pd.read_csv('df_input.csv')
loaded_model = load_model('intent_model.keras')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
model = joblib.load('xgb_clf.joblib')

# Load tokenizer and label encoder
with open('tokenizer_patterns.pkl', 'rb') as f:
    tokenizer_patterns = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Define helper functions
def preprocess_text(text, tfidf_vectorizer):
    def preprocessor(text):
        text = re.sub(r'@[^\s]+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\b(?:the|and|is|it|of|in|to|for|with|on|at|by|this|an|a)\b', ' ', text, flags=re.IGNORECASE)
        return text
    processed_text = preprocessor(text)
    #tfidf
    X_tfidf = tfidf_vectorizer.transform([processed_text]).toarray()
    return X_tfidf

def suicidal_post_detection(text):
    vectorized_sentence = preprocess_text(text, tfidf_vectorizer)
    predicted_label = model.predict(vectorized_sentence)
    return predicted_label

def get_random_response(tag):
    tag 
    tag_data = df_input[df_input['tag'] == tag]
    if not tag_data.empty:
        responses_list = literal_eval(tag_data['responses'].iloc[0])
        return random.choice(responses_list)
    else:
        return "Tag not found"

def generate_response(sentence):
    new_sentence_seq = tokenizer_patterns.texts_to_sequences([sentence])
    new_sentence_padded = pad_sequences(new_sentence_seq, maxlen=max_seq_length, padding='post')

    # Predict the intent
    intent_prediction = loaded_model.predict(new_sentence_padded)
    predicted_intent_index = np.argmax(intent_prediction)
    predicted_intent = label_encoder.classes_[predicted_intent_index]
    predicted_response = get_random_response(predicted_intent)
    return predicted_response

# Setting page configuration
st.set_page_config(page_title="App", page_icon=":robot_face:")

# Chatbot page layout
st.markdown("<h1 style='text-align: center;'>Chatbot</h1>", unsafe_allow_html=True)

# Input for post analysis
user_post = st.text_area("Please enter your message here:", key="user_post", height=100)
analyze_button = st.button("Analyze")

# Response container for chat messages
response_container = st.container()

# Initialize session state for tracking if chat should be shown
if 'show_chat' not in st.session_state:
    st.session_state['show_chat'] = False

# Analyze the input text and show the chat interface if needed
if analyze_button and user_post:
    predicted_label = suicidal_post_detection(user_post)
    if predicted_label == 0:
        st.success("The post doesn't show any signs of distress.")
        st.session_state['show_chat'] = False
    elif predicted_label == 1:
        st.error("Signs of Potential distress. Would you like to talk to someone?")
        chat_with_us = st.button("Chat with us")
        if chat_with_us:
            st.session_state['show_chat'] = True

# Show chat interface if show_chat is True
if st.session_state['show_chat']:
    with response_container:
        for idx, message_data in enumerate(st.session_state['messages']):
            message(message_data['content'], is_user=(message_data['role'] == "user"), key=f"msg_{idx}")

    with st.container():
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='chat_input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = generate_response(user_input)
            st.session_state['messages'].append({"role": "user", "content": user_input})
            st.session_state['messages'].append({"role": "assistant", "content": output})
