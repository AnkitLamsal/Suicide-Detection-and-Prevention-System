import random
import joblib
import pandas as pd
import streamlit as st 
from streamlit_chat import message
import torch
from sentence_transformers import SentenceTransformer


class BiLSTMClassifier1(torch.nn.Module):
    # for suicide
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTMClassifier1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = torch.nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        x = x.unsqueeze(1)
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            x.device
        )  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class BiLSTMClassifier2(torch.nn.Module):
    # for response
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTMClassifier2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = torch.nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a sequence length dimension
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out



def predict_tweet_suicidality(tweet, model, transformer_model):
    # Generate embedding
    embedding = transformer_model.encode([tweet], show_progress_bar=False)

    # Convert to tensor and add batch dimension
    embedding_tensor = torch.tensor(embedding).unsqueeze(0)

    # Ensure the input is 2D: [1, embedding_dim]
    embedding_tensor = embedding_tensor.squeeze(1)  # Remove extra dimension

    # Set the model to evaluation mode
    model.eval()

    # Make prediction
    with torch.no_grad():
        prediction = model(embedding_tensor).squeeze()

    # Apply sigmoid and threshold to get binary prediction
    predicted_label = torch.sigmoid(prediction).item() >= 0.5
    # return predicted_label
    return "Potential Suicide post" if predicted_label else "Not Suicide post"
    # return "This posts shows sign" if predicted_label else "Not Suicide post"

# print(predict_tweet_suicidality("Hi, I feel terrible today", model_detection, sentence_model_1))
# print(predict_tweet_suicidality("I am so", model_detection, sentence_model_1))

def predict_tag_and_response(text, model, transformer_model, label_encoder, df):
    # Existing code to predict the tag
    embedding = transformer_model.encode([text], show_progress_bar=False)
    embedding_tensor = torch.tensor(embedding).unsqueeze(0)
    embedding_tensor = embedding_tensor.squeeze(1)

    model.eval()
    with torch.no_grad():
        output = model(embedding_tensor)
        predicted_tag_index = torch.argmax(output, dim=1).item()
    predicted_tag = label_encoder.inverse_transform([predicted_tag_index])[0]

    # Retrieve responses for the predicted tag
    responses = df[df['tag'] == predicted_tag]['responses'].tolist()

    # Randomly select one response
    selected_response = random.choice(responses) if responses else "No response found."
    return selected_response


# Streamlit App 
st.set_page_config(page_title="Mental Well-being", page_icon=":brain:", layout="wide")

# Create a checkbox to toggle between chatbot and tweet analysis on the side bar
st.sidebar.title("Navigation")
st.sidebar.subheader("Choose a page")
page = st.sidebar.radio("Options", ("Home", "Chatbot"))

if page == "Home":
    # Parameters
    embedding_dim = 384  # Example, adjust based on your embeddings
    hidden_dim = 128
    num_layers = 2
    num_classes = 1  # 1 for first model, 80 for second
    num_epochs = 10

    # Initialize the model architecture
    model_detection = BiLSTMClassifier1(embedding_dim, hidden_dim, num_layers, num_classes)
    # Load the saved state dictionary
    model_detection.load_state_dict(torch.load("./models/model.pth"))
    model_detection.eval()  # Set the model to evaluation mode

    # Load the Sentence Transformer model
    sentence_model_1 = SentenceTransformer("./models/sentence_model")

    # Setting page title and header
    st.title("Text Analysis for Mental Well-being")
    user_input = st.text_area(label="Enter your text here:")
    button = st.button("Analyze")
    # Add a button
    if button and user_input:
        is_suicidal = predict_tweet_suicidality(user_input, model_detection, sentence_model_1)
        st.write(is_suicidal)
        if is_suicidal=="Not Suicide post":
            st.success("The post contains no signs of distress.")
        elif is_suicidal=="Potential Suicide post":
            st.error("The post might contain signs of distress.")
            st.write('\n\n')
            st.error("Talking to our free chatbot might help you feel better.")
        else:
            pass

elif page == "Chatbot":
    # Parameters
    embedding_dim = 384  # Example, adjust based on your embeddings
    hidden_dim = 128
    num_layers = 2
    num_classes = 80  # 1 for first model, 80 for second
    num_epochs = 10
    #Content Generation Model
    model_response = BiLSTMClassifier2(embedding_dim, hidden_dim, num_layers, num_classes)
    # Load the saved state dictionary
    model_response.load_state_dict(torch.load("./models/bilstm_model_response.pth"))
    model_response.eval()  # Set the model to evaluation mode

    # Load the Sentence Transformer model
    sentence_model_2 = SentenceTransformer("./models/sentence_transformer_model_response")

    # Load the label encoder
    with open('models/label_encoder.pkl', 'rb') as file:
        label_encoder = joblib.load(file)

    # Load the dataframe with tags and responses 
    df = pd.read_csv('./df_intents.csv')

    # Initialize session state for chat messages
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Only one functionality, so no need for a dropdown
    st.markdown("<h1 style='text-align: center;'>Chatbot</h1>", unsafe_allow_html=True)

    # Container for chat history
    response_container = st.container()

    # Container for text box
    with st.container():
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = predict_tag_and_response(user_input, model_response, sentence_model_2, label_encoder, df)
            st.session_state['messages'].append({"role": "user", "content": user_input})
            st.session_state['messages'].append({"role": "assistant", "content": output})

    if st.session_state['messages']:
        with response_container:
            for idx, message_data in enumerate(st.session_state['messages']):
                # Use the loop index as part of the key for uniqueness
                message(message_data['content'], is_user=(message_data['role'] == "user"), key=f"msg_{idx}")


