import os
import base64
import pickle
import pandas as pd
import streamlit as st
from utils.unpickle_file import unpickle_file

RFR = unpickle_file(os.path.join('artifacts', 'model.pkl'))

def train_model(data_csv, model):
    data = pd.read_csv(data_csv)
    features, target = data.drop('meantemp', axis=1), data['meantemp']
    model.fit(features, target)
    
    return model

def get_model_download_link(model, filename="trained_model.pkl"):
    """Generates a download link for the trained model."""
    model_binary = base64.b64encode(pickle.dumps(model)).decode()
    href = f'<a href="data:application/octet-stream;base64,{model_binary}" download="{filename}">Download Trained Model</a>'
    return href

def training_pipeline():
    st.header('Model Training App')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        processing_spinner = st.empty()
        processing_spinner.text("Processing uploaded data...")

        trained_model = train_model(data_csv=uploaded_file, model=RFR)

        # Display success message
        processing_spinner.text("Model trained successfully!")
            
        # Add a button to download the trained model
        trained_model_link = get_model_download_link(trained_model)
        st.markdown(trained_model_link, unsafe_allow_html=True)

if __name__ == '__main__':
    training_pipeline()