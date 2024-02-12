import os
import base64
import streamlit as st
import pickle
import pandas as pd
from utils.unpickle_file import unpickle_file

RFR = unpickle_file(os.path.join('artifacts', 'model.pkl'))

def process_data(raw_features_csv):
    raw_features = pd.read_csv(raw_features_csv)
    raw_features['meantemp (t-1)'] = raw_features.meantemp.shift(1)
    selected_columns = ['humidity', 'wind_speed', 'meanpressure', 'meantemp (t-1)']
    features = raw_features[selected_columns]
    features.dropna(inplace=True)

    return features

def get_download_link(df, filename="data_prediction.csv"):
    """Generates a download link for the DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Data with Predictions CSV File</a>'
    return href

def prediction_pipeline():
    st.title('Prediction App')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        features = process_data(uploaded_file)
        predictions = RFR.predict(features)
        features['prediction'] = predictions

        # Display DataFrame with predictions
        st.subheader("DataFrame with predictions")
        st.write(features.head())

        download_link = get_download_link(features)
        st.markdown(download_link, unsafe_allow_html=True)
    else:
        st.warning("Please upload a CSV file to make predictions.")

if __name__ == '__main__':
    prediction_pipeline()
