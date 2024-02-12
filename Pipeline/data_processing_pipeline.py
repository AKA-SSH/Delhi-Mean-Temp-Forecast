import os
import base64
import pandas as pd
import streamlit as st

def process_data(data_csv):
    data = pd.read_csv(data_csv)
    data.set_index('date', inplace=True)
    data['meantemp (t-1)'] = data.meantemp.shift(1)
    data.dropna(inplace=True)
    return data

def processing_pipeline():
    st.header('Data Preprocessing')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        processing_spinner = st.empty()
        processing_spinner.text("Processing uploaded data...")
        
        # Process the data
        processed_data = process_data(uploaded_file)
        
        processing_spinner.text("Data processed successfully!")

        # Display processed data
        st.subheader('Processed Data')
        st.write('Combined Features and Target:')
        st.write(processed_data.head())

        # Download link for processed data
        st.subheader('Download Processed Data')
            
        csv_file_processed = processed_data.to_csv(index=False)
        b64_processed = base64.b64encode(csv_file_processed.encode()).decode()
        href_processed = f'<a href="data:file/csv;base64,{b64_processed}" download="processed_data.csv">Download Processed Data CSV File</a>'
        st.markdown(href_processed, unsafe_allow_html=True)

if __name__ == '__main__':
    processing_pipeline()
