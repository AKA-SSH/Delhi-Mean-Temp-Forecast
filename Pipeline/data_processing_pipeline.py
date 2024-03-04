# data_processing_pipeline.py
import base64
import pandas as pd

def process_data(data_csv):
    data = pd.read_csv(data_csv)
    data.set_index('date', inplace=True)
    data['meantemp (t-1)'] = data.meantemp.shift(1)
    data.dropna(inplace=True)
    return data

def processing_pipeline(uploaded_file):
    processed_data = process_data(uploaded_file)

    # Convert processed data to CSV string
    csv_file_processed = processed_data.to_csv(index=False)
    
    return processed_data, csv_file_processed
