# app.py
from flask import Flask, render_template, request, redirect, url_for
from Pipeline import data_processing_pipeline, model_training_pipeline, model_prediction_pipeline
import os
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Load the pickled model
model_path = os.path.join('artifacts', 'model.pkl')
with open(model_path, 'rb') as f:
    RFR = pickle.load(f)

# Define routes and views
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data_preprocessing', methods=['GET', 'POST'])
def data_preprocessing():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        # Call the processing_pipeline function from data_processing_pipeline module
        processed_data, csv_file_processed = data_processing_pipeline.processing_pipeline(file)
        
        # Render the template with processed data
        return render_template('data_preprocessing.html', processed_data=processed_data, csv_file_processed=csv_file_processed)
    
    return render_template('data_preprocessing.html')

@app.route('/model_training', methods=['GET', 'POST'])
def model_training():
    if request.method == 'POST':
        # Handle the form submission for model training
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return render_template('model_training.html', error="No file selected")

        # Train the model
        trained_model = model_training_pipeline.train_model(uploaded_file, model=RFR)

        # Generate a download link for the trained model
        trained_model_link = model_training_pipeline.get_model_download_link(trained_model)

        # Render the template with the trained model link
        return render_template('model_training.html', trained_model_link=trained_model_link)

    return render_template('model_training.html')

@app.route('/model_prediction', methods=['GET', 'POST'])
def model_prediction():
    if request.method == 'POST':
        # Get the uploaded CSV file
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return render_template('model_prediction.html', error="No file selected")

        # Process the uploaded file and make predictions
        processed_data, csv_file_processed = data_processing_pipeline.processing_pipeline(uploaded_file)
        predictions = RFR.predict(processed_data)
        processed_data['prediction'] = predictions

        # Render the template with combined data
        return render_template('model_prediction.html', combined_data=processed_data, csv_file_combined=csv_file_processed)

    return render_template('model_prediction.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
