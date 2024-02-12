import streamlit as st
from Pipeline import data_processing_pipeline, model_training_pipeline, model_prediction_pipeline

def main():
    st.title("Machine Learning Pipeline")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Data Preprocessing", "Model Training", "Model Prediction"))

    if page == "Data Preprocessing":
        data_processing_pipeline.processing_pipeline()
    elif page == "Model Training":
        model_training_pipeline.training_pipeline()
    elif page == "Model Prediction":
        model_prediction_pipeline.prediction_pipeline()

if __name__ == "__main__":
    main()
