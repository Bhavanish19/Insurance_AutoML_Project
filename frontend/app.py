import streamlit as st
import requests
import pandas as pd
import io
import json

st.title('End-to-End AutoML Project: Insurance')

# Define the FastAPI endpoint
# endpoint = 'http://localhost:8000/predict'
endpoint = 'http://host.docker.internal:8000/predict' # Use this endpoint for Docker compatibility

# File uploader for the test CSV
test_csv = st.file_uploader('Upload Test Dataset (CSV format)', type=['csv'])

# Process the uploaded CSV file
if test_csv:
    test_df = pd.read_csv(test_csv)
    st.subheader('Preview of Uploaded Dataset')
    st.dataframe(test_df.head())

    # Convert the DataFrame to a BytesIO object for sending to the FastAPI endpoint
    test_bytes_obj = io.BytesIO()
    test_df.to_csv(test_bytes_obj, index=False)
    test_bytes_obj.seek(0)

    files = {"file": ('test_dataset.csv', test_bytes_obj, "multipart/form-data")}

    # Button to trigger the prediction
    if st.button('Start Prediction'):
        if test_df.empty:
            st.error("The uploaded dataset is empty. Please upload a valid dataset.")
        else:
            with st.spinner('Prediction in Progress. Please Wait...'):
                response = requests.post(endpoint, files=files, timeout=8000)
            st.success('Prediction Successful! Click the Download button below to retrieve the results in JSON format.')
            st.download_button(
                label='Download Prediction Results',
                data=json.dumps(response.json()),
                file_name='automl_prediction_results.json'
            )
