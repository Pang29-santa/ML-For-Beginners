# app.py
import streamlit as st
import pickle
import numpy as np

# Load the model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set up the Streamlit page
st.title('Linear Regression Prediction App')
st.write('This app predicts values using a simple Linear Regression model')

# Create input field
input_value = st.number_input('Enter a number (between 0 and 10):', 
                             min_value=0.0, 
                             max_value=10.0, 
                             value=5.0)

# Make prediction when button is clicked
if st.button('Predict'):
    # Reshape input for prediction
    X_new = np.array([[input_value]])

    # Get prediction and convert to float
    prediction = float(model.predict(X_new)[0])  # Convert numpy array to float

    # Display prediction
    st.success(f'Predicted value: {prediction:.2f}')

# Add explanation
st.markdown("""
### How it works
1. The model was trained on randomly generated data
2. It learns the relationship: y ≈ 2x + 1
3. Enter any number between 0 and 10 to get a prediction
""")
