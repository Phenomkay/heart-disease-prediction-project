**Heart Disease Prediction**

This project is a web application for predicting the likelihood of heart disease based on user input. Built using Python, Streamlit, and Scikit-learn, it leverages machine learning to provide real-time predictions based on key health metrics.

**Features**
User-Friendly Interface: The app provides an easy-to-use form for inputting health data such as age, cholesterol levels, and exercise-induced angina.
Machine Learning Model: A trained RandomForestClassifier is used for making predictions, ensuring accurate and reliable results.
Real-Time Predictions: Users receive instant feedback on their heart disease risk upon submitting the form.
Data Preprocessing: Includes encoding of categorical variables and scaling of numerical features to optimize model performance.

**How It Works**
Input Data: Users enter their health information (e.g., age, sex, cholesterol levels) via the web interface.
Data Processing: The input data is encoded and scaled using pre-trained encoders and a scaler.
Prediction: The processed data is fed into a pre-trained machine learning model to predict the likelihood of heart disease.
Output: The app displays whether the user is at risk for heart disease or not.

**Technologies Used**
Python
Streamlit
Scikit-learn
Pandas
Joblib

**Project Structure**
deploy.py: Main application script.
model.pkl: Trained machine learning model.
encoder.pkl: Encoders for categorical variables.
requirements.txt: List of Python dependencies.

**Contributing**
Contributions are welcome! Please fork this repository and submit a pull request with your improvements.
