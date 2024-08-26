# Heart Disease Prediction Web Application

## Objective
The primary objective of this project is to develop a web application that accurately predicts the likelihood of heart disease based on user-provided health metrics. This tool aims to raise awareness and assist users in understanding their heart health risks, enabling them to take proactive measures.

## Overview
This web application predicts the likelihood of heart disease based on user inputs. It uses a trained `RandomForestClassifier` to provide real-time predictions with a user-friendly interface. The application is built with Python, Streamlit, and Scikit-learn.

## Features
- **User-Friendly Interface:** An intuitive form for entering health data such as age, cholesterol levels, and exercise-induced angina.
- **Machine Learning Model:** A pre-trained `RandomForestClassifier` ensures accurate and reliable predictions.
- **Real-Time Predictions:** Instant feedback on heart disease risk is provided upon form submission.
- **Data Preprocessing:** Includes categorical variable encoding and numerical feature scaling to optimize model performance.

## How It Works
1. **Input Data:** Users input their health information via the web interface.
2. **Data Processing:** The input data is encoded and scaled using pre-trained encoders and a scaler.
3. **Prediction:** The processed data is fed into a pre-trained machine learning model to predict heart disease likelihood.
4. **Output:** The app displays whether the user is at risk for heart disease or not.

## Technologies Used
- **Python**
- **Streamlit**
- **Scikit-learn**
- **Pandas**
- **Joblib**

## Project Structure
```bash
.
├── deploy.py                # Main application script
├── model.pkl                # Trained machine learning model
├── encoder.pkl              # Encoders for categorical variables
├── requirements.txt         # List of Python dependencies
└── README.md                # Project documentation
