# employee-salary-prediction
Employee Salary Prediction

Overview

This capstone project develops a Streamlit web application for predicting employee salaries using a Gradient Boosting Regressor model. The application enables users to predict salaries for individual employees by inputting details such as Age, Department, Job Title, and Experience Level or perform batch predictions on uploaded CSV datasets. It includes data preprocessing, model training, evaluation, and visualizations to assess model performance and feature importance. The project is deployed using Streamlit and ngrok for public access, providing an interactive and user-friendly interface.

Problem Statement

The objective of this project is to build a machine learning model to accurately predict employee salaries based on features such as Age, Department, Job Title, and Experience Level. Organizations often struggle to determine fair and competitive salaries due to varying factors influencing compensation. This project addresses the challenge by developing a predictive model that provides reliable salary estimates, helping HR departments make informed decisions. The solution aims to be accessible via a web interface, allowing both single and batch predictions. The model leverages historical employee data to ensure accurate and data-driven salary predictions.

System Development Approach

The system is developed using a structured methodology, combining data preprocessing, machine learning, and web deployment. Below is the approach:
System Requirements
Hardware: Standard computer with internet access (Google Colab recommended for development).
Software:
Python 3.8+
Google Colab for development and testing.
Streamlit for the web application.
ngrok for public deployment.


Dataset: 

A CSV file (employees.csv) containing columns: Employee ID (optional), Name (optional), Age, Department, Job Title, Experience Level, and Salary.

Libraries Required:

streamlit: For building the web application.
pandas: For data manipulation and analysis.
numpy: For numerical computations.
scikit-learn: For machine learning model development and evaluation.
matplotlib and seaborn: For data visualizations.
pyngrok: For creating a public URL to access the Streamlit app.

Algorithm & Deployment:


The project follows a step-by-step procedure to develop and deploy the salary prediction system:

Install Dependencies:

Install required Python libraries using pip: streamlit, pyngrok, pandas, numpy, scikit-learn, matplotlib, and seaborn.


Upload Dataset:

Upload the employees.csv dataset to the project environment (e.g., Google Colab).
The dataset should include columns: Age, Department, Job Title, Experience Level, and Salary.


Data Preprocessing:

Load the dataset using pandas.
Handle missing values by dropping rows with null entries.
Encode categorical variables (Department, Job Title, Experience Level) using LabelEncoder.
Standardize the Age column using StandardScaler.
Split features (X) and target (Salary, y) for model training.


Model Training:

Split the dataset into training (80%) and testing (20%) sets using train_test_split.
Train a Gradient Boosting Regressor model (n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42).


Model Evaluation:

predict salaries on the test set.
Calculate Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score to evaluate model performance.


Visualizations:

Generate a scatter plot of actual vs. predicted salaries.
Create a bar plot to display feature importance.


Streamlit Application:

Develop a Streamlit app with two pages: "Home" for single predictions and "Batch Prediction" for processing CSV files.
Implement custom CSS and a Streamlit theme for a polished UI.
Allow users to input employee details or upload a CSV file for predictions.


Deployment:

Configure ngrok with an authentication token to create a public URL.
Run the Streamlit app and expose it via ngrok for public access.



Result

The project delivers a functional Streamlit application with the following outputs:


Model Performance Metrics:

Displays MSE, RMSE, and R² Scoreuencias de predicciones y métricas del modelo.

Actual vs. Predicted Salaries Plot:

A scatter plot showing the relationship between actual and predicted salaries.

Feature Importance Plot:

A bar plot illustrating the importance of each feature in the model.

Single Employee Prediction:

User inputs Age, Department, Job Title, and Experience Level to predict a salary.

Batch Prediction Output:

Uploaded CSV file processed to generate salary predictions, displayed in a table with a downloadable CSV.


Conclusion:

The Employee Salary Prediction project successfully delivers a robust and user-friendly solution for predicting employee salaries using a Gradient Boosting Regressor model. The model demonstrates strong predictive performance, as evidenced by the MSE, RMSE, and R² metrics, providing reliable salary estimates based on employee attributes. The Streamlit application enhances accessibility by offering an intuitive interface for both individual and batch predictions. Challenges during implementation included handling categorical encoding errors and ensuring dataset compatibility for batch predictions. Potential improvements include incorporating additional features (e.g., years of experience, education level) and experimenting with other algorithms like Random Forest or Neural Networks to enhance accuracy.

Future Scope:

Enhanced Features: Include additional employee attributes (e.g., education, certifications) to improve prediction accuracy.
Model Optimization: Experiment with hyperparameter tuning or alternative models (e.g., XGBoost, Deep Learning) for better performance.
Real-Time Data Integration: Incorporate APIs to fetch real-time salary data for model updates.
Mobile Optimization: Optimize the Streamlit app for mobile devices to improve accessibility.
Cloud Deployment: Deploy the application on cloud platforms like AWS or Heroku for scalability and reliability.


Installation:

Clone the repository:git clone https://
cd employee-salary-prediction


Install dependencies:pip install streamlit pyngrok pandas numpy scikit-learn matplotlib seaborn


Run the application:streamlit run employee_salary_prediction_streamlit.py

