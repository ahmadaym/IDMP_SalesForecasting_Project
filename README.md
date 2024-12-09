# SETUP README
## Description and Guide to Sales Forecasting App: Requirements, Setup, and Execution

**Authors**: Ayman Mushtaq Ahmad, Amisha Tiwari, Ronhit Neema  
**Date**: December 9, 2024

---

## Table of Contents
1. [Sales Forecasting App](#sales-forecasting-app)
    - [Description](#description)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
    - [Clone from GitHub](#clone-from-github)
    - [Download and Extract Project Zip](#download-and-extract-project-zip)
    - [Set Up a Virtual Environment (Recommended)](#set-up-a-virtual-environment-recommended)
    - [Install Dependencies](#install-dependencies)
5. [Usage](#usage)
    - [Run the Application](#run-the-application)
    - [Access the Application](#access-the-application)
6. [User Instructions](#user-instructions)
7. [Project Structure](#project-structure)
8. [Dependencies](#dependencies)
9. [Troubleshooting](#troubleshooting)
10. [Contact](#contact)

---

## Sales Forecasting App

### Description
The Sales Forecasting App is an interactive web application designed to assist businesses in predicting future sales based on historical data. Utilizing advanced data processing techniques with Pandas and NumPy, machine learning models from Scikit-learn, and time series analysis with ARIMA, the application offers comprehensive Exploratory Data Analysis (EDA), data cleaning, model training, and performance evaluation. Leveraging Dash and Flask for the web interface, users can seamlessly upload their sales data, visualize insights, train multiple models, and compare their performances through intuitive visualizations.

## Features
- **Data Upload**: Easily upload CSV files containing sales data for analysis.
- **Exploratory Data Analysis (EDA)**: Generate comprehensive insights including data summaries, missing values analysis, and statistical overviews.
- **Data Cleaning**: Handle missing values, outliers, and perform feature engineering to prepare data for modeling.
- **Model Training**: Train multiple machine learning models, including Linear Regression, Random Forest, XGBoost, and ARIMA for time series forecasting.
- **Model Comparison**: Compare models based on key performance metrics such as R², RMSE, MAE, and MAPE.
- **Interactive Visualizations**: Visualize data distributions, correlation heatmaps, feature importances, residuals, and more using Plotly.
- **User-Friendly Web Interface**: Navigate through the data analysis and modeling process with an intuitive Dash and Flask-based interface.
- **Downloadable Reports**: Export predictions and summary metrics for further analysis or reporting.

## Prerequisites
Before running the application, ensure you have the following installed:
- **Operating System**: Windows, macOS, or Linux.
- **Python**: Version 3.8 or higher.
- **Java**: JDK 8 or higher (required for certain machine learning libraries like XGBoost).

## Installation

### Clone from GitHub
To clone the repository directly from GitHub, ensure you have `git` installed on your system. Then, open your terminal or command prompt and run the following commands:

git clone https://github.com/ahmadaym/IDMP_SalesForecasting_Project.git
cd IDMP_SalesForecasting_Project

### Download and Extract Project Zip
Alternatively, you can download the project repository as a ZIP file from GitHub. After downloading, extract the ZIP file to your local machine.

### Set Up a Virtual Environment (Recommended)
Create and activate a virtual environment to manage dependencies:

For Linux/macOS
python3 -m venv venv
source venv/bin/activate

For Windows
python -m venv venv
venv\Scripts\activate

### Install Dependencies
Install the required Python packages using pip:
pip install -r requirements.txt

## Usage
### Run the Application
Start the Flask application by running:
python app.py

### Access the Application
Open your web browser and navigate to http://127.0.0.1:8050/.

## User Instructions
1.Upload Data: Navigate to the Upload page to upload your CSV file containing sales data.
2.Exploratory Data Analysis: Proceed to the EDA page to view data summaries, missing values, and initial visualizations.
3.Data Cleaning: Move to the Cleaning page to handle missing values, outliers, and perform feature engineering.
4.Model Training: Go to the Modeling page to train multiple machine learning models and compare their performances.
5.View Results: Finally, access the Results page to review model performances and download predictions or summary metrics.

## Project Structure
The Project folder is organized as follows:

app.py: Main application file to run the Flask and Dash server.
src/: Contains subdirectories for data processing and modeling.
data_process/data_processing.py: Scripts for data cleaning and EDA.
models/modeling.py: Scripts for model training and evaluation.
requirements.txt: List of Python dependencies.
README.pdf: Brief overview of project and setup instructions.
project_technical_report.pdf: Detailed technical report of the project.

## Dependencies
Ensure all dependencies in requirements.txt are installed in your virtual environment (venv).
Some important ones are listed below. If during execution, any seem missing, please use pip install to install them manually.

Flask
Dash
dash_bootstrap_components
pandas
numpy
scikit-learn
plotly
xgboost
lightgbm
catboost
pmdarima
shap
scipy
yaml

## Troubleshooting
Missing Dependencies: If you encounter ModuleNotFoundError, ensure all dependencies are installed using pip install -r requirements.txt.
Java Not Installed: Some machine learning libraries like XGBoost require Java. Ensure JDK 8 or higher is installed and JAVA_HOME is set correctly.
Port Already in Use: If http://127.0.0.1:8050/ is not accessible, ensure no other application is using port 8050 or modify the port in app.py.
Data Format Issues: Ensure the uploaded CSV contains the required columns, especially the target variable Item_Outlet_Sales.
For any other issues in running the application, please reach out to Ayman Ahmad (ahmad.ay@northeastern.edu).

## Contact
For assistance, contact ahmad.ay@northeastern.edu