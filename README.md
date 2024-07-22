# Medical-Premium-Prediction
## Overview

This project involves building an insurance cost prediction application using machine learning. It predicts medical insurance costs based on various factors such as age, sex, BMI, number of children, smoking status, and region. The project comprises data exploration, model training, and deployment of the model via a Flask web application.

## Project Structure

### 1. Data Exploration and Preprocessing

- **Reading the Dataset**: The dataset is read into a pandas DataFrame to facilitate data manipulation and analysis.

- **Data Visualization**: The distribution of variables such as age, number of children, sex, smoking status, and region is visualized using seaborn plots. A box plot is used to visualize the distribution of insurance charges.

- **Handling Outliers**: Insurance charges above a certain threshold are treated as outliers and removed to prevent them from skewing the analysis and model training.

- **Encoding Categorical Variables**: Categorical variables like sex, smoker status, and region are converted into numerical values. Sex and smoker status are encoded as binary variables, while region is one-hot encoded.

- **Normalization**: The data is normalized to ensure that all features contribute equally to the model.

### 2. Model Training

- **Splitting the Data**: The dataset is split into training and testing sets to evaluate model performance.

- **Training Multiple Models**: Several regression models are trained, including Linear Regression, Decision Tree Regressor, and Random Forest Regressor. Each model is trained on the training set and evaluated on both the training and testing sets.

- **Evaluation Metrics**: The models are evaluated using R² score, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to measure their accuracy and error rates.

### 3. Model Deployment

- **Flask Web Application**: A Flask web application is created to allow users to input data and receive insurance cost predictions.

  - **Home Route**: The home route renders an HTML template with a form for users to input their details.

  - **Predict Route**: This route processes the form data, converts categorical inputs to numerical values, and makes a prediction using the trained model. The prediction is then displayed to the user.

  - **Model Loading**: The trained machine learning model is loaded using pickle to make predictions.

### Acknowledgements

- **Kaggle Insurance Dataset**: The dataset used in this project was sourced from Kaggle.
- **Scikit-learn**: The machine learning library used for model training.
- **Flask**: The web framework used for deploying the model.
