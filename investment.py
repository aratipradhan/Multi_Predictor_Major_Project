# Backend: Train Multiple Linear Regression Model

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
data = pd.read_csv(r"D:\Arati_Project\Multi-Predictor App\Data Set\Investment.csv")

# Divide the dependent and independent variables
# X includes DigitalMarketing, Promotion, and Research
X = data.iloc[:, [0, 1, 2]].values  # Independent variables
y = data.iloc[:, -1].values         # Dependent variable (Profit)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Multiple Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train)



# Save the trained model and scaler for future use
with open('Investment_Model.pkl', 'wb') as model_file:
    pickle.dump(regressor, model_file)

with open('Investment_Scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Print confirmation
print("Model and Scaler have been successfully saved.")
