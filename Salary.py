
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
import pickle


dataset = pd.read_csv(r"D:\Arati_Project\Multi-Predictor App\Data Set\emp_sal.csv")


X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, 2].values


#svm model
from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly',degree=5, gamma='scale')
svr_regressor.fit(X,y)

svr_model_pred = svr_regressor.predict([[6.5]])
print(svr_model_pred)


#knn  model
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model = KNeighborsRegressor(n_neighbors=5,weights='distance',p=1)
knn_reg_model.fit(X,y)

knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred)

# Save the trained model to disk
filename = 'salary.pkl'
with open(filename, 'wb') as file:
    pickle.dump(knn_reg_model, file)
print("Model has been pickled and saved as knn.pkl")

import os
print(os.getcwd())