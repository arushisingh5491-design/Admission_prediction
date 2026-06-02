# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# load the data
data = pd.read_csv("C:\\Users\\Admin\\Documents\\admission_prediction_200_records.csv")
print(data.head())

# data preprocessing
print(data.shape)
print(data.info())
print(data.describe())
print(data.isnull().sum())
data.duplicated().sum()

# x and y

x = data.drop("Chance_of_Admit",axis=1)
y = data['Chance_of_Admit']
# data splitting
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    test_size=0.3,random_state=42)
# print(x_train.shape,x_test.shape)
# print(y_train.shape, y_test.shape)

# scaling the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lr_model = LinearRegression()
lr_model.fit(x_train,y_train)

y_pred_lr = lr_model.predict(x_test)

print("MSE : ",mean_squared_error(y_test,y_pred_lr))
print("MAE : ",mean_absolute_error(y_test,y_pred_lr))
print("R2 score : ",r2_score(y_test,y_pred_lr))
