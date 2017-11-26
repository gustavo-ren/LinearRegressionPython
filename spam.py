#Linear regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Importing data to be analysed
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Creating training/test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80, random_state=0)

#Creation of the LinearRegression object
regression = LinearRegression()
regression.fit(X_train, y_train)#fitting the splits

#Vector of predictions of dependent variables
y_pred = regression.predict(X_test)
print(y_pred)
print(y_test)

#Plotting a graphs
#Train split
plt.scatter(X_train, y_train, color="red")#Real values for training, in form os dots
plt.plot(X_train, regression.predict(X_train), color="black")# Predicted values from the model, in form of a line
plt.title("Salary prediction (Trainig results)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

#Test split
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regression.predict(X_train), color="black")# Using the model generated
plt.title("Salary prediction (Test results)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()
