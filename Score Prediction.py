import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

url = "http://bit.ly/w-data"
student_data = pd.read_csv(url)
Study_Hours=np.asarray(student_data['Hours'])
Student_Scores=np.asarray(student_data['Scores'])
print(student_data)

x = student_data.iloc[:, :-1].values
y = student_data.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
lg=linear_model.LinearRegression()
lg.fit(x_train,y_train)
print('Training Done')

print(x_test)  # Testing data - In Hours
y_pred = lg.predict(x_test) # Predicting the scores

# Comparing Actual vs Predicted Scores
difference = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(difference)

# Testing your own data
Hrs= input("Enter the Hours: ")
Predicted_Score = lg.predict([[Hrs]])
print("No of Hours = {}".format(Hrs))
print("Predicted Score = {}".format(Predicted_Score[0]))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root mean squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
