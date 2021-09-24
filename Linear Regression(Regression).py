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

# Plotting the regression line
d=x*lg.coef_+lg.intercept_
# Plotting for the test data
plt.plot(x,y,'r*')
plt.xlabel('Study Hours')
plt.ylabel('Student Scores')
plt.title('Scores vs Hours')
plt.grid(alpha=0.3)
plt.plot(x,d)
plt.show()
