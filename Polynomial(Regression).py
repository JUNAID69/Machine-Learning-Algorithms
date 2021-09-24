import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

#training the LR model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#training the polynomial regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
pol_reg=PolynomialFeatures(degree=4)
x_poly=pol_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)


#visualizing the linear regression results
# plt.scatter(x,y,color='r')
# plt.plot(x,lin_reg.predict(x),color="b")
# plt.title('Truth or Bluff (LR)')
# plt.xlabel('Position Level')
# plt.ylabel('Salary ')
# plt.show()

plt.scatter(x,y,color='r')
plt.plot(x,lin_reg_2.predict(pol_reg.fit_transform(x)),color="b")
plt.title('Truth or Bluff (PLR)')
plt.xlabel('Position Level')
plt.ylabel('Salary ')
plt.show()
