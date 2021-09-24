# Simple Linear Regression - Test and Training Set
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('2019happiness.csv')

X = np.array(data[['GDP per capita']]) #need double brackets or you get an error
Y = np.array(data[['Score']])#double brackets gives you a dataframe (predicting variable)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) # 30% test size

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

score = regressor.score(X_test, Y_test)
print(score)

for x in range(len(y_pred)):
    print('Predicted Value Y: ', y_pred[x], "Inputed values (x then y):", X_test[x], Y_test[x])

print('Coefficient: ', regressor.coef_)
print('Intercept: ', regressor.intercept_)





#### Plotting the Training set results
plt.scatter(X_train, Y_train, color ='red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Happiness vs. GDP per capita(Training set)')
plt.xlabel('GDP per capita')
plt.ylabel('Happiness')
plt.grid()
plt.show()

#### Plotting the Test set results
plt.scatter(X_test, Y_test, color ='red')# plot red dots
plt.plot(X_train, regressor.predict(X_train), color = 'blue')# plot a blue regression line
plt.title('Happiness vs. GDP per capita(Test set)')
plt.xlabel('GDP per capita')
plt.ylabel('Happiness')
plt.grid()
plt.show()