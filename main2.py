import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split


data = pd.read_csv(r'2019happiness.csv')
#print(data)

X = data['GDP per capita']
Y = data['Score']
#print(X)
#print(Y)

slope, intercept, r, p, std_err = stats.linregress(X, Y)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, X)) #linear regression model
print(slope, intercept, r, p)

# plotting the scatter plot and scatter
plt.xlabel('GDP per capita')
plt.ylabel('Happiness Score')
plt.title('Happiness Score and GDP per capita')

plt.scatter(X,Y) #makes a scatter plot of the data
plt.plot(X, mymodel)
plt.grid()
plt.show()



'''data2015 = data[data.Year == 2015]
print(data2015)

X = data2015[' BMI ']
Y = data2015['Life expectancy ']

print(X)
print(Y)

plt.xlabel('BMI')
plt.ylabel('Life Expectancy(Age)')
plt.title('How BMI affects your life expectancy')

plt.scatter(X,Y)
plt.grid()
plt.show()
'''
