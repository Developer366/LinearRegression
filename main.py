import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'Life expectancy.csv')

#get data from each country
USAdata = data[data.Entity == "United States"]
CANADAdata = data[data.Entity == "Canada"]
MEXICOdata = data[data.Entity == "Mexico"]

#set plots
plt.plot(USAdata['Year'], USAdata['Life expectancy'], linestyle="-", color = 'Red')

#plt.plot(USAdata['Year'].loc[] >"1849", USAdata['Life expectancy'], linestyle="-", color = 'Red')
plt.plot(CANADAdata['Year'], CANADAdata['Life expectancy'], linestyle="--", color = 'b')
plt.plot(MEXICOdata['Year'], MEXICOdata['Life expectancy'], linestyle="-.", color="Green")

#set the labels for plot
plt.xlabel('Years')
plt.ylabel('Life Expectancy(Age)')
plt.title('Life Expectancy in North American countries')
plt.legend(['USA', 'CANADA', 'MEXICO'])
plt.grid()
plt.show()

#print(USAdata)
#print(CANADAdata)
#print(MEXICOdata)



