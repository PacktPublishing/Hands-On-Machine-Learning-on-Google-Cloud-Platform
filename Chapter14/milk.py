import pandas as pd
import matplotlib.pyplot as plt
import numpy
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv('milk-production-pounds.csv',parse_dates=True, index_col='DateTime', 
                                                    names=['DateTime', 'Milk'], header=None)

print(data.info())
print(data.head())
print(data.describe())

data.plot()
plt.show()

X = [i for i in range(0, len(data))]
X = numpy.reshape(X, (len(X), 1))
y = data.values

LModel = LinearRegression()
LModel.fit(X, y)
print(LModel.intercept_,LModel.coef_)  

trend = LModel.predict(X)
plt.plot(y)
plt.plot(trend)
plt.show()

DecompDataAdd = seasonal_decompose(data, model='additive')

DecompDataAdd.plot()
plt.show()

SeasRemov= data-DecompDataAdd.seasonal
SeasRemov.plot()
plt.show()

DecompDataMult = seasonal_decompose(data, model='multiplicative')

DecompDataMult.plot()
plt.show()
