# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:07:47 2024

@author: ASUS
"""

import yfinance as yf
import pandas as pd

#Stock data input:
stock_data = yf.download ('NVO', period='5y')
print(stock_data.head())

#Linier regeression
from sklearn.linear_model import LinearRegression
import numpy as np

stock_data['Date'] = pd.to_datetime(stock_data.index). map(pd.Timestamp.toordinal)
X = stock_data['Date'].values.reshape(-1,1)
y = stock_data['Close'].values

#Fit a linier regression model
model = LinearRegression()
model.fit(X, y)

future_dates = np.array([X[-1] + i for i in range (365)]).reshape(-1, 1)
predicted_prices = model.predict(future_dates)

#Data plot
import matplotlib.pyplot as plt

plt.plot(stock_data.index, stock_data['Close'], label='Actual Price')
plt.plot(stock_data.index, model.predict(X), label='Trend (Interpolation)', linestyle='--')
future_dates = pd.to_datetime([pd.Timestamp.fromordinal(int(date[0])) for date in future_dates])
plt.plot(future_dates, predicted_prices, label='Prediction (1 year ahead)', linestyle=':')

plt.legend()
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()