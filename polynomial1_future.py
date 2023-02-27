import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Collect historical data
# df = pd.read_csv('oil_prices.csv')

oil_prices = {'Year': [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000, 1999],
    'Price': [77.90, 94.53, 68.17, 39.68, 56.99, 65.23, 50.80, 43.29, 48.66, 93.17, 97.98, 94.05, 94.88, 79.48, 61.95, 99.67, 72.34, 66.05, 56.64, 41.51, 31.08, 26.19, 25.98, 30.38, 19.35]}

oilprice_data = {'Year': [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000, 1999],
    'K': [77.90, 94.53, 68.17, 39.68, 56.99, 65.23, 50.80, 43.29, 48.66, 93.17, 97.98, 94.05, 94.88, 79.48, 61.95, 99.67, 72.34, 66.05, 56.64, 41.51, 31.08, 26.19, 25.98, 30.38, 19.35],
    'S': [80.26, 76.08, 47.62, 61.17, 46.31, 60.37, 52.36, 36.81, 52.72, 95.14, 93.14, 102.96, 91.59, 81.52, 46.17, 99.64, 60.77, 63.11, 42.16, 33.71, 31.97, 21.13, 27.29, 25.56, 12.42],
    'year_high': [81.62, 123.70, 84.65, 63.27, 66.24, 77.41, 60.46, 54.01, 61.36, 107.95, 110.62, 109.39, 113.39, 91.48, 81.03, 145.31, 99.16, 77.05, 69.91, 56.37, 37.96, 32.68, 32.21, 37.22, 28.03],
    'year_low': [72.84, 71.59, 47.62, 11.26, 46.31, 44.48, 42.48, 26.19, 34.55, 53.45, 86.65, 77.72, 75.40, 64.78, 34.03, 30.28, 50.51, 55.90, 42.16, 32.49, 25.25, 18.02, 17.50, 23.91, 11.38],
'year_close':[77.05, 80.51, 75.21, 48.52, 61.14, 45.15, 60.46, 53.75, 37.13, 53.45, 98.17, 91.83, 98.83, 91.38, 79.39, 44.60, 95.95, 60.85, 61.06, 43.36, 32.51, 31.21, 19.96, 26.72, 25.76, 12.14, 17.65, 25.90, 19.54, 17.77, 14.19, 19.49, 19.15, 28.48, 21.84, 17.12, 16.74 ],
'r':[-4.30, 7.05, 55.01, -20.64, 35.42, -25.32, 12.48, 44.76, -30.53, -45.55, 6.90, -7.08, 8.15, 15.10, 78.00, -53.52, 57.68, 0.34, 40.82, 33.37, 4.17, 56.36, -25.30, 3.73, 112.19, -31.22, -31.85, 32.55, 9.96, 25.23, -27.19, 1.78, -32.76, 30.40, 27.57, 2.27, -6.64],
'T':[1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
'sigma':[-3.029525032, 19.51761346, 30.14522517, -54.15826613, 18.74012985, 7.450559558, -3.070866142, 14.96881497, -8.343608714, -2.114414511, 4.939783629, -9.473684211, 3.467537943, -2.566683442, 25.47215496, 0.030099328, 15.99391761, 4.451173354, 25.56497175, 18.79065285, -2.863577864, 19.32035128, -5.042340262, 15.86570112, 35.81395349, -20.73509015, -23.96894711, 10.35262206, 5.317417254, 15.58139535, -3.255561584, 5.587949466, -23.16620241, 6.726457399, 11.50712831, -11.27113338, 5.572916667]}


df = pd.DataFrame(oil_prices)

#variable from oilprice_data
r = oilprice_data['r']    #risk-free interest rate

# X= df['Year']    #Year
# y = df['Price']    #Price

# Fit a polynomial equation to the historical data
X = df['Year'].values.reshape(-1, 1)
y = df['Price'].values.reshape(-1, 1)
poly = PolynomialFeatures(degree=5)  # use polynomial features up to degree 5
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Remove dollar signs and commas from the data
# X_train = X_train.replace('[\$,]', '', regex=True).astype(float)
# y_train = y_train.replace('[\$,]', '', regex=True).astype(float)

# Fit model with cleaned data
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Refine the model using cross-validation
r2_scores = []
degrees = range(1, 11)  # polynomial degrees up to 10
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    r2_scores.append(r2_score(y_test, y_pred))
best_degree = degrees[np.argmax(r2_scores)]
poly = PolynomialFeatures(degree=best_degree)
X_poly = poly.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y)

# Use the model to make predictions
future_years = np.array([2024, 2025, 2026]).reshape(-1, 1)
future_X_poly = poly.transform(future_years)
future_y_pred = regressor.predict(future_X_poly)

# Calculate futures prices using the estimated polynomial equation
future_prices = future_y_pred.flatten()
futures_price_2024 = future_prices[0]
futures_price_2025 = future_prices[1]
futures_price_2026 = future_prices[2]

# Calculate the implied spot prices using the futures parity conditions
spot_price_2024 = futures_price_2024 / (1 + r)**(2024 - df['Year'].iloc[-1])
spot_price_2025 = futures_price_2025 / (1 + r)**(2025 - df['Year'].iloc[-1])
spot_price_2026 = futures_price_2026 / (1 + r)**(2026 - df['Year'].iloc[-1])

print('Estimated futures prices:', future_prices)
print('Implied spot prices:')
print('2024:', spot_price_2024)
print('2025:', spot_price_2025)
print('2026:', spot_price_2026)