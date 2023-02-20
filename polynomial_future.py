import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Collect historical data
df = pd.read_csv('oil_prices.csv')
oilprice_data = pd.read_csv('oil_prices.csv')

#variable from oilprice_data
r = oilprice_data['r']    #risk-free interest rate

# Fit a polynomial equation to the historical data
X = df['Year'].values.reshape(-1, 1)
y = df['Average Closing Price'].values.reshape(-1, 1)
poly = PolynomialFeatures(degree=5)  # use polynomial features up to degree 5
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
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
