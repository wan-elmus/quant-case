import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: Collect historical data
df = pd.read_csv('oil_prices.csv')

# Step 2: Fit a polynomial equation to the historical data
X = df['Year'].values.reshape(-1, 1)
y = df['Price'].values.reshape(-1, 1)
poly = PolynomialFeatures(degree=5)  # use polynomial features up to degree 5
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Step 3: Refine the model using cross-validation
r2_scores = []
degrees = range(1, 11)  # try polynomial degrees up to 10
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

# Step 4: Use the model to make predictions
future_years = np.array([2023, 2024, 2025]).reshape(-1, 1)
future_X_poly = poly.transform(future_years)
future_y_pred = regressor.predict(future_X_poly)
print(future_y_pred)
