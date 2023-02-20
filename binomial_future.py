import numpy as np
from scipy.stats import norm
import pandas as pd

# Read oilprices data from CSV file
oilprice_data = pd.read_csv('oil_prices.csv')

# Extract variables from options data
S = oilprice_data['S'][0]    #initial stock price
K = oilprice_data['K'][0]    #strike price
T = oilprice_data['T'][0]    #Time to expiration
r = oilprice_data['r'][0]    #risk-free interest rate
sigma = oilprice_data['sigma'][0]   #volatility of the stock price

# Create binomial lattice
N = 100  # number of time steps
dt = T / N  # length of time step
u = np.exp(sigma * np.sqrt(dt))  # up factor
d = 1 / u  # down factor
p = (np.exp(r * dt) - d) / (u - d)  # probability of up move

# Calculate option values at expiration
option_values = np.zeros((N+1, N+1))
for i in range(N+1):
    stock_price = S * u**(N-i) * d**i
    option_values[N, i] = max(stock_price - K, 0)

# Work backwards through the lattice to calculate option values at earlier times
for i in range(N-1, -1, -1):
    for j in range(i+1):
        option_values[i, j] = np.exp(-r * dt) * (p * option_values[i+1, j+1] + (1-p) * option_values[i+1, j])

# Calculate option price and Greeks using option values at time 0
option_price = option_values[0, 0]
delta = (option_values[1, 1] - option_values[1, 0]) / (S * u - S * d)
gamma = ((option_values[2, 2] - option_values[2, 1]) / (S * u**2 - S) - (option_values[2, 1] - option_values[2, 0]) / (S - S * d**2)) / ((S * u**2 - S * d**2) / 2)**2
vega = S * np.sqrt(T) * norm.pdf((np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T)))

# Use put/call parity to calculate the price of a put option
put_price = option_price + K * np.exp(-r * T) - S

# Use Black-Scholes formula to calculate the price of a call option
d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Calculate the price of the call option using the future parity condition
call_price_parity = put_price + S - K * np.exp(-r * T)

# Print results
print('Option price:', option_price)
print('Delta:', delta)
print('Gamma:', gamma)
print('Vega:', vega)
print('Put price:', put_price)
print('Call price (Black-Scholes):', call_price)
print('Call price (parity):', call_price_parity)
