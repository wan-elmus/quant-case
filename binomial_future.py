import numpy as np
from scipy.stats import norm

# Define variables
S = 100  # initial stock price
K = 110  # strike price
T = 1  # time to expiration (in years)
r = 0.05  # risk-free interest rate
sigma = 0.2  # volatility of the stock price

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
    option_values[N, i] = max(K - stock_price, 0)

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

# Use future parity condition to check if the put and call prices are consistent
future_parity_diff = put_price + S - call_price * np.exp(-r * T) - K * np.exp(-r * T)
if np.abs(future_parity_diff) < 1e-6:
    print('Future parity condition holds.')
else:
    print('Future parity condition does not hold.')

# Print results
print('Option price:', option_price)
print('Delta:', delta)
print('Gamma:', gamma)
print('Vega:', vega)
print('Put price:', put_price)
print('Call price (Black-Scholes):', call_price)
