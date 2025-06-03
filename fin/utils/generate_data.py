import numpy as np


def simulate_stock_prices(n_steps, S0=100, mu=0.001, sigma=0.02, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = 1
    prices = [S0]
    for _ in range(n_steps):
        dS = prices[-1] * (mu * dt + sigma * np.random.randn() * np.sqrt(dt))
        prices.append(prices[-1] + dS)
    return prices

if __name__ == '__main__':
    prices = simulate_stock_prices(n_steps=10)
    print(prices)