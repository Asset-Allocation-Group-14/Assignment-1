import numpy as np
from scipy.optimize import minimize

def weight_cons(weights):
    return np.sum(weights) - 1


def base_pfolio_optimizer(func, returns):   

    bounds_lim = [(0, 1) for _ in range(len(returns.columns))]
    init = [1 / len(returns.columns) for _ in range(len(returns.columns))]
    constraint = {'type': 'eq', 'fun': weight_cons}

    optimal = minimize(fun=func,
                        x0=init,
                        bounds=bounds_lim,
                        constraints=constraint,
                        method='SLSQP')

    return list(optimal['x'])

def calculate_portfolio_stats(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights, mean_returns) * 250
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(250)
    return portfolio_return, portfolio_stddev


def generate_random_portfolios(num_portfolios, mean_returns, cov_returns):
    results = np.zeros((2, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)  # Ensure weights sum to 1

        portfolio_log_return, portfolio_volatility = calculate_portfolio_stats(weights, mean_returns, cov_returns)

        results[0, i] = portfolio_log_return
        results[1, i] = portfolio_volatility

    return results