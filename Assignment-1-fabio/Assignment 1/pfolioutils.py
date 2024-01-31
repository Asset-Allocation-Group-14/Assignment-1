import numpy as np
from scipy.optimize import minimize
import pandas as pd

import matplotlib.pyplot as plt

def weight_cons(weights):
    return np.sum(weights) - 1


def base_pfolio_optimizer(func, returns, short = False):   

    weight_costraint =  (0,1)

    if short:
        weight_costraint = (-1,1)

    bounds_lim = [weight_costraint for _ in range(len(returns.columns))]
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
        weights = np.random.uniform(size=len(mean_returns))
        weights /= np.sum(weights)  # Ensure weights sum to 1

        portfolio_return, portfolio_volatility = calculate_portfolio_stats(weights, mean_returns, cov_returns)

        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility

    return results


def calculate_stats(pfolio, is_ret = True):
    if not is_ret:
        pfolio = pfolio.pct_change().dropna()
    mu = pfolio.mean()*250
    sigma = pfolio.std()* np.sqrt(250)
    sharpe = mu/sigma
    return (mu, sigma, sharpe)



def evaluate_strategies(strategies):
    print("Max Sharpe Portfolio:")
    print("   - Return:", calculate_stats(strategies['Max_Sharpe_performance'])[0])
    print("   - Standard Deviation (Risk):", calculate_stats(strategies['Max_Sharpe_performance'])[1])
    print("   - Sharpe Ratio:", calculate_stats(strategies['Max_Sharpe_performance'])[2])
    print("\nMin Variance Portfolio:")
    print("   - Return:", calculate_stats(strategies['Min_Var_Sharpe_performance'])[0])
    print("   - Standard Deviation (Risk):", calculate_stats(strategies['Min_Var_Sharpe_performance'])[1])
    print("   - Sharpe Ratio:", calculate_stats(strategies['Min_Var_Sharpe_performance'])[2])
    print("\nMkt Cap Weight Portfolio:")
    print("   - Return:", calculate_stats(strategies['Mkt_Cap_performance'])[0])
    print("   - Standard Deviation (Risk):", calculate_stats(strategies['Mkt_Cap_performance'])[1])
    print("   - Sharpe Ratio:", calculate_stats(strategies['Mkt_Cap_performance'])[2])


def plot_weight_matrices(weight_matrix, xlabs, title):

    # Plotting the heatmap of the difference matrix
    plt.imshow(weight_matrix, cmap='RdYlGn', interpolation='none')
    # Add annotations
    for i in range(len(weight_matrix)):
        for j in range(len(weight_matrix[0])):
            plt.text(j, i, f'{weight_matrix[i, j]:.2f}', ha='center', va='center', color='black')

    plt.colorbar(label='Difference')
    plt.title(title)
    plt.xlabel('Indices')
    plt.ylabel('Years')
    plt.xticks(range(len(weight_matrix[0])), labels=[s.split()[0] for s in xlabs], fontsize=9)
    plt.yticks(range(len(weight_matrix)), labels=[str(year) for year in range(2013, 2023)])
    plt.show()