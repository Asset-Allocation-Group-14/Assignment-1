import numpy as np
from scipy.optimize import minimize
import pandas as pd

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

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


def plot_weight_matrices(weight_matrices, xlabs, titles):
    num_matrices = len(weight_matrices)

    fig, axes = plt.subplots(1, num_matrices, figsize=(15, 5), sharey=True)

    for i in range(num_matrices):
        # Plotting the heatmap of the weight matrix
        im = axes[i].imshow(weight_matrices[i], cmap='RdYlGn', interpolation='none')

        # Add annotations
        for j in range(len(weight_matrices[i])):
            for k in range(len(weight_matrices[i][0])):
                axes[i].text(k, j, f'{weight_matrices[i][j, k]:.2f}', ha='center', va='center', color='black')

        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Indices')
        axes[i].set_xticks(range(len(weight_matrices[i][0])), labels=[s.split()[0] for s in xlabs], fontsize=9)

    fig.colorbar(im, ax=axes, label='Difference')
    axes[0].set_ylabel('Years')
    axes[0].set_yticks(range(len(weight_matrices[0])), labels=[str(year) for year in range(2013, 2023)])
    plt.show()


def plot_rolling_performance(portfolio_data, rw_returns_df):
    performance_df = (1+ portfolio_data).cumprod() -1

    # Plot portfolio performance
    plt.figure(figsize=(8, 4.8))
    plt.plot(performance_df['Max_Sharpe_performance'], label='Max Sharpe Portfolio')
    plt.plot(performance_df['Min_Var_Sharpe_performance'], label='Min Variance Portfolio')
    plt.plot(performance_df['Mkt_Cap_performance'], label='Mkt Cap Weighted Portfolio')

    # Highlight each rebalancing point
    for year_start in range(2013, rw_returns_df.index.year.max()):
        rebalancing_date = rw_returns_df.index[rw_returns_df.index.year == year_start].max()
        plt.axvline(rebalancing_date, color='red', linestyle='--', linewidth=1, label='Rebalancing Point' if year_start == 2013 else '')

    plt.title('Portfolio Performance with Rebalancing Points')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Performance')
    plt.legend()
    plt.show()



def plot_acfs(logreturns_df, num_lags=50):

    # Get the number of columns in logreturns_df
    num_cols = len(logreturns_df.columns)

    # Calculate the number of rows needed to display all plots
    num_rows = (num_cols + 1) // 2  # Add 1 to round up if num_cols is odd

    # Create subplots with two columns
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, 4*num_rows))

    # Flatten the axes array to iterate over all subplots
    axes = axes.flatten()

    for i, col in enumerate(logreturns_df.columns):
        series = logreturns_df[col].dropna()
    
        # Plot ACF with significance lines
        plot_acf(series, lags=num_lags, ax=axes[i], alpha=0.05)
        
        # Add significance lines
        critical_value = 1.96 / (len(series) ** 0.5)  # 95% confidence interval
        axes[i].axhline(y=critical_value, linestyle='--', color='gray')
        axes[i].axhline(y=-critical_value, linestyle='--', color='gray')
        
        axes[i].set_ylim([-0.25, 0.25])
        axes[i].set_title(f'ACF for {col} with Significance Lines (95% Confidence)')
        axes[i].set_xlabel('Lag')
        axes[i].set_ylabel('Autocorrelation')

    # If there are extra subplots, remove them
    for j in range(num_cols, num_rows * 2):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()