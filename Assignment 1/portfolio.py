import numpy as np
import pfolioutils
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.interpolate import interp1d

class Portfolio_Optimizer:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.cov = returns.cov()
        self.mu = returns.mean()
        self.min_variance_weights = self.minimum_variance_weights()
        self.max_sharpe_weights = self.maximum_sharpe_weights()
        self.equal_weights = [1/len(returns.columns)]*len(returns.columns)
    
    def calculate_portfolio_stats(self, weights):
        portfolio_return = np.dot(weights, self.mu) * 250
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(self.cov, weights))) * np.sqrt(250)
        return portfolio_return, portfolio_stddev
        
    
    def find_port_std(self, weights):
        port_std = np.sqrt(np.dot(weights.T, np.dot(self.cov, weights))) * np.sqrt(250)
        return port_std

    def sharpe_func(self, weights):
        port_ret, port_std = self.calculate_portfolio_stats(weights)
        return -1 * port_ret / port_std
    
    def minimum_variance_weights(self):
        optimal = pfolioutils.base_pfolio_optimizer(self.find_port_std, self.returns) 
        return optimal

    def maximum_sharpe_weights(self):
        optimal = pfolioutils.base_pfolio_optimizer(self.sharpe_func, self.returns) 
        return optimal
    
    def plot_performance(self, prices: pd.DataFrame, custom_weights: dict = None, rolling_window = False):
        prices = (prices - prices.iloc[0])/prices.iloc[0]
        prices['date'] = pd.to_datetime(prices.index)
        prices.set_index('date', inplace=True)
        # Calculate cumulative returns for each portfolio
        cum_returns_min_variance = np.dot(prices, self.min_variance_weights)
        cum_returns_max_sharpe = np.dot(prices, self.max_sharpe_weights)
        cum_returns_equal_weights = np.dot(prices, self.equal_weights)



        # Plot cumulative returns
        plt.figure(figsize=(10, 6))

        if custom_weights:
            for key, weights in custom_weights.items():
                custom_returns = np.dot(prices, weights)
                plt.plot(prices.index, custom_returns, label=key)

        if not rolling_window:
            plt.plot(prices.index, cum_returns_min_variance, label='Min Variance Portfolio')
            plt.plot(prices.index, cum_returns_max_sharpe, label='Max Sharpe Portfolio')
            plt.plot(prices.index, cum_returns_equal_weights, label='Equal Weight Portfolio')
        
        plt.title('Portfolio Performances Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.show()

    def plot_efficient_frontier(self, num_portfolios = 1000):

        # pull results, weights from random portfolios
        random_portfolios = pfolioutils.generate_random_portfolios(num_portfolios, self.mu, self.cov)


        # Calculate returns and volatilities for specific portfolios
        max_sharpe_ret, max_sharpe_std = self.calculate_portfolio_stats(np.array(self.max_sharpe_weights))
        min_var_ret, min_var_std = self.calculate_portfolio_stats(np.array(self.min_variance_weights))
        equal_w_ret, equal_w_std = self.calculate_portfolio_stats(np.array(self.equal_weights))

        # Generate interpolation function
        tck, u = splprep(np.array([
                           [max_sharpe_std, min_var_std, equal_w_std],[max_sharpe_ret, min_var_ret, equal_w_ret]]), u=None, k=2, s=0)
        new_points = splev(np.linspace(0, 1, 100), tck)


        # Plot the efficient frontier
        plt.figure(figsize=(10, 6))
        plt.scatter(random_portfolios[1, :], random_portfolios[0, :], c=(random_portfolios[0, :]) / random_portfolios[1, :], cmap='viridis', marker='o', s=10, label='Random Portfolios')
        plt.scatter(max_sharpe_std, max_sharpe_ret, color='red', marker='*', s=100, label='Max Sharpe Portfolio')
        plt.scatter(min_var_std, min_var_ret, color='green', marker='*', s=100, label='Min Variance Portfolio')
        plt.scatter(equal_w_std, equal_w_ret, color='blue', marker='*', s=100, label='Equal Weight Portfolio')
        plt.plot(new_points[0], new_points[1], color='black', linestyle='-', linewidth=2, label='Smooth Curve')

        plt.title('Efficient Frontier with Random Portfolios and Specific Points')
        plt.xlim(0,0.8)
        plt.xlabel('Volatility (Log Returns)')
        plt.ylabel('Return (Log Returns)')
        plt.legend()
        plt.colorbar(label='Sharpe Ratio')
        plt.show()

        

    def performance_summary(self, custom_weights: dict = None, rolling_window=False):
        # Calculate performance metrics for portfolios
        max_sharpe_ret, max_sharpe_std = self.calculate_portfolio_stats(np.array(self.max_sharpe_weights))
        min_var_ret, min_var_std = self.calculate_portfolio_stats(np.array(self.min_variance_weights))
        equal_w_ret, equal_w_std = self.calculate_portfolio_stats(np.array(self.equal_weights))

        if not rolling_window:
            # Print or return the performance summary
            print("Max Sharpe Portfolio:")
            print("   - Return:", max_sharpe_ret)
            print("   - Standard Deviation (Risk):", max_sharpe_std)
            print("\nMin Variance Portfolio:")
            print("   - Return:", min_var_ret)
            print("   - Standard Deviation (Risk):", min_var_std)
            print("\nEqual Weight Portfolio:")
            print("   - Return:", equal_w_ret)
            print("   - Standard Deviation (Risk):", equal_w_std)

        if custom_weights:
            for key, weights in custom_weights.items():
                custom_ret, custom_std = self.calculate_portfolio_stats(np.array(weights))
                
                print(f"\n{key}:")
                print("   - Return:", custom_ret)
                print("   - Standard Deviation (Risk):", custom_std)


