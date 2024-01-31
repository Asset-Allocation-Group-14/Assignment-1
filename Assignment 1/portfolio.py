import numpy as np
import pfolioutils
import matplotlib.pyplot as plt
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting

class Portfolio_Optimizer:
    def __init__(self, returns: pd.DataFrame, short = False):
        self.returns = returns
        self.cov = returns.cov()
        self.mu = returns.mean()
        self.min_variance_weights = self.minimum_variance_weights(short)
        self.max_sharpe_weights = self.maximum_sharpe_weights(short)
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
    
    def minimum_variance_weights(self, short = False):
        optimal = pfolioutils.base_pfolio_optimizer(self.find_port_std, self.returns, short) 
        return optimal

    def maximum_sharpe_weights(self, short = False):
        optimal = pfolioutils.base_pfolio_optimizer(self.sharpe_func, self.returns, short) 
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
        plt.figure(figsize=(6, 3.6))

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

    def plot_efficient_frontier(self, short = False):
        weight_bounds = (0, 1)
        
        if weight_bounds:
            weight_bounds = (-1,1)

        mu = self.mu*252
        cov= self.cov*252
        ef = EfficientFrontier(mu, cov, weight_bounds=(-1, 1))

        fig, ax = plt.subplots()
        ef_max_sharpe = ef.deepcopy()
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

        # Find the tangency portfolio
        ef_max_sharpe.max_sharpe(risk_free_rate=0.00)
        ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
        ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
        if short:
            # Find the min_var portfolio
            plt.scatter(0.16582570142434142, 0.09243536876523031, s=50, c="y", marker = 'o', label="Min Variance")

            # Find the Equal Weight Portfolio
            plt.scatter(0.22268786570377302, 0.06216202345229385, s=100, c="b", marker = 'p', label="Equal Weight", zorder = 2)

            # Find the Mkt Cap Weighted portfolio
            plt.scatter(0.19599724489887774, 0.08969246261501455, s=100, c="m" , marker = 's', label="Mkt Cap Weighted",zorder = 2)
        else:
            # Find the min_var portfolio
            plt.scatter(0.1714903805307943, 0.09390923670506174, s=50, c="y", marker = 'o', label="Min Variance")

            # Find the Equal Weight Portfolio
            plt.scatter(0.22268786570377302, 0.06216202345229385, s=50, c="b", marker = 'p', label="Equal Weight", zorder = 2)

            # Find the Mkt Cap Weighted portfolio
            plt.scatter(0.19599724489887774, 0.08969246261501455, s=50, c="m" , marker = 's', label="Mkt Cap Weighted",zorder = 2)


        # Output
        ax.set_title("Efficient Frontier in long short senario")
        ax.legend()
        plt.tight_layout()
        plt.figure(figsize=(8, 4.8))
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
            print("   - Sharpe Ratio:", max_sharpe_ret/max_sharpe_std)

            print("\nMin Variance Portfolio:")
            print("   - Return:", min_var_ret)
            print("   - Standard Deviation (Risk):", min_var_std)
            print("   - Sharpe Ratio:", min_var_ret/min_var_std)

            print("\nEqual Weight Portfolio:")
            print("   - Return:", equal_w_ret)
            print("   - Standard Deviation (Risk):", equal_w_std)
            print("   - Sharpe Ratio:", equal_w_ret/equal_w_std)


        if custom_weights:
            for key, weights in custom_weights.items():
                custom_ret, custom_std = self.calculate_portfolio_stats(np.array(weights))
                
                print(f"\n{key}:")
                print("   - Return:", custom_ret)
                print("   - Standard Deviation (Risk):", custom_std)
                print("   - Sharpe Ratio:", custom_ret/custom_std)
    
    def print_all_weights(self):
        data = {'Indices': list(self.returns.columns), 'Min Variance Portfolio Weights': self.min_variance_weights, 'Max Sharpe Portfolio Weights': self.max_sharpe_weights}
        df = pd.DataFrame(data)

        print(df)





def get_rolling_weights(returns_df, weight_df):
    rw_returns_df = returns_df.loc['01/04/2008':]


    rw_returns_df.index = pd.to_datetime(rw_returns_df.index)


    # Initialize empty DataFrames to store performance values
    min_var_df = pd.DataFrame(index=rw_returns_df.index, columns=['Min_Var_Sharpe_performance'])
    max_sharpe_df = pd.DataFrame(index=rw_returns_df.index, columns=['Max_Sharpe_performance'])
    mkt_cap_df = pd.DataFrame(index=rw_returns_df.index, columns=['Mkt_Cap_performance'])

    min_var_weights_df = []
    max_sharpe_weights_df = []
    mkt_cap_weights_df = []

    # Iterate through each rolling window
    for year_start in range(2012, 2022):  # Adjust the start year accordingly
        # Define the start and end dates for the current window
        start_date = rw_returns_df.index[rw_returns_df.index.year == year_start - 4].min()
        end_date = rw_returns_df.index[rw_returns_df.index.year == year_start].max()

        # Extract the data for the previous 5 years
        prev_window_data = rw_returns_df.loc[start_date:end_date]

        # Calculate weights for the previous 5 years
        tmp_portfolio = Portfolio_Optimizer(prev_window_data)

        min_var_weights = tmp_portfolio.min_variance_weights
        min_var_weights_df.append(min_var_weights)
        
        max_sharpe_weights = tmp_portfolio.max_sharpe_weights
        max_sharpe_weights_df.append(max_sharpe_weights)


        mkt_cap_weights_df.append(np.array(weight_df['MktCap%'])
    )

        # Extract the data for the current year
        current_window_data = rw_returns_df.loc[start_date:end_date]

        # Calculate performance for the current year and store in DataFrames
        min_var_df.loc[start_date:end_date, 'Min_Var_Sharpe_performance'] = np.dot(current_window_data, min_var_weights)
        max_sharpe_df.loc[start_date:end_date, 'Max_Sharpe_performance'] = np.dot(current_window_data, max_sharpe_weights)
        mkt_cap_df.loc[start_date:end_date, 'Mkt_Cap_performance'] = np.dot(current_window_data, weight_df['MktCap%'])


    # Optionally, concatenate the DataFrames horizontally if needed
    portfolio_data = pd.concat([max_sharpe_df, min_var_df, mkt_cap_df], axis=1)

    # Filter data from 2013 onwards
    portfolio_data = portfolio_data.loc['2013':]

    return portfolio_data, mkt_cap_weights_df, max_sharpe_weights_df, rw_returns_df
